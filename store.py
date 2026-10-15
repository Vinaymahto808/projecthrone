"""
store.py - SQLite persistence for employees, attendance, shifts and audit.

Tables
------
employees(id, emp_code UNIQUE, name, department, face_key, password_hash,
          shift_id, weekoff_weekday, created_at)
attendance(id, employee_id, event_type IN|OUT, ts, score, source, note)
admins(id, username UNIQUE, password_hash, created_at)         # API admin login
devices(id, name UNIQUE, key_hash, created_at, last_seen)      # kiosk API keys
shifts(id, name UNIQUE, start_time, end_time, grace_min, work_hours)
holidays(date UNIQUE, name)                                    # paid holidays
audit_log(id, ts, actor_type, actor, action, detail, ip, prev_hash, hash)

`face_key` is the folder name under known_faces/ and the key inside
embeddings/face_db.pkl, so the face database and the employee record stay
linked. `password_hash` (PBKDF2-SHA256, salted) is optional and only used for
the employee self-service login that shows attendance history.

Shifts & reports
----------------
`shift_for_employee` falls back to config.DEFAULT_SHIFT when none is assigned.
`day_summary` / `monthly_report` implement the usual industry rules: grace
period before a punch counts as late, early leaving, overtime beyond the
shift's work_hours, week-offs and holidays instead of absences, and
cross-midnight (night) shifts attributed to the day the shift started.

Audit
-----
audit_log rows are hash-chained (hash = sha256(prev_hash + row fields)), so
deleting or editing history is detectable via `verify_audit_chain()`.
"""

from __future__ import annotations

import csv
import hashlib
import secrets
import sqlite3
import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator

import config

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_code         TEXT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    department       TEXT NOT NULL DEFAULT '',
    face_key         TEXT NOT NULL UNIQUE,
    password_hash    TEXT,
    shift_id         INTEGER,
    weekoff_weekday  INTEGER,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    event_type  TEXT NOT NULL CHECK (event_type IN ('IN', 'OUT')),
    ts          TEXT NOT NULL,
    score       REAL,
    source      TEXT NOT NULL DEFAULT 'kiosk',
    note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_attendance_emp_ts ON attendance(employee_id, ts);

CREATE TABLE IF NOT EXISTS admins (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS devices (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    key_hash   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen  TEXT
);
CREATE TABLE IF NOT EXISTS shifts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    start_time  TEXT NOT NULL,
    end_time    TEXT NOT NULL,
    grace_min   INTEGER NOT NULL DEFAULT 15,
    work_hours  REAL NOT NULL DEFAULT 8.0
);
CREATE TABLE IF NOT EXISTS holidays (
    date TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    ip         TEXT NOT NULL DEFAULT '',
    prev_hash  TEXT NOT NULL DEFAULT '',
    hash       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""

TS_FMT = "%Y-%m-%d %H:%M:%S"

# Weekday numbering for weekoffs: 0=Monday .. 6=Sunday (datetime.weekday()).
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        with _lock:
            yield con
            con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create tables if needed and apply light migrations. Safe to call repeatedly."""
    with _conn() as con:
        con.executescript(SCHEMA)
        emp_cols = {r["name"] for r in con.execute("PRAGMA table_info(employees)").fetchall()}
        if "password_hash" not in emp_cols:
            con.execute("ALTER TABLE employees ADD COLUMN password_hash TEXT")
        if "shift_id" not in emp_cols:
            con.execute("ALTER TABLE employees ADD COLUMN shift_id INTEGER")
        if "weekoff_weekday" not in emp_cols:
            con.execute("ALTER TABLE employees ADD COLUMN weekoff_weekday INTEGER")
        att_cols = {r["name"] for r in con.execute("PRAGMA table_info(attendance)").fetchall()}
        if "note" not in att_cols:
            con.execute("ALTER TABLE attendance ADD COLUMN note TEXT")
        # Seed the default shift so reports have something to grade against.
        if con.execute("SELECT COUNT(*) AS n FROM shifts").fetchone()["n"] == 0:
            d = config.DEFAULT_SHIFT
            con.execute(
                "INSERT INTO shifts (name, start_time, end_time, grace_min, work_hours) VALUES (?,?,?,?,?)",
                (d["name"], d["start_time"], d["end_time"], 15, float(d["work_hours"])),
            )


def _now() -> str:
    return datetime.now().strftime(TS_FMT)


def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(r) if r is not None else None


def public(emp: dict[str, Any] | None) -> dict[str, Any] | None:
    """Employee dict without secrets (safe to return from the API)."""
    if emp is None:
        return None
    return {k: v for k, v in emp.items() if k != "password_hash"} | {"has_password": bool(emp.get("password_hash"))}


# ------------------------------------------------------------------ employees
def create_employee(emp_code: str, name: str, department: str, face_key: str) -> dict[str, Any]:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO employees (emp_code, name, department, face_key, created_at) VALUES (?,?,?,?,?)",
            (emp_code.strip(), name.strip(), department.strip(), face_key, _now()),
        )
        return _row(con.execute("SELECT * FROM employees WHERE id=?", (cur.lastrowid,)).fetchone())  # type: ignore[return-value]


def get_employee(employee_id: int) -> dict[str, Any] | None:
    with _conn() as con:
        return _row(con.execute("SELECT * FROM employees WHERE id=?", (employee_id,)).fetchone())


def get_employee_by_code(emp_code: str) -> dict[str, Any] | None:
    with _conn() as con:
        return _row(con.execute("SELECT * FROM employees WHERE emp_code=? COLLATE NOCASE", (emp_code.strip(),)).fetchone())


def get_employee_by_face_key(face_key: str) -> dict[str, Any] | None:
    with _conn() as con:
        return _row(con.execute("SELECT * FROM employees WHERE face_key=?", (face_key,)).fetchone())


def list_employees() -> list[dict[str, Any]]:
    with _conn() as con:
        return [dict(r) for r in con.execute("SELECT * FROM employees ORDER BY name").fetchall()]


# ------------------------------------------------------------------ passwords
def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(8)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()
    return f"{salt}${digest}"


def set_password(employee_id: int, password: str) -> None:
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters")
    with _conn() as con:
        con.execute("UPDATE employees SET password_hash=? WHERE id=?", (_hash_password(password), employee_id))


def check_password(emp: dict[str, Any], password: str) -> bool:
    stored = emp.get("password_hash")
    if not stored or "$" not in stored:
        return False
    salt, _ = stored.split("$", 1)
    return secrets.compare_digest(_hash_password(password, salt), stored)


# ----------------------------------------------------------------- attendance
def today_events(employee_id: int, day: str | None = None) -> list[dict[str, Any]]:
    """Events for one employee on `day` (YYYY-MM-DD, default today), newest first."""
    day = day or datetime.now().strftime("%Y-%m-%d")
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM attendance WHERE employee_id=? AND ts LIKE ? ORDER BY ts DESC",
            (employee_id, f"{day}%"),
        ).fetchall()
        return [dict(r) for r in rows]


def today_status(employee_id: int) -> dict[str, Any]:
    """Summary used by the kiosk: whether the person is currently checked in.

    Calendar days for normal shifts. For a cross-midnight (night) shift the
    window extends backwards: events since yesterday's shift start count, so
    a 02:00 punch-out still sees the 22:00 check-in.
    """
    events = today_events(employee_id)
    emp = get_employee(employee_id)
    shift = shift_for_employee(emp) if emp else None
    if shift and shift["end_time"] <= shift["start_time"] and not events:
        # Night shift still open from yesterday (no punches on the calendar day yet).
        now_hm = datetime.now().strftime("%H:%M")
        if now_hm < shift["end_time"]:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            with _conn() as con:
                rows = con.execute(
                    "SELECT * FROM attendance WHERE employee_id=? AND ts >= ? AND ts < ? ORDER BY ts ASC",
                    (employee_id, f"{yesterday} {shift['start_time']}:00", f"{yesterday} 23:59:59"),
                ).fetchall()
                events = [dict(r) for r in reversed(rows)] + events
    last = events[0] if events else None
    return {
        "checked_in": bool(last and last["event_type"] == "IN"),
        "last_event": last,
        "events": events,
    }


def add_attendance(
    employee_id: int,
    event_type: str,
    score: float | None,
    source: str = "kiosk",
    note: str | None = None,
    ts: str | None = None,
    skip_sequence_check: bool = False,
) -> dict[str, Any]:
    """Insert an IN/OUT event after validating the sequence for today.

    Args:
        note: free text stored with the row (correction reasons, etc.).
        ts: override the timestamp (admin corrections; YYYY-MM-DD HH:MM:SS).
        skip_sequence_check: admin back-dated inserts may break IN/OUT order.

    Raises:
        ValueError: on an invalid type, or IN after IN / OUT without IN.
    """
    event_type = event_type.upper()
    if event_type not in ("IN", "OUT"):
        raise ValueError("event_type must be IN or OUT")
    if not skip_sequence_check:
        status = today_status(employee_id)
        if event_type == "IN" and status["checked_in"]:
            raise ValueError("Already checked in. Check out first.")
        if event_type == "OUT" and not status["checked_in"]:
            raise ValueError("Not checked in yet. Check in first.")
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO attendance (employee_id, event_type, ts, score, source, note) VALUES (?,?,?,?,?,?)",
            (employee_id, event_type, ts or _now(), score, source, note),
        )
        return dict(con.execute("SELECT * FROM attendance WHERE id=?", (cur.lastrowid,)).fetchone())


def delete_attendance(event_id: int) -> dict[str, Any] | None:
    """Remove one event (admin correction; caller must write the audit entry)."""
    with _conn() as con:
        row = con.execute("SELECT * FROM attendance WHERE id=?", (event_id,)).fetchone()
        if row is None:
            return None
        con.execute("DELETE FROM attendance WHERE id=?", (event_id,))
        return dict(row)


def list_attendance(day: str | None = None) -> list[dict[str, Any]]:
    """All events for `day` joined with employee details, newest first."""
    day = day or datetime.now().strftime("%Y-%m-%d")
    with _conn() as con:
        rows = con.execute(
            """SELECT a.id, a.event_type, a.ts, a.score, a.source, a.note,
                      e.id AS employee_id, e.emp_code, e.name, e.department
               FROM attendance a JOIN employees e ON e.id = a.employee_id
               WHERE a.ts LIKE ? ORDER BY a.ts DESC""",
            (f"{day}%",),
        ).fetchall()
        return [dict(r) for r in rows]


# -------------------------------------------------------------------- admins
def count_admins() -> int:
    with _conn() as con:
        return int(con.execute("SELECT COUNT(*) AS n FROM admins").fetchone()["n"])


def create_admin(username: str, password_hash: str) -> dict[str, Any]:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO admins (username, password_hash, created_at) VALUES (?,?,?)",
            (username.strip(), password_hash, _now()),
        )
        row = con.execute("SELECT id, username, created_at FROM admins WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get_admin(username: str) -> dict[str, Any] | None:
    with _conn() as con:
        return _row(con.execute("SELECT * FROM admins WHERE username=? COLLATE NOCASE", (username.strip(),)).fetchone())


def set_admin_password(username: str, password_hash: str) -> bool:
    with _conn() as con:
        cur = con.execute("UPDATE admins SET password_hash=? WHERE username=? COLLATE NOCASE",
                          (password_hash, username.strip()))
        return cur.rowcount > 0


# ------------------------------------------------------------------- devices
def create_device(name: str, key_hash: str) -> dict[str, Any]:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO devices (name, key_hash, created_at) VALUES (?,?,?)",
            (name.strip(), key_hash, _now()),
        )
        row = con.execute("SELECT id, name, created_at, last_seen FROM devices WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get_device_by_name(name: str) -> dict[str, Any] | None:
    with _conn() as con:
        return _row(con.execute("SELECT * FROM devices WHERE name=? COLLATE NOCASE", (name.strip(),)).fetchone())


def list_devices() -> list[dict[str, Any]]:
    with _conn() as con:
        return [dict(r) for r in con.execute(
            "SELECT id, name, created_at, last_seen FROM devices ORDER BY name").fetchall()]


def touch_device(device_id: int) -> None:
    with _conn() as con:
        con.execute("UPDATE devices SET last_seen=? WHERE id=?", (_now(), device_id))


def delete_device(device_id: int) -> bool:
    with _conn() as con:
        return con.execute("DELETE FROM devices WHERE id=?", (device_id,)).rowcount > 0


# -------------------------------------------------------------------- shifts
def list_shifts() -> list[dict[str, Any]]:
    with _conn() as con:
        return [dict(r) for r in con.execute("SELECT * FROM shifts ORDER BY start_time").fetchall()]


def get_shift(shift_id: int) -> dict[str, Any] | None:
    with _conn() as con:
        return _row(con.execute("SELECT * FROM shifts WHERE id=?", (shift_id,)).fetchone())


def get_shift_by_name(name: str) -> dict[str, Any] | None:
    with _conn() as con:
        return _row(con.execute("SELECT * FROM shifts WHERE name=? COLLATE NOCASE", (name.strip(),)).fetchone())


def create_shift(name: str, start_time: str, end_time: str, grace_min: int, work_hours: float) -> dict[str, Any]:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO shifts (name, start_time, end_time, grace_min, work_hours) VALUES (?,?,?,?,?)",
            (name.strip(), start_time, end_time, int(grace_min), float(work_hours)),
        )
        row = con.execute("SELECT * FROM shifts WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)


def delete_shift(shift_id: int) -> bool:
    """Delete a shift; employees assigned to it fall back to the default."""
    with _conn() as con:
        con.execute("UPDATE employees SET shift_id=NULL WHERE shift_id=?", (shift_id,))
        return con.execute("DELETE FROM shifts WHERE id=?", (shift_id,)).rowcount > 0


def shift_for_employee(emp: dict[str, Any]) -> dict[str, Any] | None:
    """The employee's shift, or the seeded default shift, or None."""
    shift_id = emp.get("shift_id")
    if shift_id:
        shift = get_shift(int(shift_id))
        if shift:
            return shift
    with _conn() as con:
        return _row(con.execute("SELECT * FROM shifts ORDER BY id LIMIT 1").fetchone())


def assign_shift(employee_id: int, shift_id: int | None, weekoff_weekday: int | None) -> None:
    with _conn() as con:
        con.execute("UPDATE employees SET shift_id=?, weekoff_weekday=? WHERE id=?",
                    (shift_id, weekoff_weekday, employee_id))


# ------------------------------------------------------------------ holidays
def list_holidays(month: str | None = None) -> list[dict[str, Any]]:
    """All holidays, or only those in a YYYY-MM month."""
    with _conn() as con:
        if month:
            return [dict(r) for r in con.execute(
                "SELECT * FROM holidays WHERE date LIKE ? ORDER BY date", (f"{month}%",)).fetchall()]
        return [dict(r) for r in con.execute("SELECT * FROM holidays ORDER BY date").fetchall()]


def add_holiday(date_str: str, name: str) -> None:
    with _conn() as con:
        con.execute("INSERT INTO holidays (date, name) VALUES (?,?) "
                    "ON CONFLICT(date) DO UPDATE SET name=excluded.name", (date_str, name.strip()))


def delete_holiday(date_str: str) -> bool:
    with _conn() as con:
        return con.execute("DELETE FROM holidays WHERE date=?", (date_str,)).rowcount > 0


# --------------------------------------------------------------------- audit
def audit(actor_type: str, actor: str, action: str, detail: str = "", ip: str = "") -> None:
    """Append a hash-chained audit row (call after the action succeeded)."""
    import hashlib

    with _conn() as con:
        prev = con.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = prev["hash"] if prev else ""
        ts = _now()
        material = f"{prev_hash}|{ts}|{actor_type}|{actor}|{action}|{detail}|{ip}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        con.execute(
            "INSERT INTO audit_log (ts, actor_type, actor, action, detail, ip, prev_hash, hash) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ts, actor_type, actor, action, detail, ip, prev_hash, digest),
        )


def list_audit(limit: int = 100) -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, ts, actor_type, actor, action, detail, ip FROM audit_log "
            "ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 1000)),)
        ).fetchall()
        return [dict(r) for r in rows]


def verify_audit_chain(limit: int = 1000) -> dict[str, Any]:
    """Re-hash the most recent rows and report the first break, if any."""
    import hashlib

    with _conn() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 1000)),)
        ).fetchall()]
    rows.reverse()  # oldest first
    prev_hash = ""
    for row in rows:
        material = f"{row['prev_hash']}|{row['ts']}|{row['actor_type']}|{row['actor']}|" \
                   f"{row['action']}|{row['detail']}|{row['ip']}"
        expected = hashlib.sha256(material.encode("utf-8")).hexdigest()
        if row["prev_hash"] != prev_hash or row["hash"] != expected:
            return {"ok": False, "checked": len(rows), "broken_at": row["id"], "ts": row["ts"]}
        prev_hash = row["hash"]
    return {"ok": True, "checked": len(rows), "broken_at": None, "ts": None}


def history(employee_id: int, days: int = 30) -> dict[str, Any]:
    """Per-day attendance for the last `days` days, newest first, plus totals.

    Each day: date, first_in, last_out, hours (sum of IN->OUT pairs; an open
    IN counts up to now only if it is today), complete (ends with OUT), events.
    """
    days = max(1, min(int(days), 366))
    since = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM attendance WHERE employee_id=? AND ts >= ? ORDER BY ts ASC",
            (employee_id, since),
        ).fetchall()

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_day[r["ts"][:10]].append(dict(r))

    today = datetime.now().strftime("%Y-%m-%d")
    result: list[dict[str, Any]] = []
    total_hours = 0.0
    for day in sorted(by_day, reverse=True):
        events = by_day[day]
        first_in = next((e["ts"][11:] for e in events if e["event_type"] == "IN"), None)
        last_out = next((e["ts"][11:] for e in reversed(events) if e["event_type"] == "OUT"), None)
        hours = 0.0
        open_in: datetime | None = None
        for e in events:
            t = datetime.strptime(e["ts"], TS_FMT)
            if e["event_type"] == "IN":
                open_in = t
            elif open_in is not None:
                hours += (t - open_in).total_seconds() / 3600
                open_in = None
        if open_in is not None and day == today:
            hours += (datetime.now() - open_in).total_seconds() / 3600
        hours = round(hours, 2)
        total_hours += hours
        result.append({
            "date": day,
            "first_in": first_in,
            "last_out": last_out,
            "hours": hours,
            "complete": events[-1]["event_type"] == "OUT",
            "events": [{"event_type": e["event_type"], "time": e["ts"][11:], "score": e["score"]} for e in events],
        })
    return {
        "days": days,
        "since": since,
        "days_present": len(result),
        "total_hours": round(total_hours, 2),
        "rows": result,
    }


def export_csv(path: str, day: str | None = None) -> int:
    """Write the day's events to CSV. Returns the number of rows written."""
    rows = list_attendance(day)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["emp_code", "name", "department", "event_type", "date", "time", "score"])
        for r in rows:
            d, t = r["ts"].split(" ")
            w.writerow([r["emp_code"], r["name"], r["department"], r["event_type"], d, t, r["score"]])
    return len(rows)


# ------------------------------------------------------------------- reports
def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _day_window(emp: dict[str, Any], day: str) -> tuple[str, str, dict[str, Any] | None]:
    """(window_start_ts, window_end_ts, shift) for one employee on one date.

    Normal shift: the calendar day. Cross-midnight shift (end <= start): from
    the shift's start on `day` up to its end on the following day.
    """
    shift = shift_for_employee(emp)
    if shift and shift["end_time"] <= shift["start_time"]:
        nxt = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        return f"{day} {shift['start_time']}:00", f"{nxt} {shift['end_time']}:00", shift
    return f"{day} 00:00:00", f"{day} 23:59:59", shift


def last_event(employee_id: int) -> dict[str, Any] | None:
    """Most recent attendance event for one employee (any date), or None."""
    with _conn() as con:
        return _row(con.execute(
            "SELECT * FROM attendance WHERE employee_id=? ORDER BY ts DESC LIMIT 1",
            (employee_id,),
        ).fetchone())


def grade_employee(
    emp: dict[str, Any],
    day: str,
    today: str | None = None,
    holidays: set[str] | None = None,
) -> dict[str, Any]:
    """Grade one employee for one date (used by day_summary and /attendance/mark).

    Status values: HOLIDAY, WEEKOFF, ABSENT, LATE, PRESENT, OPEN (checked in,
    still running), FUTURE. late_min counts minutes past shift start + grace,
    early_min minutes leave taken before shift end, ot_hours hours beyond
    work_hours.
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    if holidays is None:
        holidays = {h["date"] for h in list_holidays(day[:7])}
    start_ts, end_ts, shift = _day_window(emp, day)
    weekday = datetime.strptime(day, "%Y-%m-%d").weekday()
    base = {
        "employee_id": emp["id"], "emp_code": emp["emp_code"], "name": emp["name"],
        "department": emp["department"], "shift": shift["name"] if shift else None,
        "first_in": None, "last_out": None, "hours": 0.0,
        "late_min": 0, "early_min": 0, "ot_hours": 0.0,
    }
    if day in holidays:
        return {**base, "status": "HOLIDAY"}
    if emp.get("weekoff_weekday") is not None and int(emp["weekoff_weekday"]) == weekday:
        return {**base, "status": "WEEKOFF"}

    with _conn() as con:
        events = [dict(r) for r in con.execute(
            "SELECT * FROM attendance WHERE employee_id=? AND ts >= ? AND ts <= ? ORDER BY ts ASC",
            (emp["id"], start_ts, end_ts),
        ).fetchall()]

    ins = [e for e in events if e["event_type"] == "IN"]
    outs = [e for e in events if e["event_type"] == "OUT"]
    if not ins:
        return {**base, "status": "ABSENT"}

    first_in = ins[0]["ts"]
    last_out = outs[-1]["ts"] if outs else None
    hours = 0.0
    open_from: datetime | None = None
    for e in events:
        t = datetime.strptime(e["ts"], TS_FMT)
        if e["event_type"] == "IN":
            open_from = t
        elif open_from is not None:
            hours += (t - open_from).total_seconds() / 3600
            open_from = None
    running = open_from is not None
    if running:
        # Count an open shift up to now, but never past the window end.
        cap = datetime.strptime(end_ts, TS_FMT)
        hours += (min(datetime.now(), cap) - open_from).total_seconds() / 3600
    hours = round(hours, 2)

    late_min = early_min = 0
    ot_hours = 0.0
    if shift:
        s_start = datetime.strptime(f"{day} {shift['start_time']}:00", TS_FMT)
        in_dt = datetime.strptime(first_in, TS_FMT)
        if in_dt > s_start:
            late_min = int((in_dt - s_start).total_seconds() // 60)
            if late_min > int(shift["grace_min"]):
                late_min -= int(shift["grace_min"])  # grace absorbs the first few minutes
            else:
                late_min = 0
        if last_out and not running:
            out_dt = datetime.strptime(last_out, TS_FMT)
            # Shift end: same day for normal shifts, next day for night shifts.
            end_date = end_ts[:10]
            s_end = datetime.strptime(f"{end_date} {shift['end_time']}:00", TS_FMT)
            if out_dt < s_end:
                early_min = int((s_end - out_dt).total_seconds() // 60)
        ot_hours = round(max(0.0, hours - float(shift["work_hours"])), 2)

    if day > today:
        status = "FUTURE"
    elif running and day == today:
        status = "OPEN"
    elif late_min > 0:
        status = "LATE"
    else:
        status = "PRESENT"
    return {
        **base, "status": status,
        "first_in": first_in, "last_out": last_out, "hours": hours,
        "late_min": late_min, "early_min": early_min, "ot_hours": ot_hours,
    }


def day_summary(day: str | None = None) -> list[dict[str, Any]]:
    """Per-employee grading for one date: status, hours, late/early/OT."""
    day = day or datetime.now().strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    holidays = {h["date"] for h in list_holidays(day[:7])}
    rows = [grade_employee(emp, day, today, holidays) for emp in list_employees()]
    rows.sort(key=lambda r: r["name"])
    return rows


def monthly_report(month: str) -> dict[str, Any]:
    """Payroll rollup for a YYYY-MM month, one row per employee.

    Totals: present/late/absent days, week-offs, holidays, hours, overtime.
    A day counts as present when at least one IN was recorded; absence is only
    graded on that employee's working days (week-offs excluded).
    """
    try:
        year, mon = (int(x) for x in month.split("-"))
        datetime(year, mon, 1)
    except (ValueError, TypeError):
        raise ValueError("month must be YYYY-MM")
    first = datetime(year, mon, 1)
    last_day = (datetime(year, mon + 1, 1) - timedelta(days=1)) if mon < 12 else datetime(year + 1, 1, 1)
    ndays = (last_day - first).days
    employees = list_employees()

    # Grade every date once (day_summary loops all employees internally),
    # then fold each employee's days into payroll totals.
    graded: dict[str, dict[int, dict[str, Any]]] = {}
    joined_at = {emp["id"]: emp["created_at"][:10] for emp in employees}
    for i in range(ndays):
        day = (first + timedelta(days=i)).strftime("%Y-%m-%d")
        graded[day] = {r["employee_id"]: r for r in day_summary(day)}

    out: list[dict[str, Any]] = []
    for emp in employees:
        counts = {"present": 0, "late": 0, "absent": 0, "weekoff": 0, "holiday": 0}
        hours = ot = 0.0
        joined = joined_at[emp["id"]]
        for day, by_emp in graded.items():
            if day < joined:  # not employed yet - neither present nor absent
                continue
            r = by_emp.get(emp["id"])
            if r is None:
                continue
            status = r["status"]
            if status == "HOLIDAY":
                counts["holiday"] += 1
            elif status == "WEEKOFF":
                counts["weekoff"] += 1
            elif status == "ABSENT":
                counts["absent"] += 1
            else:
                counts["present"] += 1
                if status == "LATE":
                    counts["late"] += 1
                hours += r["hours"]
                ot += r["ot_hours"]
        out.append({
            "employee_id": emp["id"], "emp_code": emp["emp_code"], "name": emp["name"],
            "department": emp["department"], **counts,
            "total_hours": round(hours, 2), "ot_hours": round(ot, 2),
        })

    totals = {
        "present": sum(r["present"] for r in out), "late": sum(r["late"] for r in out),
        "absent": sum(r["absent"] for r in out), "weekoff": sum(r["weekoff"] for r in out),
        "holiday": sum(r["holiday"] for r in out),
        "total_hours": round(sum(r["total_hours"] for r in out), 2),
        "ot_hours": round(sum(r["ot_hours"] for r in out), 2),
    }
    return {"month": month, "days": ndays, "employees": out, "totals": totals}


init_db()
