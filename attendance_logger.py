"""
attendance_logger.py - CSV attendance log with per-day duplicate protection.

File format (attendance.csv):
    Name,Date,Time
    Vinay,2026-10-01,09:14:22
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

import config

HEADER = ["Name", "Date", "Time"]


def _ensure_file() -> None:
    """Create attendance.csv with a header row if it does not exist."""
    if not os.path.exists(config.ATTENDANCE_CSV):
        with open(config.ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(HEADER)


def read_all() -> list[dict[str, str]]:
    """Return every attendance row as a list of dicts."""
    _ensure_file()
    with open(config.ATTENDANCE_CSV, newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh)]


def read_for_date(date_str: str | None = None) -> list[dict[str, str]]:
    """Return rows for a given YYYY-MM-DD date (default: today)."""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    return [row for row in read_all() if row.get("Date") == date_str]


def already_marked_today(name: str) -> bool:
    """True if `name` already has an entry for today's date."""
    return any(row.get("Name") == name for row in read_for_date())


def mark_attendance(name: str) -> bool:
    """Append a row for `name` with the current date and time.

    Returns:
        True if a new row was written, False if the person was already marked
        present today (no duplicate row is added).
    """
    _ensure_file()
    if already_marked_today(name):
        return False
    now = datetime.now()
    with open(config.ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    return True


def print_today() -> None:
    """Pretty-print today's attendance to the console."""
    rows = read_for_date()
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n📅 Attendance for {today}: {len(rows)} present")
    if not rows:
        print("   (no entries yet)")
        return
    for row in rows:
        print(f"   ✅ {row['Name']:<20} {row['Time']}")
