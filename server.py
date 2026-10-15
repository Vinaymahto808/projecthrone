"""
server.py - Face Attendance API used by the Android app and the web dashboard.

    uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health                       model + database status (public)
POST /auth/login                   {username,password} -> admin token
POST /auth/device-token            {device,key} -> kiosk token
GET  /auth/me                      current token info
POST /auth/password                change the admin password

POST /register                     admin: form name, emp_code, department; files: images[]
POST /verify                       kiosk: files image or images[] -> match + liveness
POST /attendance/mark              kiosk: {employee_id, event_type IN|OUT, score}
POST /employee/login               employee self-service (emp_code + password)
GET  /employees/{id}/history       kiosk: per-day history after a match
POST /employees/{id}/password      kiosk: set the self-service password
GET  /employees                    admin: roster
GET  /attendance?date=             admin: raw events
GET  /attendance/today             admin: today's events + present count
GET  /attendance/day?date=         admin: graded day summary (late/OT/status)
GET  /attendance/export.csv?date=  admin: CSV of the day's events
POST /attendance/adjust            admin: insert a back-dated punch (audited)
DELETE /attendance/{event_id}      admin: delete a punch (audited)
GET  /reports/monthly?month=       admin: payroll rollup (present/late/OT/...)
GET  /reports/monthly.csv?month=   admin: the same as CSV
GET  /shifts  POST /shifts  DELETE /shifts/{id}          admin: shift definitions
POST /employees/{id}/shift         admin: assign shift + week-off day
GET/POST/DELETE /holidays          admin: holiday calendar
GET  /admin/devices                admin: kiosk devices
POST /admin/devices                admin: create a kiosk key (returned once)
DELETE /admin/devices/{id}         admin: revoke a kiosk
GET  /audit?limit=                 admin: audit log
GET  /audit/verify                 admin: check the audit hash chain
GET  /                             HTML dashboard (dashboard.html)

Security
--------
Admin and kiosk requests carry `Authorization: Bearer <token>` (HS256, see
security.py). The kiosk exchanges its device key for a token at
/auth/device-token. Set FACE_REQUIRE_AUTH=0 to disable auth for local testing.

Recognition uses model_loader.FaceEmbedder (your custom .h5 if present,
otherwise DeepFace's pretrained network) with cosine similarity, exactly like
the desktop tools. Faces are stored under known_faces/<face_key>/ and in
embeddings/face_db.pkl; employee details, shifts and events live in
attendance.db. /verify grades the frames with liveness.py (motion + identity +
image quality) to defeat photo / screen replay attacks.
"""

from __future__ import annotations

import csv
import io
import os
import secrets
import threading
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse

import config
import liveness
import security
import store
from build_db import add_embeddings, build_database, db_matches, load_db, save_db
from model_loader import FaceEmbedder, get_embedder
from register import sanitize_name

app = FastAPI(title="Face Attendance API", version="3.0.0")

_embedder: FaceEmbedder | None = None
_model_lock = threading.Lock()   # TF inference + pickle writes are serialised


def embedder() -> FaceEmbedder:
    global _embedder
    if _embedder is None:
        with _model_lock:
            if _embedder is None:
                _embedder = get_embedder()
    return _embedder


# ------------------------------------------------------------------- bootstrap
def _bootstrap() -> None:
    """Create the first admin and kiosk device on a fresh install."""
    if store.count_admins() == 0:
        store.create_admin(
            config.ADMIN_BOOTSTRAP_USER,
            security.hash_password(config.ADMIN_BOOTSTRAP_PASSWORD),
        )
        print(
            "\n🔐 Created bootstrap admin "
            f"'{config.ADMIN_BOOTSTRAP_USER}' with password "
            f"'{config.ADMIN_BOOTSTRAP_PASSWORD}' — change it after first login "
            "(POST /auth/password or the dashboard).\n"
        )
    if not store.list_devices():
        store.create_device(config.KIOSK_BOOTSTRAP_NAME, security.hash_api_key(config.KIOSK_BOOTSTRAP_KEY))
        print(
            f"📱 Created bootstrap kiosk '{config.KIOSK_BOOTSTRAP_NAME}'; device key "
            f"'{config.KIOSK_BOOTSTRAP_KEY}' (set the same value in the app's settings).\n"
        )


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    store.init_db()
    _bootstrap()
    embedder()  # load the model once, before the first request


# -------------------------------------------------------------------- helpers
def decode_upload(data: bytes, label: str = "image") -> np.ndarray:
    """Bytes -> BGR ndarray, honouring EXIF orientation (phone photos)."""
    if not data:
        raise HTTPException(400, f"{label} is empty")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{label} exceeds {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        from PIL import Image, ImageOps

        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # not an image
        raise HTTPException(400, f"{label} is not a valid image: {exc}") from exc

    h, w = img.shape[:2]
    scale = config.MAX_IMAGE_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _face_bgr(face_rgb01: np.ndarray) -> np.ndarray:
    """DeepFace returns RGB 0..1 crops; liveness checks want uint8 BGR."""
    face = np.clip(np.asarray(face_rgb01, dtype=np.float32), 0.0, 1.0)
    return np.ascontiguousarray((face[..., ::-1] * 255.0).astype(np.uint8))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _employee_payload(emp: dict[str, Any]) -> dict[str, Any]:
    status = store.today_status(emp["id"])
    return {"employee": store.public(emp), "today": status}


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def _audit(request: Request, claims: dict[str, Any], action: str, detail: str = "") -> None:
    store.audit(claims.get("role", "system"), str(claims.get("sub", "?")), action, detail, _client_ip(request))


def _identify(vecs: list[np.ndarray], db: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Average per-frame similarity per person; the best average wins.

    Averaging the burst is more stable than a single frame and naturally
    implements \"the same person in every frame\" consensus.
    """
    per_person: dict[str, list[float]] = {}
    for vec in vecs:
        for name, person in db["people"].items():
            embs = person["embeddings"]
            if len(embs) == 0:
                continue
            per_person.setdefault(name, []).append(float((embs @ vec).max()))
    scores = {name: float(np.mean(vals)) for name, vals in per_person.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name = ranked[0][0] if ranked else None
    best = ranked[0][1] if ranked else -1.0
    matched = bool(best_name is not None and best >= threshold)
    return {
        "matched": matched,
        "name": best_name if matched else None,
        "score": best,
        "threshold": threshold,
        "candidates": [(n, s, s) for n, s in ranked[:3]],
    }


# ----------------------------------------------------------------------- auth
def principal(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Resolve the caller from the Bearer token (or open mode)."""
    if not config.REQUIRE_AUTH:
        return {"role": "admin", "sub": "auth-disabled", "exp": 0}
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    claims = security.verify_token(token)
    if claims is None:
        raise HTTPException(401, "Missing or invalid token", headers={"WWW-Authenticate": "Bearer"})
    return claims


def require_admin(claims: dict[str, Any] = Depends(principal)) -> dict[str, Any]:
    if claims.get("role") != "admin":
        raise HTTPException(403, "Admin privileges required")
    return claims


def require_worker(claims: dict[str, Any] = Depends(principal)) -> dict[str, Any]:
    if claims.get("role") not in ("admin", "device"):
        raise HTTPException(403, "Kiosk or admin privileges required")
    return claims


@app.post("/auth/login")
def auth_login(request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Admin login -> Bearer token."""
    ip = _client_ip(request)
    if security.login_limiter.blocked(f"admin:{ip}"):
        raise HTTPException(429, f"Too many failed attempts; wait {config.LOGIN_WINDOW_SEC}s")
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    admin = store.get_admin(username) if username else None
    if admin is None or not security.check_password(admin.get("password_hash"), password):
        security.login_limiter.record_failure(f"admin:{ip}")
        store.audit("anonymous", username or "?", "login_failed", "admin login", ip)
        raise HTTPException(401, "Invalid username or password")
    security.login_limiter.reset(f"admin:{ip}")
    token, exp = security.create_token({"sub": admin["username"], "role": "admin"})
    store.audit("admin", admin["username"], "login", "admin login", ip)
    return {"token": token, "role": "admin", "expires": exp, "username": admin["username"]}


@app.post("/auth/device-token")
def auth_device_token(request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Kiosk exchanges its device key for a short-lived token."""
    ip = _client_ip(request)
    if security.login_limiter.blocked(f"device:{ip}"):
        raise HTTPException(429, f"Too many failed attempts; wait {config.LOGIN_WINDOW_SEC}s")
    name = str(body.get("device", body.get("device_name", ""))).strip()
    key = str(body.get("key", body.get("device_key", "")))
    device = store.get_device_by_name(name) if name else None
    if device is None or not security.check_api_key(device.get("key_hash"), key):
        security.login_limiter.record_failure(f"device:{ip}")
        raise HTTPException(401, "Unknown device or bad device key")
    security.login_limiter.reset(f"device:{ip}")
    store.touch_device(int(device["id"]))
    token, exp = security.create_token({"sub": device["name"], "role": "device", "device_id": device["id"]})
    store.audit("device", device["name"], "login", "device token", ip)
    return {"token": token, "role": "device", "expires": exp, "device": device["name"]}


@app.get("/auth/me")
def auth_me(claims: dict[str, Any] = Depends(principal)) -> dict[str, Any]:
    return {"role": claims.get("role"), "sub": claims.get("sub"), "expires": claims.get("exp")}


@app.post("/auth/password")
def auth_change_password(
    request: Request, body: dict[str, Any] = Body(...), claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    old = str(body.get("old_password", ""))
    new = str(body.get("new_password", ""))
    if len(new) < 6:
        raise HTTPException(400, "New password must be at least 6 characters")
    admin = store.get_admin(str(claims.get("sub", "")))
    if admin is None or not security.check_password(admin.get("password_hash"), old):
        raise HTTPException(401, "Current password is incorrect")
    store.set_admin_password(admin["username"], security.hash_password(new))
    _audit(request, claims, "password_changed", admin["username"])
    return {"ok": True}


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health() -> dict[str, Any]:
    emb = embedder()
    db = load_db()
    return {
        "status": "ok",
        "model": emb.describe(),
        "threshold": config.get_threshold(emb.model_id),
        "registered_faces": len(db["people"]) if db else 0,
        "employees": len(store.list_employees()),
        "auth_required": config.REQUIRE_AUTH,
        "liveness_required": config.REQUIRE_LIVENESS,
    }


@app.post("/register")
def register(
    request: Request,
    name: str = Form(...),
    emp_code: str = Form(...),
    department: str = Form(""),
    password: str = Form(""),
    images: list[UploadFile] = File(...),
    claims: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Admin: enrol an employee from several face photos (one face per photo).

    Every image passes the quality gate (blur/brightness), and the face is
    compared against existing enrolments to stop duplicate identities.
    """
    name = name.strip()
    emp_code = emp_code.strip()
    if not name or not emp_code:
        raise HTTPException(400, "name and emp_code are required")
    if password and len(password) < 4:
        raise HTTPException(400, "password must be at least 4 characters")
    if store.get_employee_by_code(emp_code):
        raise HTTPException(409, f"Employee {emp_code} is already registered")
    face_key = sanitize_name(emp_code) or "EMP"
    if store.get_employee_by_face_key(face_key):
        raise HTTPException(409, f"Face key {face_key} already in use")

    emb = embedder()
    accepted: list[tuple[np.ndarray, np.ndarray]] = []   # (frame, embedding)
    rejected: list[str] = []
    for idx, up in enumerate(images, start=1):
        frame = decode_upload(up.file.read(), up.filename or f"image {idx}")
        with _model_lock:
            faces = emb.detect_faces(frame)
            if len(faces) != 1:
                rejected.append(f"image {idx}: {'no face' if not faces else f'{len(faces)} faces'}")
                continue
            gate = liveness.quality_gate(_face_bgr(faces[0]["face"]))
            if not gate["passed"]:
                rejected.append(f"image {idx}: {gate['reason']}")
                continue
            vec = emb.embed_face(faces[0]["face"])
        accepted.append((frame, vec))

    if len(accepted) < config.MIN_REGISTER_IMAGES:
        raise HTTPException(
            422,
            f"Need at least {config.MIN_REGISTER_IMAGES} good photos with exactly one face; "
            f"got {len(accepted)}. Rejected: {', '.join(rejected) or 'none'}",
        )

    # Duplicate-identity guard: is this face already someone else?
    with _model_lock:
        db = load_db()
        if db and db_matches(db, emb) and db["people"]:
            for _frame, vec in accepted:
                best_name, best = None, -1.0
                for person_key, person in db["people"].items():
                    if person_key == face_key or len(person["embeddings"]) == 0:
                        continue
                    sim = float((person["embeddings"] @ vec).max())
                    if sim > best:
                        best, best_name = sim, person_key
                if best >= config.DUPLICATE_FACE_THRESHOLD:
                    existing = store.get_employee_by_face_key(best_name)
                    label = existing["name"] if existing else best_name
                    store.audit("admin", str(claims.get("sub")), "register_duplicate",
                                f"{emp_code} matches {label} ({best:.2f})", _client_ip(request))
                    raise HTTPException(
                        409,
                        f"This face is already registered as '{label}' "
                        f"(similarity {best:.2f}). Use a different person or code.",
                    )

    # Persist images + embeddings, then the employee record.
    person_dir = os.path.join(config.KNOWN_FACES_DIR, face_key)
    os.makedirs(person_dir, exist_ok=True)
    files, vecs = [], []
    for i, (frame, vec) in enumerate(accepted, start=1):
        path = os.path.join(person_dir, f"{face_key}_{i:02d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        files.append(path)
        vecs.append(vec)

    with _model_lock:
        db = load_db()
        if not db_matches(db, emb):
            db = build_database(emb, verbose=False)   # includes the files just written
        else:
            add_embeddings(db, face_key, vecs, files)
            save_db(db)

    employee = store.create_employee(emp_code, name, department, face_key)
    if password:
        store.set_password(employee["id"], password)
        employee = store.get_employee(employee["id"]) or employee
    _audit(request, claims, "register", f"{emp_code} {name}")
    return {
        "ok": True,
        "employee": store.public(employee),
        "images_saved": len(files),
        "rejected": rejected,
    }


@app.post("/verify")
def verify(
    request: Request,
    image: UploadFile | None = File(default=None),
    images: list[UploadFile] | None = File(default=None),
    claims: dict[str, Any] = Depends(require_worker),
) -> dict[str, Any]:
    """Identify a face from one or more frames. Does not write attendance.

    Send `images` (a short burst, LIVENESS_FRAMES) for anti-spoof checks, or the
    legacy single `image` field. Always returns the match plus liveness/quality.
    """
    emb = embedder()
    uploads = ([image] if image is not None else []) + list(images or [])
    if not uploads:
        raise HTTPException(400, "Send 'image' or one or more 'images'")
    uploads = uploads[: max(1, config.LIVENESS_FRAMES)]

    frames: list[np.ndarray] = []
    crops: list[np.ndarray] = []
    vecs: list[np.ndarray] = []
    areas: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for idx, up in enumerate(uploads, start=1):
        frame = decode_upload(up.file.read(), up.filename or f"image {idx}")
        with _model_lock:
            faces = emb.detect_faces(frame)
            if not faces:
                continue
            face = faces[0]
            vecs.append(emb.embed_face(face["face"]))
        frames.append(frame)
        crops.append(_face_bgr(face["face"]))
        areas.append(face["facial_area"])
        gates.append(liveness.quality_gate(crops[-1]))

    if not vecs:
        raise HTTPException(422, "No face detected")

    quality = liveness.summarise_quality(gates)
    if not quality["passed"]:
        raise HTTPException(422, f"Image quality too low: {quality['reason']}")

    live = liveness.assess_liveness(frames, vecs, areas)
    if config.REQUIRE_LIVENESS:
        if len(vecs) < config.LIVENESS_MIN_FRAMES:
            raise HTTPException(
                422, f"Liveness check needs at least {config.LIVENESS_MIN_FRAMES} frames; got {len(vecs)}"
            )
        if not live["passed"]:
            store.audit("device", str(claims.get("sub")), "liveness_failed", live["reason"], _client_ip(request))
            raise HTTPException(403, f"Liveness check failed: {live['reason']}")

    db = load_db()
    if db is None or not db["people"]:
        raise HTTPException(404, "No registered faces yet")
    if not db_matches(db, emb):
        raise HTTPException(500, "Face database was built with a different model; run build_db.py")

    threshold = config.get_threshold(emb.model_id)
    result = _identify(vecs, db, threshold)
    payload: dict[str, Any] = {
        "matched": result["matched"],
        "score": round(result["score"], 4),
        "threshold": threshold,
        "face": areas[0],
        "frames": len(vecs),
        "candidates": [{"face_key": n, "score": round(b, 4)} for n, b, _ in result["candidates"][:3]],
        "liveness": live,
        "quality": {k: v for k, v in quality.items() if k != "details"},
        "employee": None,
        "today": None,
    }
    if result["matched"]:
        emp = store.get_employee_by_face_key(result["name"])
        if emp is None:  # registered via desktop register.py, no HR record
            emp = {"id": None, "emp_code": result["name"], "name": result["name"], "department": "", "face_key": result["name"]}
            payload["employee"] = emp
        else:
            payload.update(_employee_payload(emp))
    return _json_safe(payload)


@app.post("/attendance/mark")
def mark(
    request: Request, body: dict[str, Any] = Body(...), claims: dict[str, Any] = Depends(require_worker)
) -> dict[str, Any]:
    """Record IN or OUT for an employee (after a successful /verify)."""
    try:
        employee_id = int(body["employee_id"])
        event_type = str(body["event_type"]).upper()
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "employee_id and event_type are required")
    score = body.get("score")
    emp = store.get_employee(employee_id)
    if emp is None:
        raise HTTPException(404, "Employee not found")

    # Punch cooldown: stop accidental double taps / racing the kiosk.
    last = store.last_event(employee_id)
    if last:
        try:
            prev = datetime.strptime(last["ts"], store.TS_FMT)
            if (datetime.now() - prev).total_seconds() < config.PUNCH_COOLDOWN_SEC:
                raise HTTPException(429, f"Please wait {config.PUNCH_COOLDOWN_SEC}s between punches")
        except ValueError:
            pass

    try:
        event = store.add_attendance(
            employee_id, event_type, float(score) if score is not None else None,
            source=str(body.get("source", "kiosk")),
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

    _audit(request, claims, f"punch_{event_type.lower()}", f"{emp['emp_code']} {emp['name']}")
    today = datetime.now().strftime("%Y-%m-%d")
    return {"ok": True, "event": event, "grading": store.grade_employee(emp, today), **_employee_payload(emp)}


@app.post("/employee/login")
def employee_login(request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Employee self-service: emp_code + password -> profile, today, history."""
    emp_code = str(body.get("emp_code", "")).strip()
    password = str(body.get("password", ""))
    days = int(body.get("days", 30) or 30)
    bucket = f"emp:{emp_code}:{_client_ip(request)}"
    if security.login_limiter.blocked(bucket):
        raise HTTPException(429, f"Too many failed attempts; wait {config.LOGIN_WINDOW_SEC}s")
    emp = store.get_employee_by_code(emp_code) if emp_code else None
    if emp is None or not store.check_password(emp, password):
        security.login_limiter.record_failure(bucket)
        raise HTTPException(401, "Invalid employee code or password")
    security.login_limiter.reset(bucket)
    return {**_employee_payload(emp), "history": store.history(emp["id"], days)}


@app.get("/employees/{employee_id}/history")
def employee_history(
    employee_id: int, days: int = 30, claims: dict[str, Any] = Depends(require_worker)
) -> dict[str, Any]:
    """Per-day history (used by the kiosk after a successful face match)."""
    emp = store.get_employee(employee_id)
    if emp is None:
        raise HTTPException(404, "Employee not found")
    return {**_employee_payload(emp), "history": store.history(employee_id, days)}


@app.post("/employees/{employee_id}/password")
def employee_set_password(
    request: Request,
    employee_id: int,
    body: dict[str, Any] = Body(...),
    claims: dict[str, Any] = Depends(require_worker),
) -> dict[str, Any]:
    """Set or change the self-service password (verify the face first on the kiosk)."""
    emp = store.get_employee(employee_id)
    if emp is None:
        raise HTTPException(404, "Employee not found")
    try:
        store.set_password(employee_id, str(body.get("password", "")))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _audit(request, claims, "employee_password_set", emp["emp_code"])
    return {"ok": True}


@app.get("/employees")
def employees(claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    db = load_db()
    counts = {k: len(v["files"]) for k, v in (db or {"people": {}})["people"].items()}
    rows = [{**store.public(e), "images": counts.get(e["face_key"], 0)} for e in store.list_employees()]
    return {"employees": rows}


@app.get("/attendance")
def attendance(date: str | None = None, claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    return {"date": date, "events": store.list_attendance(date)}


@app.get("/attendance/today")
def attendance_today(claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    events = store.list_attendance()
    present = {e["employee_id"] for e in events if e["event_type"] == "IN"}
    return {"events": events, "present_count": len(present), "total_employees": len(store.list_employees())}


@app.get("/attendance/day")
def attendance_day(date: str | None = None, claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    day = date or datetime.now().strftime("%Y-%m-%d")
    return {"date": day, "employees": store.day_summary(day)}


@app.get("/attendance/export.csv")
def attendance_export(date: str | None = None, claims: dict[str, Any] = Depends(require_admin)) -> FileResponse:
    path = os.path.join(config.BASE_DIR, "attendance_export.csv")
    store.export_csv(path, date)
    return FileResponse(path, media_type="text/csv", filename=f"attendance_{date or 'today'}.csv")


@app.post("/attendance/adjust")
def attendance_adjust(
    request: Request, body: dict[str, Any] = Body(...), claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    """Insert a back-dated punch (missed/incorrect swipe). Always audited."""
    try:
        employee_id = int(body["employee_id"])
        event_type = str(body["event_type"]).upper()
        date = str(body["date"])
        time = str(body.get("time") or "").strip()
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "employee_id, event_type, date are required")
    emp = store.get_employee(employee_id)
    if emp is None:
        raise HTTPException(404, "Employee not found")
    if event_type not in ("IN", "OUT"):
        raise HTTPException(400, "event_type must be IN or OUT")
    if not time:
        time = "09:00:00" if event_type == "IN" else "18:00:00"
    ts = f"{date} {time if time.count(':') == 2 else time + ':00'}"
    try:
        event = store.add_attendance(employee_id, event_type, None,
                                     source=f"manual:{claims.get('sub')}", note=str(body.get("reason", "")),
                                     ts=ts, skip_sequence_check=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _audit(request, claims, "attendance_adjust", f"{emp['emp_code']} {event_type} {ts} {body.get('reason', '')}")
    return {"ok": True, "event": event}


@app.delete("/attendance/{event_id}")
def attendance_delete(
    request: Request, event_id: int, reason: str = Query(""), claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    """Delete one event (admin correction). Always audited."""
    row = store.delete_attendance(event_id)
    if row is None:
        raise HTTPException(404, "Event not found")
    _audit(request, claims, "attendance_delete", f"id={event_id} {row['event_type']} {row['ts']} {reason}")
    return {"ok": True, "deleted": row}


@app.get("/reports/monthly")
def reports_monthly(month: str = Query(...), claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    try:
        return store.monthly_report(month)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/reports/monthly.csv")
def reports_monthly_csv(month: str = Query(...), claims: dict[str, Any] = Depends(require_admin)) -> FileResponse:
    try:
        report = store.monthly_report(month)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    path = os.path.join(config.BASE_DIR, f"report_{month}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["emp_code", "name", "department", "present", "late", "absent",
                    "weekoff", "holiday", "total_hours", "ot_hours"])
        for r in report["employees"]:
            w.writerow([r["emp_code"], r["name"], r["department"], r["present"], r["late"],
                        r["absent"], r["weekoff"], r["holiday"], r["total_hours"], r["ot_hours"]])
    return FileResponse(path, media_type="text/csv", filename=f"report_{month}.csv")


# --------------------------------------------------------------- shifts & rules
@app.get("/shifts")
def shifts(claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    return {"shifts": store.list_shifts()}


@app.post("/shifts")
def create_shift(
    request: Request, body: dict[str, Any] = Body(...), claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    name = str(body.get("name", "")).strip()
    start = str(body.get("start_time", "")).strip()
    end = str(body.get("end_time", "")).strip()
    if not name or ":" not in start or ":" not in end:
        raise HTTPException(400, "name, start_time (HH:MM) and end_time (HH:MM) are required")
    if store.get_shift_by_name(name):
        raise HTTPException(409, f"Shift '{name}' already exists")
    shift = store.create_shift(name, start, end, int(body.get("grace_min", 15)), float(body.get("work_hours", 8)))
    _audit(request, claims, "shift_create", f"{name} {start}-{end}")
    return {"shift": shift}


@app.delete("/shifts/{shift_id}")
def delete_shift(
    request: Request, shift_id: int, claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    if not store.delete_shift(shift_id):
        raise HTTPException(404, "Shift not found")
    _audit(request, claims, "shift_delete", str(shift_id))
    return {"ok": True}


@app.post("/employees/{employee_id}/shift")
def assign_shift(
    request: Request,
    employee_id: int,
    body: dict[str, Any] = Body(...),
    claims: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    emp = store.get_employee(employee_id)
    if emp is None:
        raise HTTPException(404, "Employee not found")
    shift_id = body.get("shift_id")
    weekoff = body.get("weekoff_weekday")
    shift_id = int(shift_id) if shift_id not in (None, "", "null") else None
    weekoff = int(weekoff) if weekoff not in (None, "", "null") else None
    store.assign_shift(employee_id, shift_id, weekoff)
    _audit(request, claims, "assign_shift", f"{emp['emp_code']} shift={shift_id} weekoff={weekoff}")
    return {"ok": True, "employee": store.public(store.get_employee(employee_id))}


@app.get("/holidays")
def holidays(month: str | None = None, claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    return {"holidays": store.list_holidays(month)}


@app.post("/holidays")
def add_holiday(
    request: Request, body: dict[str, Any] = Body(...), claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    date = str(body.get("date", "")).strip()
    if len(date) != 10:
        raise HTTPException(400, "date must be YYYY-MM-DD")
    store.add_holiday(date, str(body.get("name", "")))
    _audit(request, claims, "holiday_add", f"{date} {body.get('name', '')}")
    return {"ok": True}


@app.delete("/holidays/{date}")
def delete_holiday(
    request: Request, date: str, claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    if not store.delete_holiday(date):
        raise HTTPException(404, "Holiday not found")
    _audit(request, claims, "holiday_delete", date)
    return {"ok": True}


# --------------------------------------------------------------------- devices
@app.get("/admin/devices")
def list_devices(claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    return {"devices": store.list_devices()}


@app.post("/admin/devices")
def create_device(
    request: Request, body: dict[str, Any] = Body(...), claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    name = str(body.get("name", "")).strip()
    if not name:
        raise HTTPException(400, "name is required")
    if store.get_device_by_name(name):
        raise HTTPException(409, f"Device '{name}' already exists")
    key = "fk_" + secrets.token_urlsafe(24)
    device = store.create_device(name, security.hash_api_key(key))
    _audit(request, claims, "device_create", name)
    return {"device": device, "key": key, "note": "Store this key now; it is shown only once."}


@app.delete("/admin/devices/{device_id}")
def delete_device(
    request: Request, device_id: int, claims: dict[str, Any] = Depends(require_admin)
) -> dict[str, Any]:
    if not store.delete_device(device_id):
        raise HTTPException(404, "Device not found")
    _audit(request, claims, "device_delete", str(device_id))
    return {"ok": True}


# ----------------------------------------------------------------------- audit
@app.get("/audit")
def audit_log(limit: int = 100, claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    return {"entries": store.list_audit(limit)}


@app.get("/audit/verify")
def audit_verify(limit: int = 1000, claims: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    return store.verify_audit_chain(limit)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    with open(os.path.join(config.BASE_DIR, "dashboard.html"), encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=False)
