# Face Match Demo

[![Live on Vercel](https://img.shields.io/badge/Live%20on%20Vercel-face--match--9uho.vercel.app-black?logo=vercel)](https://face-match-9uho.vercel.app)

**Live deployment:** https://face-match-9uho.vercel.app — the FastAPI attendance API plus the dashboard (`/`). See [Deploying](#deploying) for what runs where.

A local Streamlit interface backed by FastAPI and DeepFace. Capture two images in sequence; the UI displays age, gender, race, emotion, region, and all returned face-analysis fields for each image before showing the same-person result and verification metadata.

Use Python 3.13 or newer, matching the `requires-python` setting in `pyproject.toml`. The current workspace has Python 3.14, but DeepFace is not installed there yet.

## Run on Windows

Create and activate an environment, then install the dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-full.txt
```

Start the API in one terminal:

```powershell
uvicorn api:app --reload
```

Start Streamlit in a second terminal, from the project directory and with the same environment activated:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. The first verification may take longer because DeepFace downloads and initializes its model weights. The API listens on `http://127.0.0.1:8000`; set `FACE_API_URL` to change the URL used by Streamlit.

Phone cameras are supported by the two camera controls in the app. Open the app from the phone using an HTTPS URL and allow camera access; mobile browsers generally block live camera access on a LAN `http://` URL. A secure HTTPS deployment or reverse proxy is needed for phone access outside the local computer.

Only use images you have permission to process. Captured images are held in Streamlit session memory and are not written to disk by this app.

## Face Attendance system (Android kiosk + API)

Production-style attendance flow: the Android app registers employees and verifies faces; the Python API runs the recognition model, enforces auth/shift rules with anti-spoof liveness, and stores everything.

```
Android app (android_app/)                 PC / server (this folder)
┌─────────────────────────┐   /register   ┌──────────────────────────────────┐
│ Register: form + 5 face │ ────────────▶ │ server.py (FastAPI)              │
│ photos (ring green/red) │               │  ├ model_loader.py  DeepFace     │
│                         │   /verify     │  │   Facenet or my_deepface_model│
│ Kiosk: auto-verify face │ ────────────▶ │  ├ known_faces/<code>/*.jpg      │
│ show employee, Check In │ /attendance/  │  ├ embeddings/face_db.pkl        │
│ / Check Out             │   mark        │  └ store.py  attendance.db       │
└─────────────────────────┘               │ dashboard.html  →  http://pc:8000│
                                          └──────────────────────────────────┘
```

Start the server (shows the URL to put in the app):

```powershell
.\run_server.ps1
```

Then open `http://localhost:8000/` for the HR dashboard. It asks you to sign in with an admin account (bootstrap `admin` / `admin123`) and then shows today's graded sheet, the monthly report (with CSV), raw events, employees/shifts, the holiday calendar, kiosk devices and the audit log. If the phone cannot connect, allow the port once from an admin PowerShell:

```powershell
netsh advfirewall firewall add rule name="Face Attendance API" dir=in action=allow protocol=TCP localport=8000
```

API summary (admin endpoints need `Authorization: Bearer <admin token>`, kiosk endpoints a device token — see *Auth* below):

| Endpoint | Purpose |
| --- | --- |
| `POST /auth/login` | `{username,password}` → admin token (rate-limited) |
| `POST /auth/device-token` | `{device,key}` → kiosk token |
| `GET /auth/me`, `POST /auth/password` | current token info; change the admin password |
| `POST /register` | **admin** form `name`, `emp_code`, `department`, `password?` + `images[]`; needs `MIN_REGISTER_IMAGES` good photos, rejects blurry photos and faces already enrolled under another code |
| `POST /verify` | **kiosk** `images[]` (a short burst) → `matched`, `score`, `employee`, `today`, plus `liveness` and `quality`. Uses the burst for anti-spoof; honours EXIF rotation |
| `POST /attendance/mark` | **kiosk** `{employee_id, event_type: IN\|OUT, score}`; enforces IN → OUT order, a punch cooldown, and returns the graded day (`late_min`, `ot_hours`, …) |
| `POST /employee/login` | `{emp_code, password, days?}` → profile, today's state and per-day history (public, rate-limited) |
| `GET /employees/{id}/history`, `POST /employees/{id}/password` | history after a face match; set/change the self-service password |
| `GET /attendance/day?date=` | graded day sheet: status, first in, last out, hours, late/early minutes, overtime |
| `GET /reports/monthly?month=YYYY-MM`, `GET /reports/monthly.csv?month=` | payroll rollup per employee (present/late/absent, week-offs, holidays, hours, OT) |
| `GET/POST/DELETE /shifts`, `POST /employees/{id}/shift` | shift definitions (grace, work hours) and per-employee shift + week-off day |
| `GET/POST/DELETE /holidays` | paid-holiday calendar (holidays are not counted as absences) |
| `GET/POST/DELETE /admin/devices` | kiosk devices; the key is shown once when created |
| `GET /audit?limit=`, `GET /audit/verify` | append-only, hash-chained audit log and its integrity check |
| `POST /attendance/adjust`, `DELETE /attendance/{id}` | admin corrections (back-dated punch / delete), always audited |
| `GET /attendance/today`, `GET /attendance?date=`, `GET /attendance/export.csv` | reporting |
| `GET /employees`, `GET /health`, `GET /` | roster, status, dashboard |

### Auth
The first start creates a bootstrap admin (**`admin` / `admin123`** — change it on first login) and a bootstrap kiosk device (**`kiosk-1`** with key **`kiosk-key-change-me`**, set the same key in the app's settings). Override the bootstrap values with `FACE_ADMIN_USER`, `FACE_ADMIN_PASSWORD`, `FACE_KIOSK_NAME`, `FACE_KIOSK_KEY`. Tokens are HS256, signed with a key generated at `attendance_secret.key` (override with `FACE_ATTENDANCE_SECRET`); `FACE_REQUIRE_AUTH=0` disables auth for local testing. Manage kiosks and rotate keys from the dashboard's *Kiosk devices* card.

### Shifts, rules and reports
`PUNCH_COOLDOWN_SEC` (default 45 s) blocks accidental double taps. Each employee gets a shift (falling back to `DEFAULT_SHIFT`, seeded as 09:00–18:00 with 15 min grace) and a week-off day; `day_summary`/`monthly_report` apply grace, early-leaving, overtime beyond the shift's `work_hours`, week-offs and holidays instead of absences, and attribute cross-midnight (night) shifts to the day the shift started.

### Liveness (anti-spoof)
`/verify` expects a burst of `LIVENESS_FRAMES` (3) frames and rejects, with a clear message, photos/screens that never move (motion), a different face across frames (identity), and dark/blurry frames (quality). Tune with `REQUIRE_LIVENESS`, `LIVENESS_MIN_MOTION`, `LIVENESS_MIN_EMB_SIM`, `MIN_BLUR_VAR`, `BRIGHTNESS_RANGE`, `DUPLICATE_FACE_THRESHOLD`.

Data: photos in `known_faces/<emp_code>/`, embeddings in `embeddings/face_db.pkl`, employees/shifts/holidays/devices/audit and events in `attendance.db` (SQLite). All are git-ignored. The desktop CLI tools below still work and share the same face database; their CSV log is separate from the server's SQLite store.

Security notes for a real deployment: put the API behind HTTPS (then remove `usesCleartextTraffic` from the Android manifest), rotate the bootstrap admin password and the `kiosk-1` device key, and keep `attendance.db` and `attendance_secret.key` off shared storage. Liveness is on by default now, so a printed photo no longer passes on its own.

### Deploying

**Render** (`render.yaml`) runs the full stack: `carrier-face-api` (`uvicorn api:app`) plus `carrier-face-ui` (Streamlit, pointed at the API through `FACE_API_URL`). Data lives on the service disk.

**Vercel** is live at **https://face-match-9uho.vercel.app** and deploys the API only, with a TensorFlow-free build: DeepFace + TensorFlow is ~1.4 GB and Vercel's function-size limit is 500 MB. The deployment therefore installs `requirements.txt` (the serverless-light set: FastAPI, OpenCV, NumPy, Pillow) and recognition switches to OpenCV's bundled DNN models (`models/face_detection_yunet_2023mar.onnx` for detection, `models/face_recognition_sface_2021dec.onnx` for 128-d embeddings, both Apache-2.0). `api/index.py` is the entry point; `.vercelignore` keeps the upload small and hides `uv.lock`/`pyproject.toml` so Vercel does not pull TensorFlow back in through `uv sync`. The heavy stack stays in `requirements-full.txt`.

Serverless caveats on Vercel: the filesystem is read-only except `/tmp`, so `config.DATA_DIR` moves the database, enrolments, embeddings and secret key to `/tmp/carrier`; that data survives warm invocations but resets when the instance is recycled, and requests are capped at 60 s. Use Render (or the desktop tools) when attendance data must be durable.

Environment variables:

| Variable | Purpose |
| --- | --- |
| `FACE_EMBEDDER_BACKEND` | `auto` (default: DeepFace if installed, else OpenCV), `deepface`, or `opencv` |
| `FACE_DATA_DIR` | Writable data directory (defaults to the project folder, or `/tmp/carrier` when `VERCEL=1`) |
| `FACE_REQUIRE_AUTH`, `FACE_REQUIRE_LIVENESS` | `0` disables the corresponding check |
| `FACE_ADMIN_USER`, `FACE_ADMIN_PASSWORD`, `FACE_KIOSK_NAME`, `FACE_KIOSK_KEY` | Bootstrap credentials |

Both backends share the same cosine-similarity matching, so `config.get_threshold()` picks the right value per model (Facenet 0.60, SFace 0.41). A `face_db.pkl` built with one backend cannot be read by the other — run `python build_db.py` after switching.

## Offline face attendance (webcam, no API)

A second, standalone tool in the same folder registers people from the webcam and marks attendance by face. It never calls `DeepFace.verify()`; it extracts embeddings with an embedding network and compares them with cosine similarity.

```
carrier/
├── config.py            # paths, model choice, THRESHOLD, camera settings
├── model_loader.py      # loads my_deepface_model.h5 if present, else DeepFace Facenet
├── camera.py            # webcam preview; SPACE = capture, ESC = quit
├── register.py          # capture 5 images -> known_faces/<name>/ -> update face_db.pkl
├── build_db.py          # (re)compute embeddings for every image in known_faces/
├── verify.py            # capture 1 image -> best match -> attendance.csv
├── attendance_logger.py # CSV writer with one-entry-per-person-per-day rule
├── tune_threshold.py    # genuine vs impostor scores -> suggested THRESHOLD
├── main.py              # menu runner
├── known_faces/         # created automatically (git-ignored)
├── embeddings/face_db.pkl
└── attendance.csv
```

Install. The webcam preview needs the GUI build of OpenCV, and the headless build from `requirements-full.txt` overwrites it when both are present, so remove headless locally:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-full.txt
pip uninstall -y opencv-python-headless
pip install "opencv-python<5"
```

OpenCV 5 removed `cv2.CascadeClassifier`, which DeepFace's default "opencv" detector needs, so OpenCV must stay on the 4.x series (the deployed API keeps using headless 4.x from `requirements.txt`).

`requirements.txt` is the light API set (no TensorFlow) because Vercel installs it; `requirements-full.txt` adds DeepFace, TensorFlow and Streamlit for local desktop/webcam work and the Render services.

Run order:

```powershell
python main.py            # menu, or run the steps directly:
python register.py Vinay  # 1. enrol (5 captures, frames without a face are rejected)
python verify.py          # 2. verify + mark attendance
python verify.py --loop   #    kiosk style, repeat until ESC
python tune_threshold.py  # 3. after 2+ people are registered, pick a threshold
python build_db.py        #    rebuild embeddings after changing the model
```

Model: `facei.ipynb` only runs DeepFace's pretrained VGG-Face, it does not train or save a model, so no `my_deepface_model.h5` exists yet. The system therefore defaults to DeepFace's pretrained Facenet (160x160 RGB scaled to 0..1, 128-d embedding), whose weights are already cached in `%USERPROFILE%\.deepface\weights`. When you do train a model, save it as `my_deepface_model.h5` in this folder and set `CUSTOM_MODEL_NORMALIZATION`, `CUSTOM_MODEL_COLOR` and `CUSTOM_OBJECTS` in `config.py` to match training. Then run `python build_db.py` once.

Threshold: cosine similarity, default 0.60 for Facenet. Raise it if strangers are accepted, lower it if registered users are rejected. `tune_threshold.py` prints both score distributions and the value that maximises balanced accuracy.

Troubleshooting:

| Symptom | Fix |
| --- | --- |
| `Could not open webcam index 0` | Close Teams/Zoom/browser tabs using the camera, check Windows camera privacy settings, or set `CAMERA_INDEX = 1`. |
| `No face detected` on every capture | Face the camera within about one metre, add light in front of you, or try `DETECTOR_BACKEND = "retinaface"` (downloads weights once). |
| `face_db.pkl was built with model ...` | The model changed since enrolment. Run `python build_db.py`. |
| `cv2.imshow` error / `The function is not implemented` | Only `opencv-python-headless` is installed. Run `pip install opencv-python`. |
| `ModuleNotFoundError: pandas._libs...` or TensorFlow DLL errors | The venv's interpreter does not match its packages. Recreate it: `uv venv --python 3.13 --clear .venv` then reinstall. |
| Emoji appear as `?` | Harmless console encoding; `config.py` already forces UTF-8 where possible. |
