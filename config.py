"""
config.py - Central configuration for the Face Attendance System.

Every other module imports its paths and tunable constants from here, so this
is the only file you should need to edit to adapt the system to your machine
or to a different embedding model.

Windows note: all paths are built with os.path.join relative to this file's
folder, so the project can be moved without breaking anything.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Console / TensorFlow housekeeping (must run before TensorFlow is imported)
# ---------------------------------------------------------------------------
# Windows consoles default to cp1252 and crash on emoji; force UTF-8 output.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - best effort only
            pass

# Silence TensorFlow's very chatty INFO/WARNING logs (0 = all, 3 = errors only).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Registered users: known_faces/<person_name>/<person_name>_01.jpg ...
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

# Cached embeddings so we never recompute them on every verification.
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FACE_DB_PATH = os.path.join(EMBEDDINGS_DIR, "face_db.pkl")

# Attendance log: columns Name, Date, Time
ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance.csv")

# ---------------------------------------------------------------------------
# Embedding model selection
# ---------------------------------------------------------------------------
# 1) If CUSTOM_MODEL_PATH exists it is loaded with tf.keras and used for
#    embeddings. This is where YOUR trained model goes once you have one.
# 2) Otherwise DeepFace's pretrained DEEPFACE_MODEL_NAME is used. The weights
#    are cached in %USERPROFILE%\.deepface\weights\ after the first download.
#    "Facenet" (160x160 input, 128-d embedding) is already cached on this PC.
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "my_deepface_model.h5")

# How the custom model expects pixels. Must EXACTLY match training:
#   "0-1"      -> img / 255.0
#   "0-255"    -> raw 0..255 floats
#   "-1-1"     -> (img / 127.5) - 1
#   "imagenet" -> 0..255 then subtract ImageNet mean / divide std (per channel)
CUSTOM_MODEL_NORMALIZATION = "0-1"

# Channel order the custom model was trained with: "rgb" or "bgr".
CUSTOM_MODEL_COLOR = "rgb"

# Custom Keras layers/loss objects needed to deserialize the .h5, e.g.
#   {"L2Normalize": L2Normalize, "triplet_loss": triplet_loss}
# Leave empty if the model only uses built-in Keras layers.
CUSTOM_OBJECTS: dict = {}

# DeepFace pretrained fallback. Options: "Facenet", "Facenet512", "VGG-Face",
# "ArcFace", "SFace", "GhostFaceNet", "OpenFace", "DeepID", "Dlib".
DEEPFACE_MODEL_NAME = "Facenet"

# Face detector used for cropping/alignment before embedding.
# "opencv" ships with OpenCV (no download, fast). Alternatives: "retinaface",
# "mtcnn", "ssd", "yunet" (more accurate, slower, may download weights).
DETECTOR_BACKEND = "opencv"

# ---------------------------------------------------------------------------
# Matching threshold (cosine SIMILARITY, higher = more similar, range -1..1)
# ---------------------------------------------------------------------------
# Set THRESHOLD to a float to override the per-model default below.
# Raising it reduces false accepts (strangers marked present) but increases
# false rejects (real users not recognised). Run tune_threshold.py after
# registering 2+ people to pick a value from genuine vs impostor scores.
THRESHOLD: float | None = None

# Defaults derived from DeepFace's tuned cosine DISTANCE thresholds
# (similarity = 1 - distance).
DEFAULT_THRESHOLDS = {
    "Facenet": 0.60,
    "Facenet512": 0.70,
    "VGG-Face": 0.32,
    "ArcFace": 0.32,
    "SFace": 0.41,
    "GhostFaceNet": 0.35,
    "OpenFace": 0.90,
    "DeepID": 0.985,
    "Dlib": 0.93,
    "DeepFace": 0.77,
    "custom": 0.60,
}

# ---------------------------------------------------------------------------
# Capture settings
# ---------------------------------------------------------------------------
IMAGES_PER_USER = 5          # images captured during registration
CAMERA_INDEX = 0             # 0 = default webcam; try 1 for an external camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 95

# Ignore detections smaller than this (pixels); filters far-away background faces.
MIN_FACE_SIZE = 60

# ---------------------------------------------------------------------------
# Server (server.py) settings
# ---------------------------------------------------------------------------
SQLITE_PATH = os.path.join(BASE_DIR, "attendance.db")   # employees + events
SERVER_HOST = "0.0.0.0"                                  # listen on LAN
SERVER_PORT = 8000
MIN_REGISTER_IMAGES = 3        # uploads that must contain exactly one face
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 1280          # downscale larger uploads before detection

# ---------------------------------------------------------------------------
# Security (server.py / security.py)
# ---------------------------------------------------------------------------
# Signing key for admin/kiosk tokens (JWT HS256, stdlib only). Generated on
# first start and stored in SECRET_KEY_PATH; FACE_ATTENDANCE_SECRET overrides.
SECRET_KEY_PATH = os.path.join(BASE_DIR, "attendance_secret.key")
TOKEN_TTL_SEC = 12 * 3600                     # tokens valid for 12 hours
REQUIRE_AUTH = os.environ.get("FACE_REQUIRE_AUTH", "1") != "0"
# Bootstrap admin created when the admins table is empty (change after login).
ADMIN_BOOTSTRAP_USER = os.environ.get("FACE_ADMIN_USER", "admin")
ADMIN_BOOTSTRAP_PASSWORD = os.environ.get("FACE_ADMIN_PASSWORD", "admin123")
# Bootstrap kiosk device created when the devices table is empty. The Android
# app's "Device key" setting must match; rotate it from the dashboard.
KIOSK_BOOTSTRAP_NAME = os.environ.get("FACE_KIOSK_NAME", "kiosk-1")
KIOSK_BOOTSTRAP_KEY = os.environ.get("FACE_KIOSK_KEY", "kiosk-key-change-me")
# Brute-force protection for POST /auth/login (per client IP).
LOGIN_RATE_LIMIT = 5                          # failed attempts allowed ...
LOGIN_WINDOW_SEC = 60                         # ... per this many seconds

# ---------------------------------------------------------------------------
# Attendance rules (industry behaviour)
# ---------------------------------------------------------------------------
# Minimum seconds between two punches by the same employee. Blocks the same
# face re-punching instantly (tailgating the kiosk / accidental double taps).
PUNCH_COOLDOWN_SEC = 45
# When no shifts are assigned, employees fall back to this default shift.
DEFAULT_SHIFT = {"name": "General", "start_time": "09:00", "end_time": "18:00",
                 "grace_min": 15, "work_hours": 8.0}

# ---------------------------------------------------------------------------
# Liveness / anti-spoof (liveness.py)
# ---------------------------------------------------------------------------
REQUIRE_LIVENESS = os.environ.get("FACE_REQUIRE_LIVENESS", "1") != "0"
LIVENESS_FRAMES = 3          # frames the kiosk should send per verification
LIVENESS_MIN_FRAMES = 2      # fewer than this -> "skipped" (legacy clients)
LIVENESS_MIN_MOTION = 0.35   # min % mean-abs-diff between face crops; less
                             # means a frozen image / printed photo replay
LIVENESS_MIN_EMB_SIM = 0.75  # cross-frame embedding cosine; lower = different
                             # people cycling through the frame
MIN_BLUR_VAR = 45.0          # Laplacian variance below this = too blurry
BRIGHTNESS_RANGE = (35, 225) # face crop mean brightness bounds
# Reject a new enrolment whose face already matches an existing person this
# well (stops the same person registering twice under different codes).
DUPLICATE_FACE_THRESHOLD = 0.72


def get_threshold(model_id: str) -> float:
    """Return the active similarity threshold for the given model id.

    Args:
        model_id: "custom" or a DeepFace model name such as "Facenet".
    """
    if THRESHOLD is not None:
        return float(THRESHOLD)
    return float(DEFAULT_THRESHOLDS.get(model_id, DEFAULT_THRESHOLDS["custom"]))


def ensure_dirs() -> None:
    """Create the data folders if they do not exist yet."""
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
