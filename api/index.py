"""
Vercel entry point - serves the full Face Attendance API (server.py).

TensorFlow/DeepFace cannot be installed on Vercel (the function-size limit is
500 MB; TensorFlow alone is ~1.4 GB), so the server runs with the OpenCV
YuNet+SFace embedding backend (FACE_EMBEDDER_BACKEND=opencv) and stores its
writable data under /tmp (see config.DATA_DIR). Everything else - auth,
registration, verification, liveness, attendance, reports, dashboard - is the
same server.py used locally and on Render.

Note: on serverless the filesystem is ephemeral per instance, so enrolments
and punches survive warm invocations but reset when Vercel recycles the
instance. Use Render (render.yaml) for a durable production install.
"""

import config
import store
from server import app, _bootstrap  # noqa: F401  (app is the ASGI entry point)

# Not every serverless ASGI runtime fires the lifespan "startup" event, so run
# the (idempotent) bootstrap eagerly: create data folders, tables and the
# bootstrap admin / kiosk device.
config.ensure_dirs()
store.init_db()
_bootstrap()
