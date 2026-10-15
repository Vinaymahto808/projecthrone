"""
camera.py - Webcam capture helpers shared by register.py and verify.py.

Controls inside the preview window:
    SPACE  -> capture the current frame
    ESC    -> abort (returns the frames captured so far)

A lightweight Haar-cascade box is drawn on the live preview purely as visual
feedback; the real detection/alignment happens in model_loader.FaceEmbedder.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

import config

_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


def open_camera(index: int | None = None) -> cv2.VideoCapture:
    """Open the webcam, trying the default backend, then DirectShow and MSMF.

    Raises:
        CameraError: if no backend can deliver frames.
    """
    index = config.CAMERA_INDEX if index is None else index
    for backend in (cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
    raise CameraError(
        f"Could not open webcam index {index}. Close other apps using the camera, "
        "check Windows Settings > Privacy > Camera, or change CAMERA_INDEX in config.py."
    )


def _draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw face boxes and instruction text on a copy of `frame`."""
    view = frame.copy()
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    for (x, y, w, h) in faces:
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0), 2)
    y0 = 28
    for line in lines:
        cv2.putText(view, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 26
    return view


def capture_frames(
    count: int,
    window_title: str = "Face Attendance",
    hint: str = "",
    validator: Callable[[np.ndarray], tuple[bool, str]] | None = None,
) -> list[np.ndarray]:
    """Interactively capture `count` frames from the webcam.

    Args:
        count: number of frames to collect.
        window_title: OpenCV window name.
        hint: extra instruction shown on the preview (e.g. "Turn head slightly").
        validator: optional callable(frame) -> (ok, message). When given, a
            captured frame is only kept if ok is True; the message is shown on
            screen either way (used to reject frames with no face).

    Returns:
        List of BGR frames (length <= count; shorter if the user pressed ESC).
    """
    cap = open_camera()
    captured: list[np.ndarray] = []
    status = ""
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        while len(captured) < count:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CameraError("Webcam stopped delivering frames.")
            lines = [f"Captured {len(captured)}/{count}   SPACE = capture   ESC = quit"]
            if hint:
                lines.append(hint)
            if status:
                lines.append(status)
            cv2.imshow(window_title, _draw_overlay(frame, lines))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("⚠️  Capture cancelled by user.")
                break
            if key == 32:  # SPACE
                if validator is not None:
                    keep, status = validator(frame)
                    print(("✅ " if keep else "❌ ") + status)
                    if not keep:
                        continue
                else:
                    status = f"Frame {len(captured) + 1} captured"
                captured.append(frame.copy())
                # brief white flash so the user sees the capture happened
                cv2.imshow(window_title, np.full_like(frame, 255))
                cv2.waitKey(80)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # lets the window actually close on Windows
    return captured
