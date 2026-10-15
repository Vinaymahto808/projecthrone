"""
opencv_embedder.py - TensorFlow-free face embedding backend.

Uses OpenCV's built-in DNN runtime with two small Apache-2.0 ONNX models from
the opencv_zoo repository (bundled under models/):

  * YuNet - face detection  (face_detection_yunet_2023mar.onnx, ~230 KB)
  * SFace - 128-d recognition embeddings (face_recognition_sface_2021dec.onnx, ~37 MB)

This backend exists so the API can run on serverless platforms (Vercel) where
TensorFlow alone (~1.4 GB installed) blows past the 500 MB function-size limit.
It exposes exactly the same interface as model_loader.FaceEmbedder, so
server.py, build_db.py, register.py and verify.py work unchanged.

Select it with FACE_EMBEDDER_BACKEND=opencv, or leave the default "auto" and
it will be used automatically whenever deepface/TensorFlow is not installed.

Recognition thresholds: SFace uses the same cosine-similarity scale as the
DeepFace models, so config.DEFAULT_THRESHOLDS["SFace"] (0.41) applies.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

import config  # noqa: F401  (import first: sets TF log level / UTF-8 console)
from model_loader import l2_normalize

YUNET_MODEL = os.path.join(config.BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(config.BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")

# YuNet's default score threshold (0.9) is too strict for a kiosk camera;
# 0.6 keeps distant/partial faces while still filtering noise.
SCORE_THRESHOLD = 0.6


class OpenCVEmbedder:
    """Detect faces (YuNet) and embed them (SFace) with plain OpenCV."""

    model_id: str = "SFace"
    input_size: tuple[int, int] = (112, 112)  # (width, height)
    embedding_size: int = 128
    _is_custom: bool = False

    def __init__(self) -> None:
        missing = [p for p in (YUNET_MODEL, SFACE_MODEL) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing OpenCV face model(s): " + ", ".join(missing)
            )
        self._detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL, "", (320, 320), score_threshold=SCORE_THRESHOLD
        )
        self._recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

    # ---------------------------------------------------------------- detect
    def detect_faces(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        """Return detected faces sorted largest-first.

        Each item has keys "face" (RGB float32 0..1 aligned 112x112 crop),
        "facial_area" (dict x,y,w,h,left,right,top,bottom,...) and "confidence".
        An empty list means no face was found.
        """
        img_h, img_w = bgr.shape[:2]
        self._detector.setInputSize((img_w, img_h))
        _ok, boxes = self._detector.detect(bgr)

        faces: list[dict[str, Any]] = []
        if boxes is None:
            return faces
        for row in boxes:
            # YuNet rows are [x, y, w, h, re, le, nt, rcm, lcm, score] -> 15 values
            # (bbox top-left + size, 5 landmarks, confidence).
            x, y, w, h = (float(v) for v in row[:4])
            score = float(row[-1])
            if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
                continue

            # alignCrop wants the raw detection row: it uses the 5 landmarks to
            # align the face to SFace's canonical 112x112 pose.
            detection = np.asarray(row, dtype=np.float32).reshape(1, -1)
            aligned = self._recognizer.alignCrop(bgr, detection)
            face_rgb01 = aligned[..., ::-1].astype(np.float32) / 255.0

            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(img_w - 1), x + w), min(float(img_h - 1), y + h)
            faces.append(
                {
                    "face": face_rgb01,
                    "confidence": round(score, 4),
                    "facial_area": {
                        "x": int(x1),
                        "y": int(y1),
                        "w": int(x2 - x1),
                        "h": int(y2 - y1),
                        "left": int(x1),
                        "top": int(y1),
                        "right": int(x2),
                        "bottom": int(y2),
                        "confidence": round(score, 4),
                    },
                }
            )
        faces.sort(key=lambda o: o["facial_area"]["w"] * o["facial_area"]["h"], reverse=True)
        return faces

    # ----------------------------------------------------------------- embed
    def embed_face(self, face_rgb01: np.ndarray) -> np.ndarray:
        """Embedding for an already-detected aligned face crop."""
        face = np.clip(np.asarray(face_rgb01, dtype=np.float32), 0.0, 1.0)
        bgr = (face[..., ::-1] * 255.0).astype(np.uint8)
        if bgr.shape[:2] != (self.input_size[1], self.input_size[0]):
            bgr = cv2.resize(bgr, self.input_size)
        bgr = np.ascontiguousarray(bgr)
        vec = np.asarray(self._recognizer.feature(bgr), dtype=np.float32)
        return l2_normalize(vec)

    def embed(self, bgr: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        """Detect the largest face in a BGR frame and embed it."""
        faces = self.detect_faces(bgr)
        if not faces:
            return None, None
        return self.embed_face(faces[0]["face"]), faces[0]["facial_area"]

    def embed_file(self, path: str) -> np.ndarray | None:
        """Embed the largest face in an image file (None if unreadable/no face)."""
        img = cv2.imread(path)
        if img is None:
            return None
        emb, _ = self.embed(img)
        return emb

    def describe(self) -> str:
        w, h = self.input_size
        return f"{self.model_id} (OpenCV YuNet+SFace ONNX) | input {w}x{h} | embedding {self.embedding_size}-d"
