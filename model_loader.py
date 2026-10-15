"""
model_loader.py - Loads the face embedding model and turns images into vectors.

Model priority
--------------
1. config.CUSTOM_MODEL_PATH (my_deepface_model.h5) if the file exists. It is
   loaded with tf.keras using config.CUSTOM_OBJECTS and pre-processed with
   config.CUSTOM_MODEL_NORMALIZATION / CUSTOM_MODEL_COLOR so inference matches
   training exactly.
2. Otherwise DeepFace's pretrained config.DEEPFACE_MODEL_NAME (default
   "Facenet": 160x160 RGB input scaled to 0..1, 128-d output).

In both cases the pipeline is identical:
    BGR frame -> DeepFace.extract_faces (detect + align, RGB 0..1)
              -> pad/resize to model input -> normalise -> model
              -> L2-normalised embedding

Embeddings are L2-normalised, so cosine similarity is a plain dot product.

NOTE: DeepFace.verify() is NOT used anywhere. Only the detector and the
embedding network are used, so swapping in your own model is a config change.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

import config  # noqa: F401  (import first: sets TF log level / UTF-8 console)


class FaceNotFound(Exception):
    """Raised when no usable face is present in an image."""


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length (no-op for the zero vector)."""
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1] between two embeddings."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def _pad_resize(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize keeping aspect ratio, then zero-pad to exactly (h, w, 3).

    Mirrors DeepFace's own `preprocessing.resize_image`, so pretrained models
    see the same geometry they were evaluated with.
    """
    target_w, target_h = size_wh
    factor = min(target_h / img.shape[0], target_w / img.shape[1])
    new_w = max(1, int(img.shape[1] * factor))
    new_h = max(1, int(img.shape[0] * factor))
    img = cv2.resize(img, (new_w, new_h))
    pad_h = target_h - img.shape[0]
    pad_w = target_w - img.shape[1]
    img = np.pad(
        img,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
        mode="constant",
    )
    if img.shape[:2] != (target_h, target_w):
        img = cv2.resize(img, (target_w, target_h))
    return img.astype(np.float32)


class FaceEmbedder:
    """Detect a face in a BGR image and produce an L2-normalised embedding."""

    def __init__(self) -> None:
        self.model_id: str = ""
        self.input_size: tuple[int, int] = (0, 0)  # (width, height)
        self.embedding_size: int = 0
        self._model: Any = None
        self._is_custom: bool = False

        if os.path.exists(config.CUSTOM_MODEL_PATH):
            self._load_custom(config.CUSTOM_MODEL_PATH)
        else:
            self._load_deepface(config.DEEPFACE_MODEL_NAME)

    # ------------------------------------------------------------------ load
    def _load_custom(self, path: str) -> None:
        """Load a user-trained Keras .h5 embedding model."""
        import tensorflow as tf

        print(f"🧠 Loading custom model: {path}")
        self._model = tf.keras.models.load_model(
            path, compile=False, custom_objects=config.CUSTOM_OBJECTS or None
        )
        shape = self._model.input_shape
        if isinstance(shape, list):  # multi-input models: use the first input
            shape = shape[0]
        _, h, w, _c = shape
        self.input_size = (int(w), int(h))
        self.embedding_size = int(self._model.output_shape[-1])
        self.model_id = "custom"
        self._is_custom = True

    def _load_deepface(self, name: str) -> None:
        """Build one of DeepFace's pretrained recognition networks."""
        from deepface.modules import modeling

        print(f"🧠 Loading DeepFace pretrained model: {name} (custom .h5 not found)")
        self._model = modeling.build_model(task="facial_recognition", model_name=name)
        h, w = self._model.input_shape  # DeepFace stores (height, width)
        self.input_size = (int(w), int(h))
        self.embedding_size = int(self._model.output_shape)
        self.model_id = name
        self._is_custom = False

    # ---------------------------------------------------------------- detect
    def detect_faces(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        """Return detected faces sorted largest-first.

        Each item has keys "face" (RGB float32 0..1 aligned crop),
        "facial_area" (dict x,y,w,h,...) and "confidence".
        An empty list means no face was found.
        """
        from deepface import DeepFace

        try:
            objs = DeepFace.extract_faces(
                img_path=bgr,
                detector_backend=config.DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
                color_face="rgb",
                normalize_face=True,
            )
        except ValueError:
            return []

        img_h, img_w = bgr.shape[:2]
        faces: list[dict[str, Any]] = []
        for obj in objs:
            area = obj["facial_area"]
            # With enforce_detection=False DeepFace returns the whole frame
            # (confidence 0, size within a pixel of the image) when nothing is
            # found. Drop that placeholder: no real face covers 90% of a frame.
            area_ratio = (area["w"] * area["h"]) / float(img_w * img_h)
            if obj.get("confidence", 0) <= 0 and area_ratio >= 0.9:
                continue
            if area["w"] < config.MIN_FACE_SIZE or area["h"] < config.MIN_FACE_SIZE:
                continue
            faces.append(obj)
        faces.sort(key=lambda o: o["facial_area"]["w"] * o["facial_area"]["h"], reverse=True)
        return faces

    # ------------------------------------------------------------ preprocess
    def preprocess(self, face_rgb01: np.ndarray) -> np.ndarray:
        """Turn an aligned RGB 0..1 crop into a (1, h, w, 3) model input."""
        img = _pad_resize(face_rgb01, self.input_size)
        if self._is_custom:
            if config.CUSTOM_MODEL_COLOR.lower() == "bgr":
                img = img[..., ::-1]
            norm = config.CUSTOM_MODEL_NORMALIZATION
            if norm == "0-1":
                pass
            elif norm == "0-255":
                img = img * 255.0
            elif norm == "-1-1":
                img = img * 2.0 - 1.0
            elif norm == "imagenet":
                img = img * 255.0
                img = (img - np.array([123.68, 116.779, 103.939], dtype=np.float32)) / np.array(
                    [58.393, 57.12, 57.375], dtype=np.float32
                )
            else:
                raise ValueError(f"Unknown CUSTOM_MODEL_NORMALIZATION: {norm!r}")
        # DeepFace pretrained models use normalization="base" = keep 0..1.
        return np.expand_dims(np.ascontiguousarray(img, dtype=np.float32), axis=0)

    # ----------------------------------------------------------------- embed
    def embed_face(self, face_rgb01: np.ndarray) -> np.ndarray:
        """Embedding for an already-detected aligned face crop."""
        batch = self.preprocess(face_rgb01)
        if self._is_custom:
            out = self._model(batch, training=False)
            vec = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        else:
            vec = np.asarray(self._model.forward(batch), dtype=np.float32)
        return l2_normalize(vec)

    def embed(self, bgr: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        """Detect the largest face in a BGR frame and embed it.

        Returns:
            (embedding, facial_area) or (None, None) if no face was found.
        """
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
        kind = "custom .h5" if self._is_custom else "DeepFace pretrained"
        return f"{self.model_id} ({kind}) | input {w}x{h} | embedding {self.embedding_size}-d"


_EMBEDDER: FaceEmbedder | None = None


def get_embedder() -> FaceEmbedder:
    """Return the process-wide embedder, loading the model on first use."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = FaceEmbedder()
        print(f"✅ Model ready: {_EMBEDDER.describe()}")
    return _EMBEDDER


if __name__ == "__main__":
    # Smoke test: load the model and print its shape info.
    emb = get_embedder()
    print(f"Threshold in use: {config.get_threshold(emb.model_id):.3f}")
