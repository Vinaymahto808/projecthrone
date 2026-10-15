"""
liveness.py - Anti-spoofing and face-quality checks for the kiosk.

Industry kiosks are attacked with printed photos, phone screens replaying a
video, and someone else "helping" a coworker punch in. This module applies
three cheap checks that defeat the common attacks without a dedicated
anti-spoof model:

1. **Motion** - consecutive frames of a *live* person always differ by a
   small amount (breathing, micro-movement, sensor noise). A printed photo or
   a frozen video frame held up to the camera produces near-zero difference.
   Checks the mean absolute difference of the aligned face crops.

2. **Cross-frame identity** - every frame must match the SAME person above
   the recognition threshold. A helper leaning in halfway through the burst
   makes the per-frame identities disagree.

3. **Quality gates** - blur (Laplacian variance) and brightness bounds on the
   face crop. Blown-out / too-dark / out-of-focus faces are rejected before
   they ever reach the matcher, which is what actually keeps FAR low in
   production systems.

The kiosk sends LIVENESS_FRAMES images per verification. With a single image
(legacy clients, desktop tools) motion and identity checks cannot run and the
result is "skipped" - quality gates still apply.

All thresholds come from config.py.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import config


def quality_gate(face_bgr: np.ndarray) -> dict[str, Any]:
    """Blur + brightness checks on a face crop (BGR, any size).

    Returns {"passed": bool, "blur": float, "brightness": float, "reason": str}.
    """
    if face_bgr is None or face_bgr.size == 0:
        return {"passed": False, "blur": 0.0, "brightness": 0.0, "reason": "empty face crop"}
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    lo, hi = config.BRIGHTNESS_RANGE
    if not (lo <= brightness <= hi):
        return {"passed": False, "blur": round(blur, 1), "brightness": round(brightness, 1),
                "reason": f"face lighting too {'dark' if brightness < lo else 'bright'} - move to better light"}
    if blur < config.MIN_BLUR_VAR:
        return {"passed": False, "blur": round(blur, 1), "brightness": round(brightness, 1),
                "reason": "face is too blurry - hold still and improve lighting"}
    return {"passed": True, "blur": round(blur, 1), "brightness": round(brightness, 1), "reason": ""}


def _aligned_face_bgr(frame_bgr: np.ndarray, area: dict[str, Any]) -> np.ndarray:
    """Crop the facial area from the frame, resized to a fixed 128x128 grid."""
    x, y, w, h = int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"])
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    return cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)


def motion_score(face_a: np.ndarray, face_b: np.ndarray) -> float:
    """Mean absolute difference between two aligned face crops, in percent."""
    if face_a.shape != face_b.shape:
        face_b = cv2.resize(face_b, (face_a.shape[1], face_a.shape[0]))
    diff = cv2.absdiff(face_a, face_b)
    return float(diff.mean()) / 255.0 * 100.0


def assess_liveness(
    frames: list[np.ndarray],
    embeddings: list[np.ndarray | None],
    areas: list[dict[str, Any] | None],
    pairwise_similarity: float | None = None,
) -> dict[str, Any]:
    """Run the burst checks. Frames/embeddings/areas are aligned by index.

    Args:
        frames: BGR frames in capture order (>= 1).
        embeddings: L2-normalised embedding per frame (None when no face).
        areas: facial_area per frame (None when no face).
        pairwise_similarity: min cosine similarity between consecutive frame
            embeddings; computed here when omitted.

    Returns:
        {
          "passed": bool,          # False only when a live person is ruled out
          "status": "passed" | "skipped" | "failed",
          "motion_pct": float,     # mean of consecutive-crop differences
          "identity_sim": float,   # min consecutive embedding similarity
          "reason": str,
        }
    """
    result: dict[str, Any] = {
        "passed": True, "status": "skipped", "motion_pct": None,
        "identity_sim": None, "reason": "",
    }
    valid = [i for i, e in enumerate(embeddings) if e is not None]
    if len(valid) < 2 or len(frames) < 2:
        result["reason"] = "single frame - motion liveness skipped"
        return result

    # --- 1. motion between consecutive frames -------------------------------
    crops = [_aligned_face_bgr(frames[i], areas[i]) for i in valid if areas[i] is not None]
    if len(crops) >= 2:
        diffs = [motion_score(crops[i], crops[i + 1]) for i in range(len(crops) - 1)]
        motion = float(np.mean(diffs))
        result["motion_pct"] = round(motion, 3)
        if motion < config.LIVENESS_MIN_MOTION:
            result.update(
                passed=False, status="failed",
                reason="no face movement between frames - printed photo or frozen screen?",
            )
            return result

    # --- 2. same person across every frame ----------------------------------
    sims = [
        float(np.dot(embeddings[valid[i]], embeddings[valid[i + 1]]))
        for i in range(len(valid) - 1)
    ]
    identity = float(np.min(sims)) if pairwise_similarity is None else float(pairwise_similarity)
    result["identity_sim"] = round(identity, 4)
    if identity < config.LIVENESS_MIN_EMB_SIM:
        result.update(
            passed=False, status="failed",
            reason="different faces across frames - only one person may be scanned",
        )
        return result

    result.update(status="passed", reason="motion + identity checks passed")
    return result


def summarise_quality(gates: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-frame quality gates into one verdict for the API response."""
    if not gates:
        return {"passed": False, "reason": "no frames"}
    failed = next((g for g in gates if not g["passed"]), None)
    if failed is not None:
        return {"passed": False, "reason": failed["reason"], "details": gates}
    return {
        "passed": True,
        "reason": "",
        "avg_blur": round(float(np.mean([g["blur"] for g in gates])), 1),
        "avg_brightness": round(float(np.mean([g["brightness"] for g in gates])), 1),
        "details": gates,
    }
