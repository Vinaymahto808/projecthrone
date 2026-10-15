"""
verify.py - Step 2: recognise a person and mark attendance.

Flow
----
1. Load embeddings/face_db.pkl (must have been built by the same model).
2. Capture one webcam frame (or read --image <file>).
3. Detect + embed the largest face, compare with every stored embedding using
   cosine similarity, and take the best-scoring person.
4. If best score >= threshold -> match -> append to attendance.csv
   (one row per person per day; repeats are reported, not re-logged).

Usage:
    python verify.py                 # one webcam verification
    python verify.py --loop          # keep verifying until ESC / Ctrl+C
    python verify.py --image me.jpg  # verify from a file instead of webcam
    python verify.py --no-mark       # test recognition without logging
"""

from __future__ import annotations

import argparse
from typing import Any

import cv2
import numpy as np

import config
from attendance_logger import mark_attendance
from build_db import db_matches, load_db, print_db_summary
from camera import CameraError, capture_frames
from model_loader import FaceEmbedder, get_embedder


def match_embedding(embedding: np.ndarray, db: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Compare one embedding against every registered person.

    Per person we take the MAX similarity over their stored images (nearest
    sample). The person with the highest max wins; it is a match only if that
    score reaches `threshold`.

    Returns:
        {
          "matched": bool, "name": str | None, "score": float,
          "threshold": float,
          "candidates": [(name, best_score, mean_score), ...]  # sorted desc
        }
    """
    q = np.asarray(embedding, dtype=np.float32).ravel()
    candidates: list[tuple[str, float, float]] = []
    for name, person in db["people"].items():
        embs = person["embeddings"]
        if embs is None or len(embs) == 0:
            continue
        sims = embs @ q  # both sides L2-normalised -> cosine similarity
        candidates.append((name, float(sims.max()), float(sims.mean())))
    candidates.sort(key=lambda c: c[1], reverse=True)

    if not candidates:
        return {"matched": False, "name": None, "score": 0.0, "threshold": threshold, "candidates": []}

    best_name, best_score, _ = candidates[0]
    matched = best_score >= threshold
    return {
        "matched": matched,
        "name": best_name if matched else None,
        "score": best_score,
        "threshold": threshold,
        "candidates": candidates,
    }


def _print_result(result: dict[str, Any]) -> None:
    thr = result["threshold"]
    print(f"\n🔎 Top candidates (threshold {thr:.3f}):")
    for name, best, mean in result["candidates"][:3]:
        flag = "✅" if best >= thr else "  "
        print(f"   {flag} {name:<20} best={best:.3f}  mean={mean:.3f}")


def verify_frame(
    frame: np.ndarray,
    embedder: FaceEmbedder,
    db: dict[str, Any],
    mark: bool = True,
) -> dict[str, Any]:
    """Run recognition on a single BGR frame and optionally log attendance."""
    threshold = config.get_threshold(embedder.model_id)
    embedding, area = embedder.embed(frame)
    if embedding is None:
        print("❌ No face detected. Face the camera, move closer, or improve lighting.")
        return {"matched": False, "name": None, "score": 0.0, "threshold": threshold, "candidates": [], "face": False}

    result = match_embedding(embedding, db, threshold)
    result["face"] = True
    result["facial_area"] = area
    _print_result(result)

    if not result["matched"]:
        print(f"❌ Unknown person (best score {result['score']:.3f} < {threshold:.3f}). Attendance NOT marked.")
        return result

    name = result["name"]
    if mark:
        if mark_attendance(name):
            print(f"✅ Welcome {name}! Attendance marked (score {result['score']:.3f}).")
            result["marked"] = True
        else:
            print(f"ℹ️  {name} is already marked present today (score {result['score']:.3f}).")
            result["marked"] = False
    else:
        print(f"✅ Recognised {name} (score {result['score']:.3f}). Marking disabled (--no-mark).")
        result["marked"] = False
    return result


def _load_ready_db(embedder: FaceEmbedder) -> dict[str, Any] | None:
    db = load_db()
    if db is None or not db["people"]:
        print("📭 No registered faces. Run register.py first.")
        return None
    if not db_matches(db, embedder):
        print(
            f"⚠️  face_db.pkl was built with model '{db.get('model_id')}' "
            f"({db.get('embedding_size')}-d) but the active model is '{embedder.model_id}' "
            f"({embedder.embedding_size}-d). Run: python build_db.py"
        )
        return None
    return db


def verify_once(embedder: FaceEmbedder | None = None, mark: bool = True, image_path: str | None = None) -> dict[str, Any] | None:
    """Capture (or load) one image and verify it. Returns the result dict or None."""
    embedder = embedder or get_embedder()
    db = _load_ready_db(embedder)
    if db is None:
        return None

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Could not read image: {image_path}")
            return None
    else:
        print("\n📸 Look at the camera and press SPACE to verify (ESC to cancel).")
        try:
            frames = capture_frames(1, window_title="Verify attendance", hint="Press SPACE when ready")
        except CameraError as exc:
            print(f"❌ {exc}")
            return None
        if not frames:
            return None
        frame = frames[0]

    return verify_frame(frame, embedder, db, mark=mark)


def verify_loop(embedder: FaceEmbedder | None = None, mark: bool = True) -> None:
    """Repeat verification until the user cancels."""
    embedder = embedder or get_embedder()
    print("🔁 Loop mode: ESC in the camera window or Ctrl+C to stop.")
    try:
        while True:
            result = verify_once(embedder, mark=mark)
            if result is None:
                break
    except KeyboardInterrupt:
        print("\n👋 Stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a face and mark attendance.")
    parser.add_argument("--image", help="Verify this image file instead of using the webcam")
    parser.add_argument("--no-mark", action="store_true", help="Recognise only; do not write attendance.csv")
    parser.add_argument("--loop", action="store_true", help="Keep verifying until cancelled")
    parser.add_argument("--list", action="store_true", help="Print registered people and exit")
    args = parser.parse_args()

    if args.list:
        print_db_summary(load_db())
        return
    if args.loop:
        verify_loop(mark=not args.no_mark)
    else:
        verify_once(mark=not args.no_mark, image_path=args.image)


if __name__ == "__main__":
    main()
