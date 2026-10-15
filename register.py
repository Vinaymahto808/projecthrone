"""
register.py - Step 1: enrol a person.

Flow
----
1. Ask for the person's name (or pass it on the command line).
2. Open the webcam. Press SPACE to capture; frames without a detectable face
   are rejected on the spot so you never save a bad sample.
3. Save config.IMAGES_PER_USER images to known_faces/<name>/<name>_NN.jpg.
4. Embed the new images with the active model and update embeddings/face_db.pkl.

Usage:
    python register.py              # prompts for the name
    python register.py "Vinay"      # name given directly
"""

from __future__ import annotations

import argparse
import os
import re

import cv2

import config
from attendance_logger import HEADER  # noqa: F401  (ensures module import order is sane)
from build_db import add_embeddings, build_database, db_matches, list_person_images, load_db, new_db, save_db
from camera import CameraError, capture_frames
from model_loader import FaceEmbedder, get_embedder


def sanitize_name(raw: str) -> str:
    """Make a name safe for use as a folder/file name on Windows."""
    name = raw.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    return name[:60]


def _next_index(person_dir: str) -> int:
    """1-based index for the next image file in `person_dir`."""
    return len(list_person_images(person_dir)) + 1


def register_user(name: str | None = None, embedder: FaceEmbedder | None = None) -> bool:
    """Capture images for one person and add them to the face database.

    Returns:
        True if at least one image was saved and embedded.
    """
    embedder = embedder or get_embedder()
    config.ensure_dirs()

    if not name:
        name = input("👤 Enter the person's name: ")
    name = sanitize_name(name or "")
    if not name:
        print("❌ Name cannot be empty.")
        return False

    person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
    existing = list_person_images(person_dir) if os.path.isdir(person_dir) else []
    if existing:
        print(f"ℹ️  '{name}' already has {len(existing)} images.")
        answer = input("   Add more images to this person? [y/N]: ").strip().lower()
        if answer != "y":
            print("⚠️  Registration cancelled.")
            return False
    os.makedirs(person_dir, exist_ok=True)

    def has_face(frame) -> tuple[bool, str]:
        faces = embedder.detect_faces(frame)
        if not faces:
            return False, "No face detected - move closer / improve lighting"
        if len(faces) > 1:
            return False, f"{len(faces)} faces in frame - only one person please"
        return True, "Face captured"

    print(f"\n📸 Capturing {config.IMAGES_PER_USER} images for '{name}'.")
    print("   Look at the camera and vary your head angle/expression slightly between shots.")
    print("   SPACE = capture, ESC = stop early.\n")
    try:
        frames = capture_frames(
            config.IMAGES_PER_USER,
            window_title=f"Register: {name}",
            hint="Vary angle slightly between captures",
            validator=has_face,
        )
    except CameraError as exc:
        print(f"❌ {exc}")
        return False

    if not frames:
        print("❌ No images captured. Nothing saved.")
        return False
    if len(frames) < config.IMAGES_PER_USER:
        print(f"⚠️  Only {len(frames)}/{config.IMAGES_PER_USER} images captured; continuing with those.")

    # ---- save to disk -----------------------------------------------------
    saved_paths: list[str] = []
    idx = _next_index(person_dir)
    for frame in frames:
        path = os.path.join(person_dir, f"{name}_{idx:02d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        saved_paths.append(path)
        idx += 1
    print(f"💾 Saved {len(saved_paths)} images to {person_dir}")

    # ---- embed + update database -------------------------------------------
    print("🔄 Computing embeddings...")
    vecs, files = [], []
    for path in saved_paths:
        vec = embedder.embed_file(path)  # re-read from disk: same pipeline as build_db
        if vec is None:
            print(f"   ❌ Face lost after JPEG save in {os.path.basename(path)}; skipped")
            continue
        vecs.append(vec)
        files.append(path)
    if not vecs:
        print("❌ None of the captured images produced an embedding. Try again with better lighting.")
        return False

    db = load_db()
    if not db_matches(db, embedder):
        if db is not None:
            print("ℹ️  Existing database was built with a different model; rebuilding everything.")
        db = build_database(embedder, verbose=False)
        # build_database already embedded the new files from disk; nothing to add.
    else:
        add_embeddings(db, name, vecs, files)
        save_db(db)

    total = len(db["people"][name]["files"])
    print(f"✅ Registered '{name}' with {len(vecs)} new embeddings ({total} total).")
    print(f"   Database: {config.FACE_DB_PATH}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a person for face attendance.")
    parser.add_argument("name", nargs="?", help="Person's name (prompted if omitted)")
    args = parser.parse_args()
    register_user(args.name)


if __name__ == "__main__":
    main()
