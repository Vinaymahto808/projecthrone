"""
build_db.py - Face database (pickle) helpers and full rebuild script.

Database layout (embeddings/face_db.pkl):
    {
        "version": 1,
        "model_id": "Facenet",          # which embedder produced the vectors
        "embedding_size": 128,
        "people": {
            "Vinay": {
                "embeddings": np.ndarray (n, 128)  # L2-normalised, one per image
                "files":      ["...\\known_faces\\Vinay\\Vinay_01.jpg", ...]
            },
            ...
        },
    }

Run `python build_db.py` to (re)compute embeddings for every image under
known_faces/. This is needed after changing the model, or if you copied
images into known_faces/ manually instead of using register.py.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Any

import numpy as np

import config
from model_loader import FaceEmbedder, get_embedder

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DB_VERSION = 1


def new_db(embedder: FaceEmbedder) -> dict[str, Any]:
    """Return an empty database tagged with the embedder's identity."""
    return {
        "version": DB_VERSION,
        "model_id": embedder.model_id,
        "embedding_size": embedder.embedding_size,
        "people": {},
    }


def load_db() -> dict[str, Any] | None:
    """Load the pickle, or return None if it does not exist / is corrupt."""
    if not os.path.exists(config.FACE_DB_PATH):
        return None
    try:
        with open(config.FACE_DB_PATH, "rb") as fh:
            db = pickle.load(fh)
        if not isinstance(db, dict) or "people" not in db:
            raise ValueError("unexpected structure")
        return db
    except Exception as exc:  # corrupt / incompatible pickle
        print(f"⚠️  Could not read {config.FACE_DB_PATH}: {exc}. It will be rebuilt.")
        return None


def save_db(db: dict[str, Any]) -> None:
    """Atomically write the database pickle."""
    config.ensure_dirs()
    tmp = config.FACE_DB_PATH + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(db, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, config.FACE_DB_PATH)


def db_matches(db: dict[str, Any] | None, embedder: FaceEmbedder) -> bool:
    """True if `db` was produced by the same model as `embedder`."""
    return (
        db is not None
        and db.get("model_id") == embedder.model_id
        and int(db.get("embedding_size", -1)) == embedder.embedding_size
    )


def add_embeddings(db: dict[str, Any], name: str, embeddings: list[np.ndarray], files: list[str]) -> None:
    """Append per-image embeddings for `name`, skipping files already present."""
    person = db["people"].setdefault(name, {"embeddings": np.empty((0, db["embedding_size"]), np.float32), "files": []})
    known = set(os.path.normcase(f) for f in person["files"])
    keep_vecs, keep_files = [], []
    for vec, path in zip(embeddings, files):
        if os.path.normcase(path) in known:
            continue
        keep_vecs.append(np.asarray(vec, dtype=np.float32).ravel())
        keep_files.append(path)
    if keep_vecs:
        person["embeddings"] = np.vstack([person["embeddings"], np.stack(keep_vecs)])
        person["files"].extend(keep_files)


def list_person_images(person_dir: str) -> list[str]:
    """Sorted image paths inside one person's folder."""
    paths: list[str] = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(person_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(person_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def build_database(embedder: FaceEmbedder | None = None, verbose: bool = True) -> dict[str, Any]:
    """Scan known_faces/<name>/*.jpg, embed every image, save and return the DB."""
    embedder = embedder or get_embedder()
    config.ensure_dirs()
    db = new_db(embedder)

    people = sorted(
        d for d in os.listdir(config.KNOWN_FACES_DIR)
        if os.path.isdir(os.path.join(config.KNOWN_FACES_DIR, d))
    )
    if verbose:
        print(f"🔄 Building face database from {config.KNOWN_FACES_DIR}")
        print(f"   Model: {embedder.describe()}")
    if not people:
        print("⚠️  known_faces/ is empty. Run register.py first.")

    total_ok = total_bad = 0
    for name in people:
        person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
        images = list_person_images(person_dir)
        vecs, files = [], []
        for path in images:
            vec = embedder.embed_file(path)
            if vec is None:
                total_bad += 1
                if verbose:
                    print(f"   ❌ {name}: no face in {os.path.basename(path)} (skipped)")
                continue
            vecs.append(vec)
            files.append(path)
        total_ok += len(vecs)
        if vecs:
            add_embeddings(db, name, vecs, files)
            if verbose:
                print(f"   ✅ {name}: {len(vecs)}/{len(images)} images embedded")
        elif verbose:
            print(f"   ⚠️  {name}: no usable images")

    save_db(db)
    if verbose:
        print(f"💾 Saved {config.FACE_DB_PATH}")
        print(f"   {len(db['people'])} people, {total_ok} embeddings, {total_bad} images skipped")
    return db


def print_db_summary(db: dict[str, Any] | None) -> None:
    """Console summary of who is registered."""
    if not db or not db["people"]:
        print("📭 Face database is empty. Register someone first.")
        return
    print(f"📚 Face database: model={db.get('model_id')} dim={db.get('embedding_size')}")
    for name, person in sorted(db["people"].items()):
        print(f"   • {name:<20} {len(person['files'])} images")


if __name__ == "__main__":
    build_database()
