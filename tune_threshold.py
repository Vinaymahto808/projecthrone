"""
tune_threshold.py - Pick a similarity threshold from your own registered data.

How it works
------------
Using the per-image embeddings already stored in face_db.pkl:
  * GENUINE scores  = cosine similarity between two images of the SAME person
  * IMPOSTOR scores = cosine similarity between images of DIFFERENT people

A good threshold sits above (almost) every impostor score and below (almost)
every genuine score. The script prints both distributions, an ASCII histogram,
and the threshold that maximises balanced accuracy. Put that value into
config.THRESHOLD.

You need at least 2 registered people (for impostor pairs) and 2+ images each
(for genuine pairs). More people and more varied photos = better estimate.

Usage:
    python tune_threshold.py
"""

from __future__ import annotations

import itertools

import numpy as np

import config
from build_db import load_db


def collect_scores(db: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (genuine_scores, impostor_scores) from the database."""
    people = {n: p["embeddings"] for n, p in db["people"].items() if len(p["embeddings"]) > 0}
    genuine: list[float] = []
    impostor: list[float] = []

    for embs in people.values():
        if len(embs) < 2:
            continue
        sims = embs @ embs.T
        iu = np.triu_indices(len(embs), k=1)
        genuine.extend(sims[iu].tolist())

    for (_, a), (_, b) in itertools.combinations(people.items(), 2):
        impostor.extend((a @ b.T).ravel().tolist())

    return np.asarray(genuine, dtype=np.float32), np.asarray(impostor, dtype=np.float32)


def best_threshold(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Threshold maximising balanced accuracy = (TPR + TNR) / 2."""
    best_t, best_acc = 0.5, -1.0
    for t in np.arange(-0.2, 1.0001, 0.005):
        tpr = float((genuine >= t).mean()) if len(genuine) else 1.0
        tnr = float((impostor < t).mean()) if len(impostor) else 1.0
        acc = (tpr + tnr) / 2
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def _stats(name: str, arr: np.ndarray) -> None:
    if len(arr) == 0:
        print(f"   {name:<9} n=0 (not enough data)")
        return
    print(
        f"   {name:<9} n={len(arr):<5} min={arr.min():.3f}  mean={arr.mean():.3f}  "
        f"max={arr.max():.3f}  p5={np.percentile(arr, 5):.3f}  p95={np.percentile(arr, 95):.3f}"
    )


def _histogram(genuine: np.ndarray, impostor: np.ndarray, bins: int = 24) -> None:
    lo = min(genuine.min() if len(genuine) else 0, impostor.min() if len(impostor) else 0, 0.0)
    edges = np.linspace(lo, 1.0, bins + 1)
    g_hist, _ = np.histogram(genuine, bins=edges)
    i_hist, _ = np.histogram(impostor, bins=edges)
    scale = max(1, max(g_hist.max() if len(genuine) else 0, i_hist.max() if len(impostor) else 0))
    print("\n   similarity   impostor (x) | genuine (#)")
    for k in range(bins):
        x = "x" * int(round(30 * i_hist[k] / scale))
        g = "#" * int(round(30 * g_hist[k] / scale))
        print(f"   {edges[k]:>6.2f}-{edges[k+1]:<5.2f} {x:<30}| {g}")


def main() -> None:
    db = load_db()
    if db is None or not db["people"]:
        print("📭 Face database is empty. Register at least 2 people first.")
        return

    genuine, impostor = collect_scores(db)
    model_id = db.get("model_id", "custom")
    current = config.get_threshold(model_id)

    print(f"📊 Threshold analysis for model '{model_id}' ({len(db['people'])} people)")
    _stats("genuine", genuine)
    _stats("impostor", impostor)

    if len(genuine) == 0 or len(impostor) == 0:
        print("\n⚠️  Need 2+ people with 2+ images each to tune. Current threshold stays", f"{current:.3f}")
        return

    _histogram(genuine, impostor)

    t, acc = best_threshold(genuine, impostor)
    gap_lo, gap_hi = float(impostor.max()), float(genuine.min())
    print(f"\n   Current threshold : {current:.3f}")
    print(f"   Suggested         : {t:.3f}  (balanced accuracy {acc:.1%})")
    if gap_lo < gap_hi:
        print(f"   Clean separation  : impostor max {gap_lo:.3f} < genuine min {gap_hi:.3f}")
        print(f"   Safe middle value : {(gap_lo + gap_hi) / 2:.3f}")
    else:
        print(f"   ⚠️  Overlap: impostor max {gap_lo:.3f} >= genuine min {gap_hi:.3f}.")
        print("      Add more varied photos per person, or accept some errors.")
    print(f"\n👉 Set THRESHOLD = {t:.3f} in config.py (raise it if strangers get accepted,")
    print("   lower it if real users are rejected).")


if __name__ == "__main__":
    main()
