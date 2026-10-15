"""
main.py - Menu-driven runner for the Face Attendance System.

    python main.py

The embedding model is loaded once (on first use) and shared by every action.
"""

from __future__ import annotations

import config
from attendance_logger import print_today
from build_db import build_database, load_db, print_db_summary
from model_loader import FaceEmbedder, get_embedder
from register import register_user
from verify import verify_loop, verify_once

MENU = """
╔══════════════════════════════════════╗
║        FACE ATTENDANCE SYSTEM        ║
╠══════════════════════════════════════╣
║  1. Register a new person            ║
║  2. Verify & mark attendance (once)  ║
║  3. Verify in a loop                 ║
║  4. Show today's attendance          ║
║  5. List registered people           ║
║  6. Rebuild embeddings database      ║
║  7. Tune threshold                   ║
║  0. Exit                             ║
╚══════════════════════════════════════╝"""


def main() -> None:
    config.ensure_dirs()
    embedder: FaceEmbedder | None = None

    def model() -> FaceEmbedder:
        nonlocal embedder
        if embedder is None:
            embedder = get_embedder()
        return embedder

    print(f"📂 Data folder : {config.BASE_DIR}")
    print(f"🧠 Model source: "
          f"{config.CUSTOM_MODEL_PATH if __import__('os').path.exists(config.CUSTOM_MODEL_PATH) else 'DeepFace ' + config.DEEPFACE_MODEL_NAME}")

    while True:
        print(MENU)
        choice = input("Select an option: ").strip()
        try:
            if choice == "1":
                register_user(embedder=model())
            elif choice == "2":
                verify_once(embedder=model())
            elif choice == "3":
                verify_loop(embedder=model())
            elif choice == "4":
                print_today()
            elif choice == "5":
                print_db_summary(load_db())
            elif choice == "6":
                build_database(model())
            elif choice == "7":
                import tune_threshold
                tune_threshold.main()
            elif choice in ("0", "q", "exit"):
                print("👋 Bye!")
                break
            else:
                print("❓ Unknown option.")
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted; back to menu.")


if __name__ == "__main__":
    main()
