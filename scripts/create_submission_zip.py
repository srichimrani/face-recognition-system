"""
Create submission ZIP excluding venv and caches.
  python scripts/create_submission_zip.py
"""
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", "node_modules", ".cursor"}
SKIP_EXT = {".pyc", ".pyo"}
OUT = ROOT / f"face_recognition_submission_{datetime.now().strftime('%Y%m%d')}.zip"


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    if path.suffix.lower() in SKIP_EXT:
        return True
    return False


def main():
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ROOT.rglob("*"):
            if f.is_file() and not should_skip(f):
                rel = f.relative_to(ROOT)
                zf.write(f, rel)
                count += 1
    print(f"Created: {OUT}")
    print(f"Files packed: {count}")
    print("Submit this ZIP + PDF report to your university portal.")


if __name__ == "__main__":
    main()
