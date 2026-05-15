"""
Import / verify dataset from dataset/ME and dataset/NOT_ME structure.

Default layout (recommended — no copy needed):
  dataset/ME/
  dataset/NOT_ME/

Optional: copy into data/raw/me and data/raw/not_me for backup.
  python scripts/import_dataset.py --copy
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def copy_images(files: list[Path], dest: Path, prefix: str):
    dest.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        ext = src.suffix.lower()
        name = f"{prefix}{src.stem}_{i:03d}{ext}"
        shutil.copy2(src, dest / name)
        print(f"  {src.name} -> {dest.name}/{name}")


def resolve_source(source: Path | None) -> Path:
    """Root folder containing ME and NOT_ME (default: project dataset/)."""
    if source:
        return source
    return config.DATASET_DIR


def get_class_dirs(source: Path) -> tuple[Path, Path]:
    """Resolve ME and NOT_ME folders (supports dataset/ME or source/ME)."""
    if (source / "ME").is_dir() and (source / "NOT_ME").is_dir():
        return source / "ME", source / "NOT_ME"
    if source.name.upper() == "ME":
        return source, source.parent / "NOT_ME"
    raise FileNotFoundError(
        f"Expected folders ME and NOT_ME under:\n  {source}\n"
        f"Example:\n  {config.DATASET_DIR / 'ME'}\n  {config.DATASET_DIR / 'NOT_ME'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Verify or import ME / NOT_ME dataset")
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=f"Dataset root (default: {config.DATASET_DIR})",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images to data/raw/me and data/raw/not_me",
    )
    args = parser.parse_args()

    source = resolve_source(Path(args.source) if args.source else None)
    print(f"Dataset root: {source}")
    print(f"Exists: {source.exists()}\n")

    if not source.exists():
        config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        config.ME_SOURCE_DIR.mkdir(exist_ok=True)
        config.NOT_ME_SOURCE_DIR.mkdir(exist_ok=True)
        print("Created empty dataset/ME and dataset/NOT_ME — add your images and run again.")
        sys.exit(1)

    me_dir, not_dir = get_class_dirs(source)
    me_files = find_images(me_dir)
    not_files = find_images(not_dir)

    print(f"ME/      ({me_dir}): {len(me_files)} images")
    print(f"NOT_ME/  ({not_dir}): {len(not_files)} images")

    if not me_files or not not_files:
        print("\nERROR: Both ME and NOT_ME must contain at least one image.")
        sys.exit(1)

    if args.copy:
        print("\nCopying to data/raw/ ...")
        copy_images(me_files, config.ME_DIR, "me_")
        copy_images(not_files, config.NOT_ME_DIR, "not_")
        print("Copy complete.")
    else:
        print("\nDataset OK. Preprocessing reads directly from dataset/ME and dataset/NOT_ME.")
        print("Next: python run_pipeline.py")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
