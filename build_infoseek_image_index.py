#!/usr/bin/env python3
"""
Build a CSV mapping InfoSeek image IDs to their actual file paths.

This helps resolve misplaced buckets quickly by avoiding per-request directory scans.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

from infoseek_image_utils import ALLOWED_IMAGE_EXTS, resolve_images_root


def _discover_images(images_root: Path, show_progress: bool) -> Tuple[Dict[str, Path], int]:
    mapping: Dict[str, Path] = {}
    duplicates = 0
    iterator = images_root.rglob("*")
    if show_progress:
        iterator = tqdm(iterator, desc="Indexing images", unit="file", leave=False)

    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_IMAGE_EXTS:
            continue
        image_id = path.stem
        rel_path = path.relative_to(images_root)
        if image_id in mapping:
            duplicates += 1
            continue
        mapping[image_id] = rel_path

    return mapping, duplicates


def _write_csv(index: Dict[str, Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "relative_path"])
        writer.writeheader()
        for image_id, rel_path in sorted(index.items()):
            writer.writerow({"image_id": image_id, "relative_path": rel_path.as_posix()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CSV index for InfoSeek images.")
    parser.add_argument("--images-root", default="../benchmark/oven/", help="Root directory containing bucketed images.")
    parser.add_argument(
        "--output",
        default="infoseek_image_index.csv",
        help="CSV output path (stores image_id and relative_path columns).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar while indexing.",
    )
    args = parser.parse_args()

    images_root = resolve_images_root(args.images_root)
    output_path = Path(args.output).expanduser().resolve()

    index, duplicates = _discover_images(images_root, show_progress=not args.no_progress)
    _write_csv(index, output_path)

    print(f"Wrote {len(index)} records to {output_path}")
    if duplicates:
        print(f"Skipped {duplicates} duplicate image_ids (kept the first occurrence).")


if __name__ == "__main__":
    main()
