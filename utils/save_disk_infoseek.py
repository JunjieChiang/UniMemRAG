#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python save_disk_infoseek.py \
  --oven_root ../benchmark/oven \
  --index_csv infoseek_image_index.csv \
  --val_jsonl benchmark/infoseek_val.jsonl \
  --trash_dir ../benchmark/ovem_trash \
  --dry_run
'''

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Set, Dict


def load_needed_ids(val_jsonl: Path) -> Set[str]:
    ids = set()
    with val_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "image_id" in obj:
                ids.add(str(obj["image_id"]))
    return ids


def load_index_map(index_csv: Path) -> Dict[str, str]:
    """
    image_id -> relative_path
    """
    mapping = {}
    with index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["image_id"]] = row["relative_path"].lstrip("/")
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oven_root", required=True, help="Path to oven/ directory")
    ap.add_argument("--index_csv", required=True, help="infoseek_image_index.csv")
    ap.add_argument("--val_jsonl", required=True, help="infoseek_val_annotation.jsonl")
    ap.add_argument("--trash_dir", required=True, help="Directory to move unused images")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    oven_root = Path(args.oven_root).resolve()
    trash_dir = Path(args.trash_dir).resolve()
    trash_dir.mkdir(parents=True, exist_ok=True)

    needed_ids = load_needed_ids(Path(args.val_jsonl))
    index_map = load_index_map(Path(args.index_csv))

    # image paths we want to keep (relative to oven_root)
    keep_relpaths = set()

    for iid in needed_ids:
        if iid not in index_map:
            raise RuntimeError(f"image_id not found in index csv: {iid}")
        keep_relpaths.add(index_map[iid])

    kept = 0
    moved = 0

    for img_path in oven_root.rglob("*.jpg"):
        rel = img_path.relative_to(oven_root).as_posix()
        if rel in keep_relpaths:
            kept += 1
            continue

        dst = trash_dir / rel
        if args.dry_run:
            print(f"[DRY] move {img_path} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_path), str(dst))
        moved += 1

    print("=== DONE ===")
    print(f"kept images : {kept}")
    print(f"moved images: {moved}")
    print(f"trash dir   : {trash_dir}")

    if args.dry_run:
        print("[DRY RUN] no files were modified")


if __name__ == "__main__":
    main()