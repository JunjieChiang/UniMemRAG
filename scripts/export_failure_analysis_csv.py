#!/usr/bin/env python3
'''
python scripts/export_failure_analysis_csv.py \
  --input results/v3_leaf_retrieval_text_embedding/infoseek_memtree_predictions_all_5k_collapsed_beta_0.1_update_top10.jsonl \
  --output results/v3_leaf_retrieval_text_embedding/infoseek_memtree_predictions_all_5k_collapsed_beta_0.1_update_top10.csv \
  --only-incorrect
'''

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import quote


def load_entity_map(path: Path) -> Dict[str, str]:
    entity_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data_id = rec.get("data_id")
            entity_text = rec.get("entity_text")
            if data_id and entity_text:
                entity_map[str(data_id)] = str(entity_text)
    return entity_map


def build_entity_variants(entity_text: str) -> List[str]:
    underscore = entity_text.replace(" ", "_")
    return [
        entity_text,
        underscore,
        quote(entity_text),
        quote(underscore),
    ]


def contains_entity(tree_id: str, entity_text: str) -> bool:
    if not tree_id or not entity_text:
        return False
    lowered_tree = tree_id.lower()
    for variant in build_entity_variants(entity_text):
        if variant.lower() in lowered_tree:
            return True
    return False


def compute_retrieve_correct(rec: Dict[str, Any], entity_map: Dict[str, str]) -> bool:
    data_id = rec.get("data_id")
    entity_text = entity_map.get(str(data_id), "") if data_id is not None else ""
    retrieved_roots = rec.get("retrieved_roots") or []
    if not entity_text or not retrieved_roots:
        return False
    for root in retrieved_roots:
        if contains_entity(str(root.get("tree_id", "")), entity_text):
            return True
    return False


def normalize_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fields for failure analysis from a memtree result JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="Result JSONL file.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark/infoseek_val_withkb.jsonl"),
        help="Benchmark JSONL to map data_id -> entity_text.",
    )
    parser.add_argument("--output", type=Path, help="CSV output path.")
    parser.add_argument(
        "--only-incorrect",
        action="store_true",
        help="Only include rows where is_correct is False.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Missing input file: {args.input}", file=sys.stderr)
        return 1

    entity_map: Dict[str, str] = {}
    if args.benchmark.exists():
        entity_map = load_entity_map(args.benchmark)
    else:
        print(f"Missing benchmark file: {args.benchmark}", file=sys.stderr)

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_suffix(".analysis.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "data_id",
        "image_id",
        "question",
        "answer",
        "answer_eval",
        "model_answer",
        "context",
        "retrieve_correct",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in iter_records(args.input):
            if args.only_incorrect and rec.get("is_correct"):
                continue
            retrieve_correct = compute_retrieve_correct(rec, entity_map)
            row = {
                "data_id": normalize_field(rec.get("data_id")),
                "image_id": normalize_field(rec.get("image_id")),
                "question": normalize_field(rec.get("question")),
                "answer": normalize_field(rec.get("answer")),
                "answer_eval": normalize_field(rec.get("answer_eval")),
                "model_answer": normalize_field(rec.get("model_answer")),
                "context": normalize_field(rec.get("context")),
                "retrieve_correct": int(bool(retrieve_correct)),
            }
            writer.writerow(row)

    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
