#!/usr/bin/env python3
'''
python compute_memtree_metrics.py --dir results/v4_leaf_retrieval_vit_l_p14_text_embedding
'''


from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


def compute_metrics(total: int, correct: int, retrieve_correct: int) -> dict:
    vqa_acc = correct / total if total else 0.0
    retrieve_accuracy = retrieve_correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "vqa_acc": vqa_acc,
        "retrieve_acc": retrieve_accuracy,
    }


def load_entity_map(path: Path) -> dict[str, str]:
    entity_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            data_id = rec.get("data_id")
            entity_text = rec.get("entity_text")
            if data_id and entity_text:
                entity_map[data_id] = entity_text
    return entity_map


def build_entity_variants(entity_text: str) -> list[str]:
    underscore = entity_text.replace(" ", "_")
    return [
        entity_text,
        underscore,
        quote(entity_text),
        quote(underscore),
    ]


def contains_entity(tree_id: str, entity_text: str) -> bool:
    if not tree_id:
        return False
    lowered_tree = tree_id.lower()
    for variant in build_entity_variants(entity_text):
        if variant.lower() in lowered_tree:
            return True
    return False


def load_counts(path: Path, entity_map: dict[str, str]) -> tuple[int, int, int, dict]:
    total = 0
    correct = 0
    retrieve_correct = 0
    split_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            is_correct = bool(rec.get("is_correct"))
            total += 1
            if is_correct:
                correct += 1
            data_id = rec.get("data_id")
            entity_text = entity_map.get(data_id, "")
            retrieved_roots = rec.get("retrieved_roots") or []
            retrieve_hit = False
            if entity_text and retrieved_roots:
                for root in retrieved_roots:
                    if contains_entity(str(root.get("tree_id", "")), entity_text):
                        retrieve_hit = True
                        break
            if retrieve_hit:
                retrieve_correct += 1
            split = rec.get("data_split", "unknown")
            split_counts[split][0] += 1
            if is_correct:
                split_counts[split][1] += 1
            if retrieve_hit:
                split_counts[split][2] += 1
    return total, correct, retrieve_correct, split_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics for Infoseek memtree prediction JSONL files."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        help="Explicit list of JSONL files to evaluate (overrides --pattern/beta range).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory to scan recursively for JSONL files (used when --files is not provided).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels aligned with --files. If omitted, labels are derived.",
    )
    parser.add_argument(
        "--pattern",
        default="infoseek_memtree_predictions_all_5k_collapsed_beta_{beta}_update_top{topk}.jsonl",
        help="Filename pattern, with {beta} and {topk} placeholders.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark/infoseek_val_withkb.jsonl"),
        help="Benchmark JSONL file used to map data_id -> entity_text.",
    )
    parser.add_argument(
        "--betas",
        nargs="*",
        type=float,
        help="Explicit beta list, e.g. --betas 0.0 0.1 0.2",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=0.0,
        help="Start beta (used when --betas is not provided).",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=0.9,
        help="End beta (used when --betas is not provided).",
    )
    parser.add_argument(
        "--beta-step",
        type=float,
        default=0.1,
        help="Beta step (used when --betas is not provided).",
    )
    parser.add_argument(
        "--topks",
        nargs="*",
        type=int,
        default=[5, 10],
        help="Top-k values to evaluate (default: 5 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for aggregated results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful with --output).",
    )
    return parser.parse_args()


def generate_betas(args: argparse.Namespace) -> list[float]:
    if args.betas:
        betas = sorted(set(args.betas))
        return betas
    steps = int(round((args.beta_end - args.beta_start) / args.beta_step))
    betas = [args.beta_start + i * args.beta_step for i in range(steps + 1)]
    betas = [round(beta + 1e-9, 1) for beta in betas]
    return sorted(set(betas))


def derive_label(path: Path) -> str:
    match = re.search(r"beta_(\d+(?:\.\d+)?)_.*top(\d+)", path.name)
    if match:
        return f"beta_{match.group(1)}_top{match.group(2)}"
    return path.stem


def main() -> int:
    args = parse_args()

    entity_map = {}
    if args.benchmark.exists():
        entity_map = load_entity_map(args.benchmark)
    elif not args.quiet:
        print(f"Missing benchmark file: {args.benchmark}", file=sys.stderr)

    results = []
    if args.files or args.dir:
        if args.files:
            files = [Path(p) for p in args.files]
        else:
            files = sorted(args.dir.rglob("*.jsonl"))
            if not files and not args.quiet:
                print(f"No JSONL files found under: {args.dir}", file=sys.stderr)
        labels = list(args.labels) if args.labels else [derive_label(p) for p in files]
        if len(labels) != len(files):
            raise ValueError("Length of --labels must match --files/--dir results when provided.")
        for path, label in zip(files, labels):
            if not path.exists():
                if not args.quiet:
                    print(f"Missing: {path}", file=sys.stderr)
                continue
            total, correct, retrieve_correct, split_counts = load_counts(path, entity_map)
            overall = compute_metrics(total, correct, retrieve_correct)
            split_metrics = {
                split: compute_metrics(counts[0], counts[1], counts[2])
                for split, counts in sorted(split_counts.items())
            }
            record = {
                "label": label,
                "path": str(path),
                "overall": overall,
                "splits": split_metrics,
            }
            results.append(record)
            if not args.quiet:
                print(f"{label} ({path})")
                print("overall:", overall)
                for split, metrics in split_metrics.items():
                    print(f"{split}: {metrics}")
                print("-" * 60)
    else:
        betas = generate_betas(args)
        topks = sorted(set(args.topks))
        for beta in betas:
            beta_str = f"{beta:.1f}"
            for topk in topks:
                path = Path(args.pattern.format(beta=beta_str, topk=topk))
                if not path.exists():
                    if not args.quiet:
                        print(f"Missing: {path}", file=sys.stderr)
                    continue
                total, correct, retrieve_correct, split_counts = load_counts(path, entity_map)
                overall = compute_metrics(total, correct, retrieve_correct)
                split_metrics = {
                    split: compute_metrics(counts[0], counts[1], counts[2])
                    for split, counts in sorted(split_counts.items())
                }
                record = {
                    "beta": beta,
                    "topk": topk,
                    "path": str(path),
                    "overall": overall,
                    "splits": split_metrics,
                }
                results.append(record)
                if not args.quiet:
                    print(f"beta={beta_str} topk={topk} ({path})")
                    print("overall:", overall)
                    for split, metrics in split_metrics.items():
                        print(f"{split}: {metrics}")
                    print("-" * 60)

    if args.output:
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"Wrote {len(results)} records to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
