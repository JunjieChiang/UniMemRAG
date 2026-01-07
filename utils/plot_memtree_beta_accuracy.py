#!/usr/bin/env python3
"""
Plot VQA and retrieval accuracy vs beta for memtree results.

Example:
python utils/plot_memtree_beta_accuracy.py \
    --dir results/v4_leaf_retrieval_vit_l_p14_text_embedding
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from compute_memtree_metrics import compute_metrics, load_counts, load_entity_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs beta for memtree result JSONL files."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing memtree prediction JSONL files.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark/infoseek_val_withkb.jsonl"),
        help="Benchmark JSONL file used to map data_id -> entity_text.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write plots (default: <dir>/plots).",
    )
    return parser.parse_args()


def find_result_files(results_dir: Path) -> list[tuple[Path, float, int]]:
    pattern = re.compile(r"beta_(\d+(?:\.\d+)?)_.*top(\d+)\.jsonl$")
    found: list[tuple[Path, float, int]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        match = pattern.search(path.name)
        if not match:
            continue
        beta = float(match.group(1))
        topk = int(match.group(2))
        found.append((path, beta, topk))
    return found


def collect_metrics(
    files: list[tuple[Path, float, int]], entity_map: dict[str, str]
) -> dict[int, dict[float, dict]]:
    results: dict[int, dict[float, dict]] = {}
    for path, beta, topk in files:
        total, correct, retrieve_correct, _ = load_counts(path, entity_map)
        metrics = compute_metrics(total, correct, retrieve_correct)
        results.setdefault(topk, {})[beta] = metrics
    return results


def plot_metric(
    metric_key: str,
    title: str,
    output_path: Path,
    results_by_topk: dict[int, dict[float, dict]],
) -> None:
    all_betas = sorted({beta for data in results_by_topk.values() for beta in data})
    all_values: list[float] = []
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for topk in sorted(results_by_topk):
        beta_map = results_by_topk[topk]
        betas = sorted(beta_map)
        values = [beta_map[beta][metric_key] for beta in betas]
        all_values.extend(values)
        ax.plot(betas, values, marker="o", linewidth=2, label=f"recall@{topk}")
    ax.set_xlabel("beta")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    if all_betas:
        ax.set_xticks(all_betas)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if all_values:
        vmin = min(all_values)
        vmax = max(all_values)
        if vmax - vmin < 1e-6:
            pad = 0.01
        else:
            pad = max(0.005, (vmax - vmin) * 0.15)
        ax.set_ylim(max(0.0, vmin - pad), min(1.0, vmax + pad))
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.dir.exists():
        raise FileNotFoundError(f"Missing results dir: {args.dir}")

    entity_map = load_entity_map(args.benchmark) if args.benchmark.exists() else {}
    results_dir = args.dir
    output_dir = args.output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_result_files(results_dir)
    if not files:
        raise FileNotFoundError(f"No matching JSONL files found in {results_dir}")

    results_by_topk = collect_metrics(files, entity_map)
    plot_metric(
        "vqa_acc",
        "VQA Accuracy vs Beta",
        output_dir / "vqa_accuracy_vs_beta.png",
        results_by_topk,
    )
    plot_metric(
        "retrieve_acc",
        "Retrieval Accuracy vs Beta",
        output_dir / "retrieval_accuracy_vs_beta.png",
        results_by_topk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
