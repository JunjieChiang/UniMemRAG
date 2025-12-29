# UniMemRAG
Unified memory representation for multimodal RAG (Memory Tree / Memory Forest).

## Overview
UniMemRAG organizes knowledge into a hierarchical Memory Tree with Root, Event, and Leaf nodes. It fuses text and image embeddings, stores them in Qdrant, and supports structured retrieval for multimodal QA evaluation.

## Core Files
- `unimemrag/memory_forest/memory_forest.py`: Memory Tree construction, ingestion, and retrieval logic. Includes Root/Event/Leaf data structures, fusion utilities, and helpers like `build_tree` and `iter_wiki_dict`.
- `infoseek_memtree_batch_eval.py`: InfoSeek evaluation with Memory Tree retrieval (with KB). Supports indexing and batch evaluation.
- `infoseek_wo_rag_batch_eval.py`: InfoSeek evaluation without KB (direct VLM baseline).

## Memory Tree Structure
- Root node: document-level topic and root image (text and image fusion).
- Event node: section summary linking Root and Leaf.
- Leaf node: paragraph-level text (supports chunking and fused leaves via `ingest_trees_new`).

## Repository Layout
```text
.
├── README.md
├── requirements.txt
├── config.py
├── unimemrag/
│   ├── embedding/
│   ├── memory_forest/
│   │   └── memory_forest.py
│   ├── utils/
│   ├── vector_store/
│   └── vlm/
├── infoseek_memtree_batch_eval.py
├── infoseek_wo_rag_batch_eval.py
├── run_memtree_multigpu.sh
├── build_infoseek_image_index.py
├── compute_memtree_metrics.py
├── benchmark/
├── examples/
```

## Requirements
- Install dependencies: `pip install -r requirements.txt`
- Local Qdrant service is required. Defaults are in `config.py` (`6336/6337`); update if needed.
- CLIP and Qwen2.5-VL checkpoints must be available (see script arguments or script defaults).

## Data Preparation
- InfoSeek dataset JSONL (question, answer, image_id). Paths are configurable via script args or script defaults.
- Images root (for example `benchmark/oven`). Building a CSV index is recommended for fast lookup.
- Example paths match script defaults; adjust if you run from a different working directory.

```bash
python build_infoseek_image_index.py --images-root benchmark/oven --output infoseek_image_index.csv
```

- For KB evaluation, prepare a wiki dict JSON (for example `wiki_*_dict*.json`) and an image cache directory.

## InfoSeek Evaluation (Memory Tree / KB)
For first-time usage, ingest the KB before running evaluation:

```bash
python infoseek_memtree_batch_eval.py \
  --ingest-kb \
  --kb-path ../benchmark/infoseek/wiki_text/wiki_100_dict_v4.json \
  --image-cache-dir ../benchmark/infoseek/wiki_text/images_100k \
  --collection memtree

python infoseek_memtree_batch_eval.py \
  --collection memtree \
  --batch-size 8 \
  --save-path infoseek_memtree_predictions.jsonl
```

Common arguments:
- `--dataset`/`--images-root`/`--image-index-csv`: dataset and image paths.
- `--retrieval-mode`: `collapsed` or `hierarchical`.
- `--root-top-k`/`--event-top-k`/`--leaf-top-k`: per-level top-k settings.
- `--alpha`: text-image fusion weight.

## Multi-GPU Runner
`run_memtree_multigpu.sh` splits the dataset into shards, launches one worker per GPU, merges predictions, and recomputes metrics on the merged file.

Example:
```bash
GPU_IDS="0 1 2 3" \
DATASET=../benchmark/infoseek/subset/infoseek_val_5k.jsonl \
MERGED_PATH=infoseek_memtree_predictions_5k.jsonl \
./run_memtree_multigpu.sh
```

You can override variables like `BATCH_SIZE`, `ROOT_TOP_K`, `ALPHA`, `OUTPUT_DIR`, and `SHARD_DIR` via environment variables.

## InfoSeek Evaluation (No KB / Baseline)
This script uses fixed paths inside the file. Update `model_path`, dataset paths, and images directory before running:

```bash
python infoseek_wo_rag_batch_eval.py
```

## Outputs and Metrics
- Evaluation outputs JSONL files (for example `infoseek_memtree_predictions.jsonl`) with `model_answer`, `is_correct`, `context`, and `retrieved_*` fields.
- The scripts print `accuracy` and `f1` metrics to stdout.
