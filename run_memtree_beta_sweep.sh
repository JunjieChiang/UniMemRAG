#!/usr/bin/env bash
# Sweep beta values, build index, run eval for each LEAF_TOP_K, then delete index.
'''
BETAS="1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0" LEAF_TOP_KS="5 10" SLEEP_SECS=60 TWO_STAGE=1 bash run_memtree_beta_sweep.sh
'''

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ---------------- user settings (override via env) ----------------
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
# GPU_IDS="${GPU_IDS:-0 4}"
LEAF_TOP_KS="${LEAF_TOP_KS:-5 10}"
BETAS="${BETAS:-}"  # Optional explicit list, e.g. "0.0 0.1 0.2"
BETA_START="${BETA_START:-0.0}"
BETA_END="${BETA_END:-1.0}"
BETA_STEP="${BETA_STEP:-0.1}"
SLEEP_SECS="${SLEEP_SECS:-60}"

TREES_JSON="${TREES_JSON:-examples/trees_5k.json}"
COLLECTION="${COLLECTION:-memtree}"
CLIP_MODEL="${CLIP_MODEL:-../ckpts/clip-vit-large-patch14}"
# LEAF_TEXT_MODEL="${LEAF_TEXT_MODEL-../ckpts/Qwen3-Embedding-0.6B}"
LEAF_TEXT_MODEL="${LEAF_TEXT_MODEL:-}"
INGEST_BATCH_SIZE="${INGEST_BATCH_SIZE:-32}"
TEXT_WORKERS="${TEXT_WORKERS:-1}"
IMAGE_WORKERS="${IMAGE_WORKERS:-8}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-8}"
LEAF_TEXT_WORKERS="${LEAF_TEXT_WORKERS:-${TEXT_WORKERS}}"
LEAF_TEXT_DEVICE="${LEAF_TEXT_DEVICE:-cuda:auto}"
TWO_STAGE="${TWO_STAGE:-1}"
ANSWER_MODE="${ANSWER_MODE:-vlm}"
TEXT_MODEL="${TEXT_MODEL:-../ckpts/Qwen2.5-VL-7B-Instruct}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
# DATASET="${DATASET:-../benchmark/infoseek/subset/infoseek_val_5k.jsonl}"
DATASET="${DATASET:-benchmark/infoseek_val.jsonl}"
IMAGES_ROOT="${IMAGES_ROOT:-../benchmark/oven}"
IMAGE_INDEX_CSV="${IMAGE_INDEX_CSV:-infoseek_image_index.csv}"
VLM_MODEL="${VLM_MODEL:-../ckpts/Qwen2.5-VL-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-2}"
RETRIEVAL_WORKERS="${RETRIEVAL_WORKERS:-8}"
ROOT_TOP_K="${ROOT_TOP_K:-3}"
EVENT_TOP_K="${EVENT_TOP_K:-3}"
ALPHA="${ALPHA:-0.05}"
MAX_TREES="${MAX_TREES:-5}"
MAX_SECTIONS="${MAX_SECTIONS:-5}"
MAX_LEAVES="${MAX_LEAVES:-5}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SHARD_DIR_BASE="${SHARD_DIR_BASE:-shards}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-out}"
LOG_TO_FILE="${LOG_TO_FILE:-0}"
# ------------------------------------------------------------------

IFS=' ' read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ID_ARR[@]}}"

generate_betas() {
  if [[ -n "${BETAS}" ]]; then
    for b in ${BETAS}; do
      echo "${b}"
    done
    return
  fi
  BETA_START="${BETA_START}" BETA_END="${BETA_END}" BETA_STEP="${BETA_STEP}" python - <<'PY'
from decimal import Decimal
import os

start = Decimal(os.environ["BETA_START"])
end = Decimal(os.environ["BETA_END"])
step = Decimal(os.environ["BETA_STEP"])

decimals = max(-step.as_tuple().exponent, 0)
value = start
epsilon = Decimal("0.0000001")
while value <= end + epsilon:
    print(format(value, f".{decimals}f"))
    value += step
PY
}

build_index() {
  local beta="$1"
  BETA="${beta}" TREES_JSON="${TREES_JSON}" COLLECTION="${COLLECTION}" CLIP_MODEL="${CLIP_MODEL}" \
  INGEST_BATCH_SIZE="${INGEST_BATCH_SIZE}" TEXT_WORKERS="${TEXT_WORKERS}" IMAGE_WORKERS="${IMAGE_WORKERS}" \
  LEAF_TEXT_MODEL="${LEAF_TEXT_MODEL}" LEAF_TEXT_DEVICE="${LEAF_TEXT_DEVICE}" \
  LEAF_TEXT_WORKERS="${LEAF_TEXT_WORKERS}" TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE}" \
  python - <<'PY'
import json
import os
import sys
from pathlib import Path

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

sys.path.append(os.getcwd())

from config import Config
from unimemrag.retriever import ClipEmbedding
from unimemrag.embedding.models.QwenTextEmbedding import QwenTextEmbedding
from unimemrag.memory_forest.memory_forest import MemoryForestStore, MemoryTree, RootNode, EventNode, LeafNode

trees_path = Path(os.environ["TREES_JSON"]).expanduser()
if not trees_path.exists():
    raise FileNotFoundError(f"Trees JSON not found: {trees_path}")

cfg = Config(collection=os.environ["COLLECTION"])
embed_model = ClipEmbedding(model_name=os.environ["CLIP_MODEL"])
leaf_text_model = os.environ.get("LEAF_TEXT_MODEL") or ""
leaf_text_device = os.environ.get("LEAF_TEXT_DEVICE") or None
leaf_text_embedder = None
leaf_text_vector_size = None
if leaf_text_model:
    leaf_text_embedder = QwenTextEmbedding(model_name=leaf_text_model, device=leaf_text_device)
    leaf_text_vector_size = leaf_text_embedder.dim
memforest_store = MemoryForestStore(
    cfg,
    vector_size=embed_model.dim,
    leaf_text_vector_size=leaf_text_vector_size,
)

with trees_path.open("r", encoding="utf-8") as fh:
    data = json.load(fh)

def tree_from_dict(d):
    root = RootNode(**d["root"])
    events = [EventNode(**e) for e in d.get("events", [])]
    leaves = [LeafNode(**l) for l in d.get("leaves", [])]
    return MemoryTree(tree_id=d["tree_id"], root=root, events=events, leaves=leaves)

trees = [tree_from_dict(x) for x in data]

beta = float(os.environ["BETA"])
text_batch_size = os.environ.get("TEXT_BATCH_SIZE") or ""
text_batch_size = int(text_batch_size) if text_batch_size else None
memforest_store.ingest_trees_new(
    trees,
    embed_model,
    beta=beta,
    batch_size=int(os.environ["INGEST_BATCH_SIZE"]),
    text_batch_size=text_batch_size,
    text_workers=int(os.environ["TEXT_WORKERS"]),
    image_workers=int(os.environ["IMAGE_WORKERS"]),
    leaf_text_embedder=leaf_text_embedder,
    leaf_text_workers=int(os.environ["LEAF_TEXT_WORKERS"]),
    show_progress=True,
)
PY
}

delete_index() {
  COLLECTION="${COLLECTION}" CLIP_MODEL="${CLIP_MODEL}" python - <<'PY'
import os
import sys

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

sys.path.append(os.getcwd())

from config import Config
from unimemrag.embedding.models.ClipEmbedding import ClipEmbedding
from unimemrag.memory_forest import MemoryForestStore

cfg = Config(collection=os.environ["COLLECTION"])
embed_model = ClipEmbedding(model_name=os.environ["CLIP_MODEL"])
memforest_store = MemoryForestStore(cfg, vector_size=embed_model.dim)

def safe_delete(name):
    try:
        memforest_store.client.delete_collection(name)
        print(f"Deleted collection: {name}")
    except Exception as exc:
        print(f"Skip delete (not found or failed): {name} -> {exc}")

safe_delete(memforest_store.cfg.collection)
safe_delete(memforest_store.event_collection)
safe_delete(memforest_store.leaf_collection)
PY
}

if [[ ! -f "${TREES_JSON}" ]]; then
  echo "Trees JSON not found: ${TREES_JSON}" >&2
  exit 1
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "Dataset not found: ${DATASET}" >&2
  exit 1
fi

mapfile -t BETAS_LIST < <(generate_betas)
if (( ${#BETAS_LIST[@]} == 0 )); then
  echo "No beta values to run." >&2
  exit 1
fi

for beta in "${BETAS_LIST[@]}"; do
  echo "==== Beta ${beta}: build index ===="
  build_index "${beta}"
  echo "Sleep ${SLEEP_SECS}s before evaluation..."
  sleep "${SLEEP_SECS}"

  for leaf_top_k in ${LEAF_TOP_KS}; do
    run_id="beta_${beta}_top${leaf_top_k}"
    shard_dir="${SHARD_DIR_BASE}/${run_id}"
    output_dir="${OUTPUT_DIR_BASE}/${run_id}"
    merged_path="infoseek_memtree_predictions_all_collapsed_beta_${beta}_update_top${leaf_top_k}.jsonl"

    rm -f "${shard_dir}"/part_* 2>/dev/null || true
    rm -f "${output_dir}"/pred_*.jsonl "${output_dir}"/log_*.txt 2>/dev/null || true

    echo "---- Eval: beta=${beta}, LEAF_TOP_K=${leaf_top_k} -> ${merged_path}"
    GPU_IDS="${GPU_IDS}" NUM_SHARDS="${NUM_SHARDS}" BATCH_SIZE="${BATCH_SIZE}" \
    PREFETCH_BATCHES="${PREFETCH_BATCHES}" RETRIEVAL_WORKERS="${RETRIEVAL_WORKERS}" \
    LEAF_TOP_K="${leaf_top_k}" MERGED_PATH="${merged_path}" SHARD_DIR="${shard_dir}" \
    OUTPUT_DIR="${output_dir}" DATASET="${DATASET}" IMAGES_ROOT="${IMAGES_ROOT}" \
    IMAGE_INDEX_CSV="${IMAGE_INDEX_CSV}" COLLECTION="${COLLECTION}" CLIP_MODEL="${CLIP_MODEL}" \
    LEAF_TEXT_MODEL="${LEAF_TEXT_MODEL}" LEAF_TEXT_DEVICE="${LEAF_TEXT_DEVICE}" \
    VLM_MODEL="${VLM_MODEL}" ANSWER_MODE="${ANSWER_MODE}" TEXT_MODEL="${TEXT_MODEL}" \
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" TWO_STAGE="${TWO_STAGE}" ROOT_TOP_K="${ROOT_TOP_K}" EVENT_TOP_K="${EVENT_TOP_K}" \
    ALPHA="${ALPHA}" MAX_TREES="${MAX_TREES}" MAX_SECTIONS="${MAX_SECTIONS}" \
    MAX_LEAVES="${MAX_LEAVES}" MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" LOG_TO_FILE="${LOG_TO_FILE}" \
    bash run_memtree_multigpu.sh

    echo "Sleep ${SLEEP_SECS}s before next task..."
    sleep "${SLEEP_SECS}"
  done

  echo "==== Beta ${beta}: delete index ===="
  delete_index
  echo "Sleep ${SLEEP_SECS}s before next beta..."
  sleep "${SLEEP_SECS}"
done

echo "All sweeps finished."
