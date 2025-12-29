#!/usr/bin/env bash
# Multi-GPU launcher for infoseek_memtree_batch_eval.py
# Customise variables below or override via env (e.g., GPU_IDS="0 1" BATCH_SIZE=1 ./run_memtree_multigpu.sh)

set -euo pipefail

# ---------------- user settings ----------------
GPU_IDS=(${GPU_IDS:-0 1})
DATASET=${DATASET:-../benchmark/infoseek/subset/infoseek_val_5k.jsonl}
IMAGES_ROOT=${IMAGES_ROOT:-../benchmark/oven}
IMAGE_INDEX_CSV=${IMAGE_INDEX_CSV:-infoseek_image_index.csv}
COLLECTION=${COLLECTION:-memtree}
CLIP_MODEL=${CLIP_MODEL:-../ckpts/clip-vit-base-patch32}
VLM_MODEL=${VLM_MODEL:-../ckpts/Qwen2.5-VL-7B-Instruct}
BATCH_SIZE=${BATCH_SIZE:-1}
PREFETCH_BATCHES=${PREFETCH_BATCHES:-1}
RETRIEVAL_WORKERS=${RETRIEVAL_WORKERS:-4}
ROOT_TOP_K=${ROOT_TOP_K:-3}
EVENT_TOP_K=${EVENT_TOP_K:-3}
LEAF_TOP_K=${LEAF_TOP_K:-5}
ALPHA=${ALPHA:-0.1}
MAX_TREES=${MAX_TREES:-5}
MAX_SECTIONS=${MAX_SECTIONS:-5}
MAX_LEAVES=${MAX_LEAVES:-5}
MAX_CONTEXT_CHARS=${MAX_CONTEXT_CHARS:-32768}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
SHARD_DIR=${SHARD_DIR:-shards}
OUTPUT_DIR=${OUTPUT_DIR:-out}
MERGED_PATH=${MERGED_PATH:-infoseek_memtree_predictions_all_5k_collapsed_beta_0.4_update_top10.jsonl}
LOG_TO_FILE=${LOG_TO_FILE:-0}
# ------------------------------------------------

NUM_SHARDS=${NUM_SHARDS:-${#GPU_IDS[@]}}

mkdir -p "${SHARD_DIR}" "${OUTPUT_DIR}"

if [[ ! -f "${DATASET}" ]]; then
  echo "Dataset not found: ${DATASET}" >&2
  exit 1
fi

echo "Splitting dataset into ${NUM_SHARDS} shards..."
split -n l/${NUM_SHARDS} --additional-suffix .jsonl "${DATASET}" "${SHARD_DIR}/part_"

mapfile -t SHARDS < <(ls "${SHARD_DIR}"/part_* | sort)

if (( ${#SHARDS[@]} != NUM_SHARDS )); then
  echo "Shard count mismatch: expected ${NUM_SHARDS}, got ${#SHARDS[@]}" >&2
  exit 1
fi

echo "Launching ${NUM_SHARDS} workers..."
for idx in "${!SHARDS[@]}"; do
  GPU=${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}
  SHARD=${SHARDS[$idx]}
  OUT_FILE="${OUTPUT_DIR}/pred_${idx}.jsonl"
  LOG_FILE="${OUTPUT_DIR}/log_${idx}.txt"
  # Assign unique tqdm position per worker for cleaner multi-progress display
  POS=${idx}

  cmd=(python infoseek_memtree_batch_eval.py
    --dataset "${SHARD}"
    --save-path "${OUT_FILE}"
    --batch-size "${BATCH_SIZE}"
    --prefetch-batches "${PREFETCH_BATCHES}"
    --retrieval-workers "${RETRIEVAL_WORKERS}"
    --collection "${COLLECTION}"
    --clip-model "${CLIP_MODEL}"
    --vlm-model "${VLM_MODEL}"
    --root-top-k "${ROOT_TOP_K}"
    --event-top-k "${EVENT_TOP_K}"
    --leaf-top-k "${LEAF_TOP_K}"
    --alpha "${ALPHA}"
    --max-trees "${MAX_TREES}"
    --max-sections "${MAX_SECTIONS}"
    --max-leaves "${MAX_LEAVES}"
    --max-context-chars "${MAX_CONTEXT_CHARS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --tqdm-position "${POS}"
  )
  if [[ -n "${IMAGES_ROOT}" ]]; then
    cmd+=(--images-root "${IMAGES_ROOT}")
  fi
  if [[ -n "${IMAGE_INDEX_CSV}" ]]; then
    cmd+=(--image-index-csv "${IMAGE_INDEX_CSV}")
  fi

  if [[ "${LOG_TO_FILE}" -eq 1 ]]; then
    echo "Worker ${idx} -> GPU ${GPU}, log: ${LOG_FILE}"
    CUDA_VISIBLE_DEVICES=${GPU} "${cmd[@]}" >"${LOG_FILE}" 2>&1 &
  else
    echo "Worker ${idx} -> GPU ${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} "${cmd[@]}" &
  fi
done

wait
echo "All workers finished."

echo "Merging outputs -> ${MERGED_PATH}"
cat "${OUTPUT_DIR}"/pred_*.jsonl > "${MERGED_PATH}"

echo "Recomputing metrics on merged file..."
MERGED_PATH="${MERGED_PATH}" python - <<'PY'
import json
import os
from pathlib import Path
from infoseek_memtree_batch_eval import match_answer

merged_path = Path(os.environ["MERGED_PATH"]).expanduser()
if not merged_path.exists():
    raise FileNotFoundError(f"Merged file not found: {merged_path}")

records = []
with merged_path.open("r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

total = len(records)
num_correct = 0
for rec in records:
    gold = rec.get("answer_eval") or rec.get("answer") or []
    if match_answer(rec.get("model_answer", ""), gold, question=rec.get("question")):
        num_correct += 1

accuracy = num_correct / total if total else 0.0
f1 = accuracy  # precision=recall when only correctness counted
print({"total": total, "correct": num_correct, "accuracy": accuracy, "f1": f1})
PY
