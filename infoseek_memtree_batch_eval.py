"""
For index has built:
    - python infoseek_memtree_batch_eval.py --collection memtree --batch-size 8 --save-path infoseek_memtree_predictions.jsonl

For index has not built:
    - python infoseek_memtree_batch_eval.py --ingest-kb --kb-path ../benchmark/infoseek/wiki_text/wiki_100_dict_v4.json --image-cache-dir ../benchmark/infoseek/wiki_text/images_100k
"""


import argparse
import csv
import json
import os
import re
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from unimemrag.embedding.models.ClipEmbedding import ClipEmbedding
from unimemrag.embedding.models.QwenTextEmbedding import QwenTextEmbedding
from unimemrag.memory_forest.memory_forest import (
    MemoryForestStore,
    CollapsedRetrievalResult,
    TreeRetrievalResult,
    build_tree,
    iter_wiki_dict,
)
from unimemrag.utils.image_cache import (
    download_images_for_kb,
    load_image_cache,
    replace_payload_image_urls,
    save_image_cache,
)
from unimemrag.vlm.QwenVL import QwenVL
from unimemrag.llm.QwenText import Qwen
from infoseek_image_utils import ALLOWED_IMAGE_EXTS, resolve_images_root

_IMAGE_INDEX_CACHE: Dict[Tuple[Path, Path], Dict[str, Path]] = {}


def _disable_proxies() -> None:
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def load_infoseek_dataset(path: Union[str, Path]) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dataset.append(json.loads(line))
    return dataset


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _quick_match(prediction: str, gold_answers: Sequence[str]) -> bool:
    if not prediction:
        return False
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    for ans in gold_answers:
        if not ans:
            continue
        gold_norm = normalize_answer(ans)
        if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
            return True
    return False


def match_answer(
    prediction: str,
    gold_answers: Sequence[str],
    *,
    question: Optional[str] = None,
    judge_vlm: Optional[object] = None,
    max_new_tokens: int = 64,
) -> bool:
    _ = (question, judge_vlm, max_new_tokens)
    if not gold_answers:
        return False
    return _quick_match(prediction, gold_answers)


def _load_image_index(images_root: Path, image_index_csv: Path) -> Dict[str, Path]:
    cache_key = (images_root, image_index_csv)
    cached = _IMAGE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mapping: Dict[str, Path] = {}
    with image_index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image_id" not in reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise ValueError(f"Image index CSV must contain 'image_id' and 'relative_path' columns: {image_index_csv}")
        for row in reader:
            image_id = Path((row.get("image_id") or "").strip()).stem
            rel_path_str = (row.get("relative_path") or "").strip()
            if not image_id or not rel_path_str:
                continue
            rel_path = Path(rel_path_str)
            if rel_path.suffix.lower() not in ALLOWED_IMAGE_EXTS:
                continue
            mapping[image_id] = rel_path

    _IMAGE_INDEX_CACHE[cache_key] = mapping
    return mapping


def resolve_infoseek_image(
    image_id: Union[str, Path],
    images_root: Optional[Union[str, Path]] = None,
    image_index_csv: Optional[Union[str, Path]] = None,
) -> Path:
    image_stem = Path(image_id).stem
    id_parts = image_stem.split("_")
    if len(id_parts) != 2 or len(id_parts[1]) < 2:
        raise ValueError(f"Unexpected image id format: {image_id!r}")

    bucket = id_parts[1][:2]

    images_root = resolve_images_root(images_root)

    if image_index_csv:
        image_index_path = Path(image_index_csv).expanduser()
        if not image_index_path.exists():
            raise FileNotFoundError(f"Image index CSV not found: {image_index_path}")
        index = _load_image_index(images_root, image_index_path)
        rel_path = index.get(image_stem)
        if rel_path:
            abs_path = (images_root / rel_path).resolve()
            if abs_path.exists():
                return abs_path
            raise FileNotFoundError(f"Indexed image path not found: {abs_path}")

    bucket_dir = images_root / bucket
    if not bucket_dir.exists():
        raise FileNotFoundError(f"Image bucket not found: {bucket_dir}")

    candidates = sorted(bucket_dir.glob(f"{image_stem}.*"))
    for candidate in candidates:
        if candidate.suffix.lower() in ALLOWED_IMAGE_EXTS:
            return candidate
    raise FileNotFoundError(f"Image file not found for stem {image_stem!r} in {bucket_dir}")


def format_tree_context(
    tree_result: TreeRetrievalResult,
    *,
    max_sections: int,
    max_leaves: int,
) -> str:
    root = tree_result.root
    meta = root.payload.get("metadata", {}) or {}
    lines = [
        f"Topic: {root.payload.get('topic') or meta.get('source_url', 'n/a')}",
        f"Tree ID: {tree_result.tree_id}",
        f"Root score: {root.score:.4f}",
    ]
    alignment_score = meta.get("alignment_best_score")
    if alignment_score is not None:
        lines.append(f"Alignment score: {alignment_score}")
    lines.append("")

    for event in tree_result.events[:max_sections]:
        emeta = event.payload.get("metadata", {}) or {}
        title = emeta.get("section_title") or event.payload.get("summary") or "Unknown section"
        lines.append(f"Section: {title}")
        section_preview = (emeta.get("section_preview") or event.payload.get("summary") or "").strip()
        if section_preview:
            lines.append(section_preview)
        leaf_hits = tree_result.leaves.get(event.id, [])
        for idx, leaf_hit in enumerate(leaf_hits[:max_leaves], start=1):
            snippet = (leaf_hit.payload.get("content") or "").strip()
            if not snippet:
                continue
            lines.append(f"Paragraph {idx}: {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def build_context(
    results: Sequence[Union[TreeRetrievalResult, CollapsedRetrievalResult]],
    *,
    max_trees: int,
    max_sections: int,
    max_leaves: int,
    max_chars: int,
) -> str:
    if not results:
        return ""
    chunks: List[str] = []
    for idx, result in enumerate(results[:max_trees], start=1):
        if isinstance(result, TreeRetrievalResult):
            chunk = format_tree_context(result, max_sections=max_sections, max_leaves=max_leaves)
        else:
            chunk = format_collapsed_context(result, max_sections=max_sections, max_leaves=max_leaves)
        if not chunk:
            continue
        chunks.append(f"[Tree {idx}]\n{chunk}")
    context = "\n\n".join(chunks).strip()
    if max_chars and max_chars > 0 and len(context) > max_chars:
        truncated = context[:max_chars]
        if "\n" in truncated:
            truncated = truncated.rsplit("\n", 1)[0]
        context = truncated
    return context


def build_infoseek_message(
    question: str,
    context: str,
    image_path: Optional[Union[str, Path]] = None,
    *,
    include_image: bool = True,
) -> List[Dict[str, Any]]:
    if context:
        prompt = f"Here's the context:\n{context}\n\nNow, answer the question:\n{question}"
        system_text = "You are a helpful assistant. Please answer the question based on the context provided."
    else:
        prompt = f"Answer the question:\n{question}"
        system_text = "You are a helpful assistant. Please answer the user's question."

    if include_image and image_path is not None:
        return [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Path(image_path).as_posix()},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt},
    ]


def format_collapsed_context(
    tree_result: CollapsedRetrievalResult,
    *,
    max_sections: int,
    max_leaves: int,
) -> str:
    lines = [
        f"Tree ID: {tree_result.tree_id}",
    ]
    for event in tree_result.events[:max_sections]:
        emeta = event.payload.get("metadata", {}) or {}
        summary_text = (event.payload.get("summary") or "").strip()
        title = (str(emeta.get("section_title") or "")).strip() or summary_text or "Unknown section"
        lines.append(f"Section: {title}")
        section_summary = summary_text
        if section_summary:
            lines.append(section_summary)
        leaf_hits = tree_result.leaves.get(event.id, [])
        for idx, leaf_hit in enumerate(leaf_hits[:max_leaves], start=1):
            snippet = (leaf_hit.payload.get("content") or "").strip()
            if not snippet:
                continue
            lines.append(f"Paragraph {idx}: {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def build_retrieval_metadata(
    results: Sequence[Union[TreeRetrievalResult, CollapsedRetrievalResult]],
) -> Dict[str, Any]:
    roots: List[Dict[str, Any]] = []
    for res in results:
        if isinstance(res, TreeRetrievalResult):
            roots.append(
                {
                    "tree_id": res.tree_id,
                    "score": res.root.score,
                    "topic": res.root.payload.get("topic"),
                }
            )
        else:
            roots.append({"tree_id": res.tree_id, "score": None, "topic": None})
    event_count = sum(len(res.events) for res in results)
    leaf_count = sum(len(leaves) for res in results for leaves in res.leaves.values())
    return {
        "retrieved_roots": roots,
        "retrieved_root_count": len(results),
        "retrieved_event_count": event_count,
        "retrieved_leaf_count": leaf_count,
    }


def maybe_ingest_kb(
    store: MemoryForestStore,
    embedder: ClipEmbedding,
    leaf_text_embedder: Optional[QwenTextEmbedding],
    *,
    kb_path: Path,
    image_cache_dir: Path,
    image_cache_index: Path,
    download_images: bool,
    ingest_batch_size: int,
    text_workers: int,
    image_workers: int,
    leaf_text_workers: Optional[int],
    alpha: float,
    show_progress: bool,
) -> None:
    with kb_path.open("r", encoding="utf-8") as fh:
        kb = json.load(fh)

    image_cache = load_image_cache(image_cache_index)
    if not image_cache and download_images:
        image_cache = download_images_for_kb(kb, image_cache_dir, max_workers=64, resume=True)
        save_image_cache(image_cache, image_cache_index)

    def localize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        return replace_payload_image_urls(dict(payload), image_cache)

    total = len(kb)
    iterator = iter_wiki_dict(kb)
    if show_progress:
        iterator = tqdm(iterator, total=total, desc="Building trees", leave=False)

    trees = []
    for wiki_url, payload in iterator:
        payload = localize_payload(payload)
        trees.append(build_tree(wiki_url, payload))

    store.ingest_trees_new(
        trees,
        embedder,
        beta=alpha,
        batch_size=ingest_batch_size,
        text_workers=text_workers,
        image_workers=image_workers,
        leaf_text_embedder=leaf_text_embedder,
        leaf_text_workers=leaf_text_workers or text_workers,
        show_progress=show_progress,
    )


def run_infoseek_evaluation(
    answer_model: Any,
    memforest_store: MemoryForestStore,
    embedder: ClipEmbedding,
    leaf_text_embedder: Optional[QwenTextEmbedding],
    dataset: Sequence[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
    images_root: Optional[Union[str, Path]] = None,
    image_index_csv: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    max_new_tokens: int = 4096,
    batch_size: int = 8,
    root_top_k: int = 3,
    event_top_k: int = 3,
    leaf_top_k: int = 10,
    alpha: float = 0.1,
    max_trees: int = 3,
    max_sections: int = 3,
    max_leaves: int = 2,
    max_context_chars: int = 2048,
    retrieval_workers: int = 4,
    prefetch_batches: int = 4,
    tqdm_position: int = 0,
    retrieval_mode: str = "collapsed",
    two_stage: bool = True,
    include_images_in_messages: bool = True,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if retrieval_workers <= 0:
        retrieval_workers = 1
    if prefetch_batches <= 0:
        prefetch_batches = 1

    max_inflight = max(batch_size * prefetch_batches, batch_size)

    records: List[Dict[str, Any]] = []
    skipped = 0
    iterable = dataset if limit is None else dataset[:limit]
    iterator = tqdm(iterable, desc="Evaluating InfoSeek", leave=False, position=tqdm_position) if show_progress else iterable

    pending_examples: List[Dict[str, Any]] = []
    pending_messages: List[List[Dict[str, Any]]] = []

    def _log(message: str) -> None:
        if show_progress and hasattr(iterator, "write"):
            iterator.write(message)
        else:
            # print(message)
            pass

    def _flush_batch() -> None:
        nonlocal pending_examples, pending_messages, skipped
        if not pending_messages:
            return

        try:
            if len(pending_messages) > 1 and hasattr(answer_model, "chat_batch"):
                predictions = answer_model.chat_batch(pending_messages, max_new_tokens=max_new_tokens)
            else:
                predictions = [answer_model.chat(pending_messages[0], max_new_tokens=max_new_tokens)]
        except Exception as exc:
            _log(f"Batch inference failed ({exc}); falling back to sequential execution.")
            predictions = []
            for item, messages in zip(pending_examples, pending_messages):
                try:
                    predictions.append(answer_model.chat(messages, max_new_tokens=max_new_tokens))
                except Exception as inner_exc:
                    example = item.get("example", {})
                    data_id = example.get("data_id") or example.get("image_id") or "unknown"
                    _log(f"Skipping {data_id}: {inner_exc}")
                    skipped += 1
                    predictions.append(None)

        if len(predictions) != len(pending_examples):
            _log("Mismatch between predictions and examples; skipping incomplete results.")
            skipped += len(pending_examples)
            pending_examples = []
            pending_messages = []
            return

        for item, prediction in zip(pending_examples, predictions):
            if prediction is None:
                continue
            example = item.get("example", {})
            gold_candidates = example.get("answer_eval") or example.get("answer") or []
            correct = match_answer(
                prediction,
                gold_candidates,
                question=example.get("question"),
                judge_vlm=answer_model,
            )

            record = dict(example)
            record["model_answer"] = prediction
            record["is_correct"] = bool(correct)
            record["context"] = item.get("context", "")
            record.update(item.get("retrieval", {}))
            records.append(record)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pending_examples = []
        pending_messages = []

    def _prepare_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            image_path = resolve_infoseek_image(
                example["image_id"],
                images_root=images_root,
                image_index_csv=image_index_csv,
            )
        except FileNotFoundError as exc:
            data_id = example.get("data_id") or example.get("image_id") or "unknown"
            _log(f"Skipping {data_id}: {exc}")
            return None

        question = example.get("question", "")
        # results = memforest_store.retrieve(
        #     embedder,
        #     query_text=question,
        #     query_image=image_path.as_posix(),
        #     root_top_k=root_top_k,
        #     event_top_k=event_top_k,
        #     leaf_top_k=leaf_top_k,
        #     alpha=alpha,
        # )
        if retrieval_mode == "hierarchical":
            results = memforest_store.retrieve(
                embedder,
                query_text=question,
                query_image=image_path.as_posix(),
                root_top_k=root_top_k,
                event_top_k=event_top_k,
                leaf_top_k=leaf_top_k,
                alpha=alpha,
            )
        else:
            results = memforest_store.collapsed_retrieve(
                embedder,
                query_text=question,
                query_image=image_path.as_posix(),
                leaf_text_embedder=leaf_text_embedder,
                two_stage=two_stage,
                leaf_top_k=leaf_top_k,
                alpha=alpha,
            )
        context = build_context(
            results,
            max_trees=max_trees,
            max_sections=max_sections,
            max_leaves=max_leaves,
            max_chars=max_context_chars,
        )
        messages = build_infoseek_message(
            question,
            context=context,
            image_path=image_path,
            include_image=include_images_in_messages,
        )
        retrieval_meta = build_retrieval_metadata(results)
        return {
            "example": example,
            "context": context,
            "retrieval": retrieval_meta,
            "messages": messages,
        }

    done_queue: "queue.Queue[Any]" = queue.Queue()
    inflight = 0

    def _submit(example: Dict[str, Any], executor: ThreadPoolExecutor) -> None:
        nonlocal inflight
        future = executor.submit(_prepare_example, example)
        future.add_done_callback(done_queue.put)
        inflight += 1

    def _drain_one(block: bool = False) -> bool:
        nonlocal inflight, skipped
        try:
            fut = done_queue.get(block=block, timeout=1 if block else 0)
        except queue.Empty:
            return False
        inflight -= 1
        try:
            prepared = fut.result()
        except Exception as exc:  # pragma: no cover - defensive
            _log(f"Skipping example due to worker error: {exc}")
            skipped += 1
            return True
        if prepared is None:
            skipped += 1
            return True

        pending_examples.append(prepared)
        pending_messages.append(prepared["messages"])
        if len(pending_messages) >= batch_size:
            _flush_batch()
        return True

    with ThreadPoolExecutor(max_workers=retrieval_workers) as executor:
        for example in iterator:
            _submit(example, executor)
            while inflight >= max_inflight:
                _drain_one(block=True)

        while inflight > 0:
            _drain_one(block=True)

    _flush_batch()

    total = len(records)
    num_correct = sum(1 for record in records if record["is_correct"])
    accuracy = num_correct / total if total else 0.0

    if total:
        tp = num_correct
        fp = 0
        fn = total - num_correct
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    else:
        f1 = 0.0

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "total": total,
        "correct": num_correct,
        "skipped": skipped,
    }

    if save_path and records:
        output_path = Path(save_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            for record in records:
                json.dump(record, fout, ensure_ascii=False)
                fout.write("\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"records": records, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="InfoSeek evaluation with MemoryForest retrieval.")
    parser.add_argument("--dataset", default="../benchmark/infoseek/subset/infoseek_val_5k.jsonl")
    parser.add_argument("--images-root", default="../benchmark/oven")
    parser.add_argument("--image-index-csv", default="infoseek_image_index.csv", help="CSV mapping image_id to relative_path under images-root.")
    parser.add_argument("--save-path", default="infoseek_memtree_predictions_test_qwen2.5.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--collection", default="memtree")
    parser.add_argument("--clip-model", default="../ckpts/clip-vit-base-patch32")
    parser.add_argument("--vlm-model", default="../ckpts/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--text-model", default="../ckpts/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--answer-mode",
        choices=["vlm", "text"],
        default="vlm",
        help="Select answer model: vlm (QwenVL) or text (Qwen2.5-7B).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for transformers models.",
    )
    parser.add_argument("--root-top-k", type=int, default=3)
    parser.add_argument("--event-top-k", type=int, default=3)
    parser.add_argument("--leaf-top-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-trees", type=int, default=3)
    parser.add_argument("--max-sections", type=int, default=3)
    parser.add_argument("--max-leaves", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=32768)
    parser.add_argument("--retrieval-workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--tqdm-position", type=int, default=0)
    parser.add_argument(
        "--retrieval-mode",
        choices=["collapsed", "hierarchical"],
        default="collapsed",
        help="Select retrieval strategy: collapsed (leaf-only) or hierarchical (root->event->leaf).",
    )
    parser.add_argument(
        "--two-stage",
        type=int,
        choices=[0, 1],
        default=1,
        help="When retrieval-mode=collapsed, set 0 to disable stage-2 per-tree search (leaf_text_embedder ignored).",
    )
    parser.add_argument("--ingest-kb", action="store_true")
    parser.add_argument("--kb-path", default="../benchmark/infoseek/subset/wiki_text/wiki_5k_dict.json")
    parser.add_argument("--image-cache-dir", default="../benchmark/infoseek/subset/wiki_text/images_5k")
    parser.add_argument("--image-cache-index", default=None)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--ingest-batch-size", type=int, default=256)
    parser.add_argument("--text-workers", type=int, default=16)
    parser.add_argument("--image-workers", type=int, default=16)
    parser.add_argument("--leaf-text-model", default=None, help="Optional text embedding model for stage-2 leaf search.")
    parser.add_argument(
        "--leaf-text-device",
        default="cuda:auto",
        help="Device for leaf text embedding model (e.g., cuda:auto, auto, cuda:1, cpu).",
    )
    args = parser.parse_args()

    _disable_proxies()

    dataset = load_infoseek_dataset(args.dataset)

    cfg = Config(collection=args.collection)
    embedder = ClipEmbedding(model_name=args.clip_model, device_map="balanced", image_processor_name_or_path="../ckpts/clip-vit-large-patch14")
    # embedder = ClipEmbedding(model_name=args.clip_model, image_processor_name_or_path="../ckpts/clip-vit-large-patch14", device="cuda:auto")
    leaf_text_embedder: Optional[QwenTextEmbedding] = None
    leaf_text_vector_size: Optional[int] = None
    if args.leaf_text_model:
        leaf_text_embedder = QwenTextEmbedding(
            model_name=args.leaf_text_model,
            # device=args.leaf_text_device,
            model_kwargs={'device_map': "balanced"}
        )
        leaf_text_vector_size = leaf_text_embedder.dim
    memforest_store = MemoryForestStore(
        cfg,
        vector_size=embedder.dim,
        leaf_text_vector_size=leaf_text_vector_size,
    )

    if args.ingest_kb:
        kb_path = Path(args.kb_path).expanduser()
        image_cache_dir = Path(args.image_cache_dir).expanduser()
        image_cache_index = (
            Path(args.image_cache_index).expanduser()
            if args.image_cache_index
            else image_cache_dir / "image_cache_index.json"
        )
        maybe_ingest_kb(
            memforest_store,
            embedder,
            leaf_text_embedder,
            kb_path=kb_path,
            image_cache_dir=image_cache_dir,
            image_cache_index=image_cache_index,
            download_images=args.download_images,
            ingest_batch_size=args.ingest_batch_size,
            text_workers=args.text_workers,
            image_workers=args.image_workers,
            leaf_text_workers=args.text_workers,
            alpha=args.alpha,
            show_progress=True,
        )

    if args.answer_mode == "text":
        answer_model = Qwen(
            args.text_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        include_images_in_messages = False
    else:
        answer_model = QwenVL(
            model_path=args.vlm_model,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        include_images_in_messages = True

    results = run_infoseek_evaluation(
        answer_model,
        memforest_store,
        embedder,
        leaf_text_embedder,
        dataset,
        limit=args.limit,
        images_root=args.images_root,
        image_index_csv=args.image_index_csv,
        save_path=args.save_path,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        root_top_k=args.root_top_k,
        event_top_k=args.event_top_k,
        leaf_top_k=args.leaf_top_k,
        alpha=args.alpha,
        max_trees=args.max_trees,
        max_sections=args.max_sections,
        max_leaves=args.max_leaves,
        max_context_chars=args.max_context_chars,
        retrieval_workers=args.retrieval_workers,
        prefetch_batches=args.prefetch_batches,
        tqdm_position=args.tqdm_position,
        retrieval_mode=args.retrieval_mode,
        two_stage=bool(args.two_stage),
        include_images_in_messages=include_images_in_messages,
    )

    print(results["metrics"])


if __name__ == "__main__":
    main()
