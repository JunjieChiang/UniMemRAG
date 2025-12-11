#!/usr/bin/env python3
"""
Evaluate UniMemRAG on the MRAG-Bench dataset.

The script wires together the CLIP retriever backed by Qdrant and the Qwen2.5-VL
model to answer the multiple-choice questions in MRAG-Bench. It assumes the
image corpus has been ingested into Qdrant (run with --reindex to rebuild).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset  # type: ignore
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config
from unimemrag.embedding.models.ClipEmbedding import ClipEmbedding
from unimemrag.retriever.retriever import Retriever
from unimemrag.utils.image_node import ImageNode
from unimemrag.utils.log_config import setup_logger
from unimemrag.vector_store.qdrant import QdrantStore
from unimemrag.vlm.QwenVL import QwenVL

logger = setup_logger()
CHOICE_IDS: Sequence[str] = ("A", "B", "C", "D")


@dataclass
class ExampleResult:
    example_id: str
    aspect: str
    scenario: str
    question: str
    choice_text: Dict[str, str]
    gt_choice: str
    gt_answer: str
    model_answer_raw: str
    model_choice: Optional[str]
    is_correct: bool
    retrieved_uris: List[str]
    retrieved_scores: List[float]



def dedupe_nodes(nodes: Sequence[ImageNode]) -> List[ImageNode]:
    seen = set()
    deduped: List[ImageNode] = []
    for node in nodes:
        key = node.uri or node.id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(node)
    return deduped


def collect_retrieved_nodes(
    retriever: Retriever,
    *,
    text_query: str,
    query_image: Optional[object],
    use_image_query: bool,
    top_k: int,
) -> List[ImageNode]:
    nodes: List[ImageNode] = []
    text_nodes = retriever.search_by_text(
        text_query,
        target_modality="image",
        as_nodes=True,
    )
    nodes.extend(text_nodes)

    if use_image_query and query_image is not None:
        image_nodes = retriever.search_by_image(
            query_image,
            target_modality="image",
            as_nodes=True,
        )
        nodes.extend(image_nodes)

    deduped = dedupe_nodes(nodes)
    return deduped[:top_k]


def generate_answer_with_vlm(
    query: str,
    nodes: Sequence[ImageNode],
    *,
    vlm: QwenVL,
    base_dir: Path,
    question_image: Optional[object] = None,
    include_captions: bool = True,
    gen_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[str, List[ImageNode]]:
    gen_kwargs = gen_kwargs or {}

    loaded_images: List[object] = []
    caption_lines: List[str] = []

    if question_image is not None:
        loaded_images.append(question_image)
        if include_captions:
            caption_lines.append("Image 1: Question image provided with the example.")

    usable_nodes: List[ImageNode] = []
    for node in nodes:
        if (node.modality or "").lower() != "image":
            continue
        try:
            loaded = node.load_image(base_dir=base_dir)
        except Exception:
            continue
        loaded_images.append(loaded)
        usable_nodes.append(node)

    if not nodes and question_image is None and not loaded_images:
        prompt = (
            "No related images were retrieved, but please answer the question.\n"
            f"Question: {query}"
        )
        answer = vlm.complete(prompt, images=None, **gen_kwargs)
        return answer, []

    if question_image is None and not usable_nodes and not loaded_images:
        captions = " | ".join(filter(None, (n.content for n in nodes))) or "N/A"
        prompt = (
            "Images were retrieved but could not be loaded. Use the captions below if helpful.\n"
            f"Captions: {captions}\n"
            f"Question: {query}"
        )
        answer = vlm.complete(prompt, images=None, **gen_kwargs)
        return answer, list(nodes)

    if include_captions:
        start_idx = 1 if question_image is None else 2
        for idx, node in enumerate(usable_nodes, start_idx):
            caption = node.content or Path(node.uri).stem
            caption_lines.append(f"Image {idx}: {caption}")

    prompt_parts = [
        "You are a helpful assistant. Use the provided images to answer the user's question.",
    ]
    if caption_lines:
        prompt_parts.append("Captions for the images:\n" + "\n".join(caption_lines))
    prompt_parts.append(f"Question: {query}")
    prompt = "\n\n".join(prompt_parts)

    images_arg = loaded_images if loaded_images else None
    answer = vlm.complete(prompt, images=images_arg, **gen_kwargs)
    return answer, usable_nodes


def remove_proxies() -> None:
    """Ensure local Qdrant access is not impacted by proxy settings."""
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    existing = os.environ.get("NO_PROXY", "")
    tokens = {token.strip() for token in existing.split(",") if token.strip()}
    tokens.update({"localhost", "127.0.0.1"})
    os.environ["NO_PROXY"] = ",".join(sorted(tokens))


def iter_image_files(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def ensure_corpus_indexed(
    store: QdrantStore,
    embedder: ClipEmbedding,
    corpus_root: Path,
    *,
    reindex: bool = False,
) -> None:
    """Ingest MRAG-Bench images into Qdrant if needed."""
    if reindex:
        store.clear_collection()

    try:
        count = store.client.count(collection_name=store.cfg.collection, exact=True)
        existing = int(getattr(count, "count", 0))
    except Exception:
        existing = 0

    if existing > 0 and not reindex:
        return

    image_paths = sorted(iter_image_files(corpus_root / "image_corpus"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {corpus_root / 'image_corpus'}")

    store.ingest_images(
        embedder,
        image_paths,
        caption_builder=store.default_caption_from_name,
        show_progress=True,
    )


def build_prompt(example: Dict[str, object], *, include_meta: bool = True) -> str:
    """Format the question and answer choices into a prompt for the VLM."""
    lines: List[str] = []
    # print(example)
    if include_meta:
        aspect = example.get("aspect")
        scenario = example.get("scenario")
        if aspect:
            lines.append(f"Aspect: {aspect}")
        if scenario:
            lines.append(f"Scenario: {scenario}")
    lines.append(f"Question: {example['question']}")
    lines.append("Choices:")
    for choice_id in CHOICE_IDS:
        choice_text = example.get(choice_id)
        if choice_text:
            lines.append(f"{choice_id}. {choice_text}")
    lines.append(
        "Answer by writing only the letter (A, B, C, or D) that best answers the question."
    )
    return "\n".join(lines)


CHOICE_PATTERN = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def pick_choice(raw_answer: str, options: Dict[str, str]) -> Optional[str]:
    """Convert the raw model output into a choice identifier."""
    if not raw_answer:
        return None

    upper = raw_answer.upper()
    # Direct letter extraction (e.g., "Answer: B")
    match = CHOICE_PATTERN.search(upper)
    if match:
        return match.group(1).upper()

    # Pattern like "Option C" or "Choice A"
    for token in CHOICE_IDS:
        if f"OPTION {token}" in upper or f"CHOICE {token}" in upper:
            return token

    # Fall back to matching full choice text.
    lower_answer = raw_answer.lower()
    for token, text in options.items():
        if text and text.lower() in lower_answer:
            return token

    return None


def evaluate(
    retriever: Retriever,
    vlm: QwenVL,
    dataset_split: Sequence[Dict[str, object]],
    *,
    base_dir: Path,
    top_k: int,
    use_image_query: bool,
    gen_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[List[ExampleResult], float]:
    results: List[ExampleResult] = []
    correct = 0
    if gen_kwargs is None:
        gen_kwargs = {"max_new_tokens": 512}

    for example in tqdm(dataset_split, desc="Evaluating", leave=False):
        query_text = build_prompt(example)
        nodes = collect_retrieved_nodes(
            retriever,
            text_query=query_text,
            query_image=example.get("image"),
            use_image_query=use_image_query,
            top_k=top_k,
        )
        answer, _ = generate_answer_with_vlm(
            query_text,
            nodes,
            vlm=vlm,
            base_dir=base_dir,
            question_image=example.get("image"),
            gen_kwargs=gen_kwargs,
        )

        options = {token: str(example.get(token, "")) for token in CHOICE_IDS}
        predicted_choice = pick_choice(answer, options)
        gold_choice = str(example.get("answer_choice"))
        gold_answer = str(example.get("answer"))
        is_correct = predicted_choice == gold_choice
        if is_correct:
            correct += 1

        result = ExampleResult(
            example_id=str(example.get("id")),
            aspect=str(example.get("aspect")),
            scenario=str(example.get("scenario")),
            question=str(example.get("question")),
            choice_text=options,
            gt_choice=gold_choice,
            gt_answer=gold_answer,
            model_answer_raw=answer,
            model_choice=predicted_choice,
            is_correct=is_correct,
            retrieved_uris=[node.uri for node in nodes],
            retrieved_scores=[float(node.score) for node in nodes],
        )
        results.append(result)

    # dump results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "mrag_bench_results_qwen2.5-7b.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for item in results:
            json.dump(asdict(item), fout, ensure_ascii=False)
            fout.write("\n")

    # report metrics
    accuracy = correct / max(len(results), 1)
    logger.info("Accuracy: ", accuracy)

    return results, accuracy

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UniMemRAG on MRAG-Bench.")
    parser.add_argument(
        "--dataset-split",
        default="test",
        help="Which split to load from uclanlp/MRAG-Bench (default: test).",
    )
    parser.add_argument(
        "--collection",
        default="mmrag_bench",
        help="Name of the Qdrant collection to use for the corpus.",
    )
    parser.add_argument(
        "--model-name",
        default="../ckpts/clip-vit-base-patch32",
        help="CLIP checkpoint to use for embedding.",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant host address.",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6336,
        help="Qdrant HTTP port.",
    )
    parser.add_argument(
        "--qdrant-grpc-port",
        type=int,
        default=6337,
        help="Qdrant gRPC port.",
    )
    parser.add_argument(
        "--prefer-grpc",
        action="store_true",
        help="Use gRPC instead of HTTP for Qdrant queries.",
    )
    parser.add_argument(
        "--corpus-root",
        default="benchmark/mrag-bench",
        help="Root directory of the MRAG-Bench image corpus.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved images to pass to the VLM.",
    )
    parser.add_argument(
        "--use-image-query",
        action="store_true",
        help="Retrieve with both the question text and the instance image.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-ingestion of the image corpus into Qdrant.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Evaluate on at most this many examples (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional path to save per-example results as JSONL.",
    )
    parser.add_argument(
        "--vlm-path",
        default="../ckpts/Qwen2.5-VL-7B-Instruct",
        help="Local path or model name for Qwen2.5-VL.",
    )
    return parser.parse_args()


def main() -> None:
    remove_proxies()
    args = parse_args()

    cfg = Config(
        model_name=args.model_name,
        collection=args.collection,
        prefer_grpc=args.prefer_grpc,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_grpc_port=args.qdrant_grpc_port,
        top_k=args.top_k,
    )

    embedder = ClipEmbedding(model_name=cfg.model_name)
    store = QdrantStore(cfg, embedder.dim)
    corpus_root = Path(args.corpus_root)
    ensure_corpus_indexed(store, embedder, corpus_root, reindex=args.reindex)

    retriever = Retriever(embedder, store, top_k=cfg.top_k)
    vlm = QwenVL(
        model_path=args.vlm_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    dataset = load_dataset("uclanlp/MRAG-Bench", split=args.dataset_split)
    if args.max_examples is not None:
        dataset = dataset.select(range(args.max_examples))

    base_dir = corpus_root
    results, accuracy = evaluate(
        retriever,
        vlm,
        dataset,
        base_dir=base_dir,
        top_k=args.top_k,
        use_image_query=args.use_image_query,
    )

    print(f"Accuracy: {accuracy * 100:.2f}% ({sum(r.is_correct for r in results)}/{len(results)})")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            for item in results:
                json.dump(asdict(item), fout, ensure_ascii=False)
                fout.write("\n")


if __name__ == "__main__":
    main()

