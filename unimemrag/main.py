from __future__ import annotations

import math
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from config import Config as QdrantConfig
#from examples.MMLongBench.data import TestItemDataset, default_post_process
from unimemrag.embedding.models.ClipEmbedding import ClipEmbedding
from unimemrag.retriever.retriever import Retriever
from unimemrag.utils.log_config import setup_logger
from unimemrag.vector_store.qdrant import QdrantStore

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


logger = setup_logger(__name__)


def _resolve_path(path_like: Any) -> Path:
    path = Path(path_like).expanduser()
    if path.exists():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    candidate = (repo_root / path).expanduser()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'Unable to resolve path: {path_like}')


@dataclass
class RagPipelineConfig:
    """
    Configuration helper for notebook-driven RAG experiments on MMLongBench.

    Args:
        dataset_file: Path to the JSONL dataset file (e.g. viquae_K16_dep6.jsonl).
        image_root: Root directory that contains the dataset images (mmlb_image).
        collection_name: Qdrant collection name used for the temporary index.
        clip_model_name: CLIP model name for embedding (text + image).
        top_k: Default number of documents retrieved per query.
        text_batch_size: Batch size for text embedding during indexing.
        use_in_memory_qdrant: If True, spin up an embedded Qdrant instance (":memory:").
        qdrant_path: Filesystem path for embedded Qdrant (":memory:" keeps everything in RAM).
        qdrant_host / qdrant_port / qdrant_grpc_port / prefer_grpc: Remote Qdrant endpoints when
            use_in_memory_qdrant is False.
    """

    dataset_file: str
    image_root: str
    collection: str = "mmlb_demo"
    model_name: str = "openai/clip-vit-base-patch32"
    top_k: Optional[int] = 3
    image_top_k: Optional[int] = 5
    text_batch_size: int = 32
    use_in_memory_qdrant: bool = True
    qdrant_path: str = ":memory:"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6336
    qdrant_grpc_port: int = 6337
    prefer_grpc: bool = False
    timeout_sec: int = 60


class NotebookQdrantStore(QdrantStore):
    """
    Thin wrapper around QdrantStore that allows spinning up an embedded instance.

    This avoids the need for an external Qdrant service when prototyping inside a notebook.
    """

    def __init__(self, cfg: QdrantConfig, vector_size: int, *, path: str = ":memory:"):
        self.cfg = cfg
        self.vector_name = "clip"
        from qdrant_client import QdrantClient

        if path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(
                host=cfg.qdrant_host,
                port=cfg.qdrant_port,
                grpc_port=cfg.qdrant_grpc_port,
                prefer_grpc=cfg.prefer_grpc,
                timeout=cfg.timeout_sec,
            )
        self._ensure_collection(vector_size)


def create_retriever(config: RagPipelineConfig) -> Tuple[Retriever, ClipEmbedding, QdrantStore]:
    """
    Instantiate the CLIP embedder, Qdrant vector store and Retriever using the provided config.

    Returns:
        (retriever, embedder, store)
    """

    embedder = ClipEmbedding(config.model_name)
    qcfg = QdrantConfig(
        qdrant_host=config.qdrant_host,
        qdrant_port=config.qdrant_port,
        qdrant_grpc_port=config.qdrant_grpc_port,
        prefer_grpc=config.prefer_grpc,
        collection=config.collection,
    )
    if config.use_in_memory_qdrant:
        store = NotebookQdrantStore(qcfg, vector_size=embedder.dim, path=config.qdrant_path)
    else:
        store = QdrantStore(qcfg, vector_size=embedder.dim)

    retriever = Retriever(embedder, store, top_k=config.top_k)
    return retriever, embedder, store


def load_viquae_dataset(dataset_file: str, max_samples: Optional[int] = None) -> Dataset:
    """
    Load the viquae split from disk as a Hugging Face dataset.

    Args:
        dataset_file: JSONL file path.
        max_samples: Optional cap on the number of samples returned.
    """

    dataset_path = _resolve_path(dataset_file)
    dataset = load_dataset("json", data_files=str(dataset_path))["train"]
    if max_samples is not None:
        max_count = min(len(dataset), max_samples)
        dataset = dataset.select(range(max_count))
    return dataset

def _iter_unique_contexts(dataset: Dataset) -> Iterable[Dict[str, str]]:
    seen_ids: Dict[str, bool] = {}
    for row in dataset:
        for ctx in row.get("ctxs", []):
            doc_id = str(ctx.get("doc_id"))
            if not doc_id:
                continue
            if doc_id in seen_ids:
                continue
            seen_ids[doc_id] = True
            yield {
                "doc_id": doc_id,
                "title": ctx.get("title") or "",
                "text": ctx.get("text") or "",
                "image": ctx.get("image"),
                "image_url": ctx.get("image_url"),
            }


def index_text_corpus(
    dataset: Dataset,
    *,
    embedder: ClipEmbedding,
    store: QdrantStore,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Embed and upsert all textual contexts from the dataset into the Qdrant store.
    """

    contexts = list(_iter_unique_contexts(dataset))
    if not contexts:
        return {"texts_indexed": 0, "batches": 0}

    batches = int(math.ceil(len(contexts) / batch_size))
    iterator = range(batches)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Indexing texts", leave=False)

    indexed = 0
    for idx in iterator:
        chunk = contexts[idx * batch_size : (idx + 1) * batch_size]
        texts = []
        payloads = []
        ids = []
        for item in chunk:
            title = item["title"]
            text = item["text"]
            content = f"{title}\n\n{text}" if title else text
            texts.append(content)
            payload = {
                "uri": item["doc_id"],
                "modality": "text",
                "title": title,
                "content": text,
            }
            if item.get("image"):
                payload["image"] = item["image"]
            if item.get("image_url"):
                payload["image_url"] = item["image_url"]
            payloads.append(payload)
            ids.append(item["doc_id"])
        vectors = embedder.embed_texts(texts)
        store.upsert(vectors, payloads, ids=ids)
        indexed += len(chunk)

    logger.info("Indexed %s text passages into Qdrant", indexed)
    return {"texts_indexed": indexed, "batches": batches}


def index_image_corpus(
    dataset: Dataset,
    *,
    embedder: ClipEmbedding,
    store: QdrantStore,
    image_root: Union[str, Path],
) -> Dict[str, Any]:
    """
    Optionally embed all images referenced in the dataset.

    Images are stored with payload modality='image' so they can be retrieved later if needed.
    """

    if isinstance(image_root, Path):
        base_root = image_root if image_root.exists() else _resolve_path(image_root)
    else:
        base_root = _resolve_path(image_root)
    images = {Path(row["image"]) for row in dataset if row.get("image")}
    if not images:
        return {"images_indexed": 0}

    def payload_builder(path_obj: Path) -> Dict[str, Any]:
        absolute = (base_root / path_obj).resolve()
        return {
            "uri": absolute.as_posix(),
            "modality": "image",
            "content": path_obj.stem,
        }

    stats = store.ingest_images(
        embedder=embedder,
        image_items=[(base_root / img).resolve() for img in images],
        payload_builder=payload_builder,
        caption_builder=None,
        show_progress=True,
    )
    logger.info(
        "Indexed %s images (failures=%s)",
        stats.get("images_indexed", 0),
        len(stats.get("failures", [])),
    )
    return stats


def build_rag_bundle(
    dataset: Dataset,
    *,
    retriever: Retriever,
    image_root: Union[str, Path],
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convert the raw dataset into the evaluation bundle consumed by TestItemDataset.
    Retrieval is performed per-question using the provided Retriever.
    """

    user_template = (
        "Use the given documents to write a concise and short answer to the question about the "
        "entity shown in the image. Write your answer in the following format:\n"
        "Answer: [answer]\n\n{context}\n\nQuestion: {question}"
    )
    system_template = "Answer:"
    passage_template = "Document (Title: {title}): {text}"
    bundle_rows: List[Dict[str, Any]] = []

    if isinstance(image_root, Path):
        base_root = image_root if image_root.exists() else _resolve_path(image_root)
    else:
        base_root = _resolve_path(image_root)

    iterator: Iterable[Any] = dataset
    if tqdm is not None:
        iterator = tqdm(dataset, desc="Building RAG bundle", leave=False)

    for row in iterator:
        question = row.get("original_question") or row["question"]
        image_rel = row.get("image")
        has_image = bool(image_rel)

        if has_image:
            image_abs = (base_root / Path(image_rel)).resolve()
            nodes = retriever.search_by_image(
                image=image_abs.as_posix(),
                target_modality="image",
                as_nodes=True,
            )
            if not nodes:
                nodes = retriever.search_by_text(
                    question,
                    target_modality="text",
                    as_nodes=True,
                )
        else:
            nodes = retriever.search_by_text(
                question,
                target_modality="text",
                as_nodes=True,
            )
        k = top_k or retriever.top_k
        nodes = nodes[:k]

        documents: List[str] = []
        retrieved_doc_ids: List[str] = []
        retrieved_scores: List[float] = []
        retrieved_titles: List[str] = []
        for node in nodes:
            payload = getattr(node, "payload", {}) or {}
            title = payload.get("title") or Path(node.uri).stem
            content_text = getattr(node, "content", None) or payload.get("content") or title
            documents.append(passage_template.format(title=title, text=content_text))
            retrieved_doc_ids.append(node.uri)
            retrieved_scores.append(float(node.score))
            retrieved_titles.append(title)

        context_block = "\n\n".join(documents)
        if has_image:
            question_with_media = "<image>" + row["question"]
            image_list = [image_abs.as_posix()]
        else:
            question_with_media = question
            image_list = []
        bundle_rows.append(
            {
                "id": row["id"],
                "context": context_block,
                "question": question_with_media,
                "answer": row["answer"],
                "image_list": image_list,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_scores": retrieved_scores,
                "retrieved_titles": retrieved_titles,
                "original_image": row.get("image"),
                "original_question": row.get("original_question"),
            }
        )

    rag_dataset = Dataset.from_list(bundle_rows)
    bundle = {
        "data": rag_dataset,
        "prompt_template": user_template + "\n" + system_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="sub_em", prefix=system_template),
    }
    return bundle


def evaluate_bundle_with_model(
    model,
    bundle: Dict[str, Any],
    *,
    num_workers: int = 0,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the provided VLM on the bundle and compute metrics identical to eval.py.
    By default results are written under `bundle["output_dir"]`, but this can be disabled via `write_results`.
    """

    hf_dataset: Dataset = bundle["data"]
    if max_examples is not None:
        hf_dataset = hf_dataset.select(range(min(len(hf_dataset), max_examples)))

    dataset_wrapper = {**bundle, "data": hf_dataset}
    test_dataset = TestItemDataset(dataset_wrapper, model, model.processor)
    dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=num_workers,
    )

    metrics: Dict[str, List[float]] = {}
    results: List[Dict[str, Any]] = []

    write_results = bool(bundle.get("write_results", True))
    result_filename = bundle.get("result_filename", "rag_inference.json")
    score_filename = bundle.get("score_filename", "rag_inference.score")
    output_dir = Path(bundle.get("output_dir", "examples/results/demo"))
    if write_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / result_filename
        score_path = output_dir / score_filename
    else:
        result_path = None
        score_path = None

    iterator = enumerate(dataloader)
    if tqdm is not None:
        iterator = enumerate(tqdm(dataloader, desc="Evaluating", leave=False))

    for idx, batch in iterator:
        if max_examples is not None and idx >= max_examples:
            break

        (inputs, _) = batch[0]
        test_item = hf_dataset[idx]
        output = model.generate(inputs=inputs)

        mets, extras = bundle["post_process"](output, test_item)
        output.update({**extras, **mets})
        for key, value in mets.items():
            metrics.setdefault(key, []).append(value)

        output["input_len"] = inputs.input_ids.shape[1]
        results.append({**test_item, **output})

    averaged_metrics = {k: float(np.mean(v)) * (100 if "_len" not in k else 1) for k, v in metrics.items()}
    output_payload = {
        "results": results,
        "metrics": metrics,
        "averaged_metrics": averaged_metrics,
    }

    if write_results and result_path is not None and score_path is not None:
        with result_path.open('w', encoding='utf-8') as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        with score_path.open('w', encoding='utf-8') as f:
            json.dump(averaged_metrics, f, ensure_ascii=False, indent=2)
        logger.info("Saved RAG outputs to %s and %s", result_path.as_posix(), score_path.as_posix())

    return output_payload


__all__ = [
    "RagPipelineConfig",
    "NotebookQdrantStore",
    "create_retriever",
    "load_viquae_dataset",
    "index_text_corpus",
    "index_image_corpus",
    "build_rag_bundle",
    "evaluate_bundle_with_model",
]
