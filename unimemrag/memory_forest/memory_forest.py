from __future__ import annotations

import csv
import hashlib
import logging
import os
import uuid
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import numpy as np
from qdrant_client.http import models as qmodels

from unimemrag.embedding.models.ClipEmbedding import ClipEmbedding
from unimemrag.vector_store.qdrant import QdrantStore


logger = logging.getLogger(__name__)

_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg"}
_IMAGE_INDEX_CACHE: Dict[Tuple[Path, Path], Dict[str, Path]] = {}
_REPO_ROOT = Path(__file__).resolve().parents[2]

INFOSEEK_IMAGES_ROOT = "benchmark/oven"
INFOSEEK_IMAGE_INDEX_CSV = "infoseek_image_index.csv"


def _resolve_infoseek_path(path_value: Union[str, Path]) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (_REPO_ROOT / path).resolve()


def _get_infoseek_images_root() -> Optional[Path]:
    images_root = INFOSEEK_IMAGES_ROOT
    if not images_root:
        return None
    return _resolve_infoseek_path(images_root)


def _get_infoseek_image_index_csv() -> Optional[Path]:
    csv_path = INFOSEEK_IMAGE_INDEX_CSV
    if not csv_path:
        return None
    return _resolve_infoseek_path(csv_path)


def _load_image_index(images_root: Path, image_index_csv: Path) -> Dict[str, Path]:
    cache_key = (images_root, image_index_csv)
    cached = _IMAGE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mapping: Dict[str, Path] = {}
    with image_index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image_id" not in reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise ValueError(
                f"Image index CSV must contain 'image_id' and 'relative_path' columns: {image_index_csv}"
            )
        for row in reader:
            image_id = Path((row.get("image_id") or "").strip()).stem
            rel_path_str = (row.get("relative_path") or "").strip()
            if not image_id or not rel_path_str:
                continue
            rel_path = Path(rel_path_str)
            if rel_path.suffix.lower() not in _ALLOWED_IMAGE_EXTS:
                continue
            mapping[image_id] = rel_path

    _IMAGE_INDEX_CACHE[cache_key] = mapping
    return mapping


def _resolve_query_image_path(query_image: Union[str, Path, Any]) -> Union[str, Any]:
    if not isinstance(query_image, (str, Path)):
        return query_image
    if isinstance(query_image, str) and query_image.startswith(("http://", "https://")):
        return query_image
    path = Path(query_image)
    if path.is_absolute():
        if path.exists():
            return path.as_posix()
        raise FileNotFoundError(f"Query image not found: {path}")
    if path.exists():
        return path.resolve().as_posix()

    images_root = _get_infoseek_images_root()
    image_index_csv = _get_infoseek_image_index_csv()
    if not image_index_csv:
        raise FileNotFoundError("Image index CSV is required but not set in INFOSEEK_IMAGE_INDEX_CSV/IMAGE_INDEX_CSV.")
    if not image_index_csv.exists():
        raise FileNotFoundError(f"Image index CSV not found: {image_index_csv}")
    if images_root is None:
        raise FileNotFoundError("Images root is required but not set in INFOSEEK_IMAGES_ROOT/IMAGES_ROOT.")

    index = _load_image_index(images_root, image_index_csv)
    rel_path = index.get(path.stem)
    if not rel_path:
        raise FileNotFoundError(f"Image id not found in CSV index: {path.stem!r}")
    abs_path = rel_path if rel_path.is_absolute() else (images_root / rel_path)
    abs_path = abs_path.resolve()
    if abs_path.exists():
        return abs_path.as_posix()
    raise FileNotFoundError(f"Indexed image path not found: {abs_path}")


class NodeRole(str, Enum):
    ROOT = "root"
    EVENT = "event"
    LEAF = "leaf"


@dataclass(frozen=True)
class LeafNode:
    text: str
    parent_id: str
    leaf_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    text_hash: Optional[str] = None


@dataclass(frozen=True)
class EventNode:
    summary: str
    parent_id: Optional[str] = None
    event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    leaf_ids: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class RootNode:
    topic: str
    root_id: Optional[str] = None
    image_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_candidates: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class MemoryTree:
    tree_id: str
    root: RootNode
    events: Sequence[EventNode] = field(default_factory=tuple)
    leaves: Sequence[LeafNode] = field(default_factory=tuple)


@dataclass(frozen=True)
class RetrievalHit:
    id: str
    score: float
    payload: Dict[str, Any]
    role: NodeRole
    collection: str


@dataclass(frozen=True)
class TreeRetrievalResult:
    tree_id: str
    root: RetrievalHit
    events: List[RetrievalHit]
    leaves: Dict[str, List[RetrievalHit]]


@dataclass(frozen=True)
class CollapsedRetrievalResult:
    tree_id: str
    events: List[RetrievalHit]
    leaves: Dict[str, List[RetrievalHit]]


class MemoryForestStore(QdrantStore):
    """
    Hierarchical memory structure built on top of QdrantStore.

    The base collection (cfg.collection) holds fused multimodal root nodes.
    Event and leaf nodes live in dedicated collections that reference their parents
    via payload metadata enabling tree-aware retrieval.
    """

    def __init__(
        self,
        cfg,
        vector_size: int,
        *,
        event_collection: Optional[str] = None,
        leaf_collection: Optional[str] = None,
        fusion_alpha: float = 0.4,
    ) -> None:
        super().__init__(cfg, vector_size)
        self.event_collection = event_collection or f"{cfg.collection}_events"
        self.leaf_collection = leaf_collection or f"{cfg.collection}_leaves"
        self.fusion_alpha = fusion_alpha

        self._ensure_named_collection(self.event_collection, vector_size)
        self._ensure_named_collection(self.leaf_collection, vector_size)
        self._ensure_payload_indexes_for_collection(
            self.event_collection, ["modality", "tree_id", "node_type", "parent_id"]
        )
        self._ensure_payload_indexes_for_collection(
            self.leaf_collection, ["modality", "tree_id", "node_type", "parent_id"]
        )

    # ------------------------------------------------------------------ creation utils
    def _ensure_named_collection(self, collection_name: str, vector_size: int) -> None:
        collections = {c.name for c in self.client.get_collections().collections}
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=self.cfg.distance,
                    on_disk=self.cfg.on_disk,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=20000),
            )

    def _ensure_payload_indexes_for_collection(self, collection_name: str, fields: Iterable[str]) -> None:
        for field in fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_type="keyword",
                )
            except Exception:
                continue

    # ------------------------------------------------------------------ ingestion
    def ingest_trees(
        self,
        trees: Sequence[MemoryTree],
        embedder: ClipEmbedding,
        *,
        alpha: Optional[float] = None,
        fusion_fn: Optional[
            Callable[[np.ndarray, Optional[np.ndarray], MemoryTree], np.ndarray]
        ] = None,
        batch_size: int = 32,
        show_progress: bool = False,
        text_workers: int = 1,
        image_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Index a batch of MemoryTree instances into Qdrant.

        Args:
            trees: Iterable of MemoryTree describing the hierarchy.
            embedder: Multimodal encoder used for all embeddings.
            alpha: Optional override for default fusion weight.
            fusion_fn: Custom fusion callable if more advanced blending is required.
            batch_size: Embedding batch size for text nodes.
            show_progress: If True, display a tqdm bar tracking embedding throughput.
            text_workers: Number of worker threads for text embeddings.
            image_workers: Number of worker threads for image embeddings.
        """
        if not trees:
            return {"roots": 0, "events": 0, "leaves": 0}

        processed = [self._normalize_tree(tree) for tree in trees]
        leaf_entries = self._collect_leaf_entries(processed)
        event_entries = self._collect_event_entries(processed, leaf_entries)
        root_entries = self._collect_root_entries(processed, event_entries)

        embed_progress = None
        if show_progress and tqdm is not None:
            total_embeddings = (
                len(leaf_entries)
                + len(event_entries)
                + len(root_entries) * 2
                + sum(len(entry.get("image_candidates") or []) for entry in root_entries)
            )
            if total_embeddings > 0:
                embed_progress = tqdm(total=total_embeddings, desc="Embedding Trees", leave=False)

        leaf_vectors = self._embed_texts(
            [entry["text"] for entry in leaf_entries],
            embedder,
            batch_size,
            progress_bar=embed_progress,
            num_workers=text_workers,
        )
        event_vectors = self._embed_texts(
            [entry["summary"] for entry in event_entries],
            embedder,
            batch_size,
            progress_bar=embed_progress,
            num_workers=text_workers,
        )
        root_text_vectors = self._embed_texts(
            [entry["topic"] for entry in root_entries],
            embedder,
            batch_size,
            progress_bar=embed_progress,
            num_workers=text_workers,
        )
        alignment_texts = [self._build_root_alignment_text(entry["tree"]) for entry in root_entries]
        alignment_vectors = self._embed_texts(
            alignment_texts,
            embedder,
            batch_size,
            progress_bar=embed_progress,
            num_workers=text_workers,
        )
        root_image_vectors = self._select_root_image_vectors(
            root_entries,
            embedder,
            alignment_vectors,
            image_batch_size=batch_size,
            image_workers=image_workers,
            progress_bar=embed_progress,
        )

        fused_root_vectors = []
        for idx, entry in enumerate(root_entries):
            text_vec = root_text_vectors[idx] if len(root_text_vectors) > idx else None
            image_vec = root_image_vectors.get(idx)
            fused_vec = self._compose_root_vector(      # generate fused vector for root node
                tree=entry["tree"],
                text_vec=text_vec,
                image_vec=image_vec,
                alpha_override=alpha,
                fusion_fn=fusion_fn,
            )
            fused_root_vectors.append(fused_vec)
        fused_root_array = np.vstack(fused_root_vectors).astype(np.float32)

        self._upsert_roots(fused_root_array, root_entries)
        self._upsert_events(event_vectors, event_entries)
        self._upsert_leaves(leaf_vectors, leaf_entries)
        if embed_progress is not None:
            embed_progress.close()

        return {
            "roots": len(root_entries),
            "events": len(event_entries),
            "leaves": len(leaf_entries),
        }

    def ingest_trees_new(
        self,
        trees: Sequence[MemoryTree],
        embedder: ClipEmbedding,
        *,
        beta: Optional[float] = None,
        fusion_fn: Optional[
            Callable[[np.ndarray, Optional[np.ndarray], MemoryTree], np.ndarray]
        ] = None,
        leaf_fusion_fn: Optional[
            Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
        ] = None,
        batch_size: int = 32,
        show_progress: bool = False,
        text_workers: int = 1,
        image_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Index a batch of MemoryTree instances with multimodal leaf fusion.

        - Select an event image using event summary vs section_images (or fallback to root image).
        - Fuse leaf text with the selected event image to build unified leaf vectors.
        """
        if not trees:
            return {"roots": 0, "events": 0, "leaves": 0}

        processed = [self._normalize_tree(tree) for tree in trees]
        leaf_entries = self._collect_leaf_entries(processed)
        event_entries = self._collect_event_entries(processed, leaf_entries)
        root_entries = self._collect_root_entries(processed, event_entries)

        embed_progress = None
        if show_progress and tqdm is not None:
            event_image_candidates = 0
            for entry in event_entries:
                candidates = (entry.get("metadata") or {}).get("section_images") or []
                if isinstance(candidates, str):
                    event_image_candidates += 1
                else:
                    event_image_candidates += len(candidates)
            total_embeddings = (
                len(leaf_entries)
                + len(event_entries)
                + len(root_entries) * 2
                + sum(len(entry.get("image_candidates") or []) for entry in root_entries)
                + event_image_candidates
            )
            if total_embeddings > 0:
                embed_progress = tqdm(
                    total=total_embeddings,
                    desc="Embedding Trees (fused leaves)",
                    leave=False,
                )

        leaf_text_vectors = self._embed_texts(
            [entry["text"] for entry in leaf_entries],
            embedder,
            batch_size,
            progress_bar=embed_progress,
            num_workers=text_workers,
        )
        event_vectors = self._embed_texts(
            [entry["summary"] for entry in event_entries],
            embedder,
            batch_size,
            progress_bar=embed_progress,
            num_workers=text_workers,
        )
        root_text_vectors = self._embed_texts(
            [entry["topic"] for entry in root_entries],
            embedder,
            batch_size,
            progress_bar=embed_progress,
            num_workers=text_workers,
        )
        alignment_texts = [self._build_root_alignment_text(entry["tree"]) for entry in root_entries]
        alignment_vectors = self._embed_texts(
            alignment_texts,
            embedder,
            batch_size,
            progress_bar=embed_progress,
            num_workers=text_workers,
        )
        root_image_vectors = self._select_root_image_vectors(
            root_entries,
            embedder,
            alignment_vectors,
            image_batch_size=batch_size,
            image_workers=image_workers,
            progress_bar=embed_progress,
        )

        root_image_vectors_by_tree: Dict[str, np.ndarray] = {}
        root_image_uris_by_tree: Dict[str, Optional[str]] = {}
        for idx, entry in enumerate(root_entries):
            tree_id = str(entry["tree_id"])
            root_image_uris_by_tree[tree_id] = entry.get("image_uri")
            vec = root_image_vectors.get(idx)
            if vec is not None:
                root_image_vectors_by_tree[tree_id] = vec

        event_image_vectors, event_image_uris = self._select_event_image_vectors(
            event_entries,
            embedder,
            event_vectors,
            root_image_vectors_by_tree,
            root_image_uris_by_tree,
            image_batch_size=batch_size,
            image_workers=image_workers,
            progress_bar=embed_progress,
        )

        fused_leaf_vectors: List[np.ndarray] = []
        for idx, entry in enumerate(leaf_entries):
            text_vec = leaf_text_vectors[idx] if idx < len(leaf_text_vectors) else None
            event_id = entry.get("parent_id")
            tree_id = str(entry.get("tree_id"))
            image_vec = event_image_vectors.get(str(event_id))
            image_uri = event_image_uris.get(str(event_id))
            image_source = "event" if image_uri else None
            if image_vec is None:
                image_vec = root_image_vectors_by_tree.get(tree_id)
                if image_uri is None:
                    image_uri = root_image_uris_by_tree.get(tree_id)
                    image_source = "root" if image_uri else None
            fused_vec = self._build_query_vector(text_vec, image_vec, beta, leaf_fusion_fn)
            fused_leaf_vectors.append(fused_vec)
            entry["image_uri"] = image_uri
            entry["image_source"] = image_source

        fused_leaf_array = (
            np.vstack(fused_leaf_vectors).astype(np.float32)
            if fused_leaf_vectors
            else np.empty((0, embedder.dim), dtype=np.float32)
        )

        fused_root_vectors = []
        for idx, entry in enumerate(root_entries):
            text_vec = root_text_vectors[idx] if len(root_text_vectors) > idx else None
            image_vec = root_image_vectors.get(idx)
            fused_vec = self._compose_root_vector(
                tree=entry["tree"],
                text_vec=text_vec,
                image_vec=image_vec,
                alpha_override=beta,
                fusion_fn=fusion_fn,
            )
            fused_root_vectors.append(fused_vec)
        fused_root_array = np.vstack(fused_root_vectors).astype(np.float32)

        self._upsert_roots(fused_root_array, root_entries)
        self._upsert_events(event_vectors, event_entries)
        self._upsert_leaves_fused(fused_leaf_array, leaf_entries)
        if embed_progress is not None:
            embed_progress.close()

        return {
            "roots": len(root_entries),
            "events": len(event_entries),
            "leaves": len(leaf_entries),
        }

    def _normalize_tree(self, tree: MemoryTree) -> Dict[str, Any]:
        root_raw_id = tree.root.root_id or tree.tree_id
        normalized_root_id = self._normalize_id(root_raw_id)
        id_lookup: Dict[str, str] = {root_raw_id: normalized_root_id}
        event_items: List[Dict[str, Any]] = []
        event_id_lookup: Dict[str, str] = {}
        for event in tree.events:
            event_id = event.event_id or str(uuid.uuid4())
            normalized_event_id = self._normalize_id(event_id)
            event_id_lookup[event_id] = normalized_event_id
            parent_key = event.parent_id or root_raw_id
            parent_id = id_lookup.get(parent_key)
            if parent_id is None:
                parent_id = self._normalize_id(parent_key)
                id_lookup[parent_key] = parent_id
            event_items.append(
                {
                    "event_id": normalized_event_id,
                    "parent_id": parent_id,
                    "summary": event.summary,
                    "metadata": dict(event.metadata),
                    "leaf_ids": list(event.leaf_ids or ()),
                }
            )
        leaf_items: List[Dict[str, Any]] = []
        for leaf in tree.leaves:
            leaf_raw_id = leaf.leaf_id or str(uuid.uuid4())
            normalized_leaf_id = self._normalize_id(leaf_raw_id)
            if not leaf.parent_id:
                raise ValueError(f"Leaf node missing parent_id in tree={tree.tree_id}")
            parent_id = event_id_lookup.get(leaf.parent_id)
            if parent_id is None:
                parent_id = self._normalize_id(leaf.parent_id)
                event_id_lookup[leaf.parent_id] = parent_id
            leaf_items.append(
                {
                    "leaf_id": normalized_leaf_id,
                    "parent_id": parent_id,
                    "text": leaf.text,
                    "metadata": dict(leaf.metadata),
                    "text_hash": leaf.text_hash,
                }
            )
        return {
            "tree": tree,
            "tree_id": tree.tree_id,
            "root_id": normalized_root_id,
            "root_topic": tree.root.topic,
            "root_image_uri": tree.root.image_uri,
            "root_metadata": dict(tree.root.metadata),
            "root_image_candidates": list(tree.root.image_candidates),
            "events": event_items,
            "leaves": leaf_items,
        }

    def _collect_leaf_entries(self, processed: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for tree in processed:
            tree_id = tree["tree_id"]
            for leaf in tree["leaves"]:
                entries.append(
                    {
                        "tree_id": tree_id,
                        "leaf_id": leaf["leaf_id"],
                        "parent_id": leaf["parent_id"],
                        "text": leaf["text"],
                        "metadata": leaf["metadata"],
                        "text_hash": leaf.get("text_hash"),
                    }
                )
        return entries

    def _collect_event_entries(
        self,
        processed: Sequence[Dict[str, Any]],
        leaf_entries: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        event_leaf_ids: Dict[str, List[str]] = {}
        for leaf in leaf_entries:
            event_leaf_ids.setdefault(leaf["parent_id"], []).append(leaf["leaf_id"])

        entries: List[Dict[str, Any]] = []
        for tree in processed:
            tree_id = tree["tree_id"]
            for event in tree["events"]:
                metadata = dict(event["metadata"])
                collected_leaf_ids = list(event_leaf_ids.get(event["event_id"], []))
                if collected_leaf_ids:
                    metadata["leaf_ids"] = collected_leaf_ids
                entries.append(
                    {
                        "tree_id": tree_id,
                        "event_id": event["event_id"],
                        "parent_id": event["parent_id"],
                        "summary": event["summary"],
                        "metadata": metadata,
                    }
                )
        return entries

    def _collect_root_entries(
        self,
        processed: Sequence[Dict[str, Any]],
        event_entries: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for idx, tree in enumerate(processed):
            entries.append(
                {
                    "index": idx,
                    "tree": tree["tree"],
                    "tree_id": tree["tree_id"],
                    "root_id": tree["root_id"],
                    "topic": tree["root_topic"],
                    "image_uri": tree["root_image_uri"],
                    "metadata": tree["root_metadata"],
                    "image_candidates": tree.get("root_image_candidates") or [],
                }
            )
        return entries

    def _build_root_alignment_text(self, tree: MemoryTree, max_sections: int = 3) -> str:
        pieces: List[str] = []
        topic_clean = (tree.root.topic or "").strip()
        if topic_clean:
            pieces.append(topic_clean)
        for idx, event in enumerate(tree.events):
            summary = (event.summary or "").strip()
            if summary:
                pieces.append(summary[:512])
            if idx + 1 >= max_sections:
                break
        return " ".join(pieces).strip()

    def _embed_texts(
        self,
        texts: Sequence[str],
        embedder: ClipEmbedding,
        batch_size: int,
        *,
        progress_bar: Optional[Any] = None,
        num_workers: int = 1,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, embedder.dim), dtype=np.float32)
        vectors: List[np.ndarray] = []
        chunks: List[List[str]] = [
            list(texts[start : start + batch_size])
            for start in range(0, len(texts), max(1, batch_size))
        ]
        if num_workers <= 1 or len(chunks) <= 1:
            for chunk in chunks:
                try:
                    chunk_vecs = embedder.embed_texts(chunk)
                except Exception as exc:
                    logger.exception("Failed to embed text chunk (%d items): %s", len(chunk), exc)
                    raise
                vectors.append(chunk_vecs)
                if progress_bar is not None:
                    progress_bar.update(len(chunk_vecs))
        else:
            ordered_results: Dict[int, np.ndarray] = {}
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(embedder.embed_texts, chunk): idx for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        chunk_vecs = future.result()
                    except Exception as exc:
                        logger.exception("Failed to embed text chunk in worker: %s", exc)
                        raise
                    ordered_results[idx] = chunk_vecs
                    if progress_bar is not None:
                        progress_bar.update(len(chunk_vecs))
            for idx in range(len(chunks)):
                chunk_vecs = ordered_results.get(idx)
                if chunk_vecs is not None:
                    vectors.append(chunk_vecs)
        if not vectors:
            return np.empty((0, embedder.dim), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def _select_root_image_vectors(
        self,
        root_entries: Sequence[Dict[str, Any]],
        embedder: ClipEmbedding,
        alignment_vectors: np.ndarray,
        *,
        image_batch_size: int,
        image_workers: int = 1,
        progress_bar: Optional[Any] = None,
    ) -> Dict[int, np.ndarray]:
        vectors: Dict[int, np.ndarray] = {}
        if not root_entries:
            return vectors
        flat_urls: List[str] = []
        flat_meta: List[Tuple[int, int]] = []
        for idx, entry in enumerate(root_entries):
            candidates = entry.get("image_candidates") or []
            if not candidates:
                continue
            for local_idx, url in enumerate(candidates):
                flat_urls.append(url)
                flat_meta.append((idx, local_idx))
        if not flat_urls or alignment_vectors.size == 0:
            return vectors

        per_entry_vectors: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        batch_size = max(1, image_batch_size)
        chunk_specs: List[Tuple[List[str], List[Tuple[int, int]]]] = []
        for start in range(0, len(flat_urls), batch_size):
            chunk_urls = flat_urls[start : start + batch_size]
            chunk_meta = flat_meta[start : start + batch_size]
            chunk_specs.append((chunk_urls, chunk_meta))

        def embed_chunk(urls: List[str], meta: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], np.ndarray]]:
            results: List[Tuple[Tuple[int, int], np.ndarray]] = []
            try:
                batch_vectors = embedder.embed_images(urls)
                results = list(zip(meta, batch_vectors))
            except Exception as exc:
                logger.warning("Batch root image embedding failed: %s", exc)
                for meta_item, url in zip(meta, urls):
                    try:
                        vec = embedder.embed_images([url])[0]
                        results.append((meta_item, vec))
                    except Exception as inner_exc:
                        logger.error("Failed to embed root image %s: %s", url, inner_exc)
            return results

        chunk_results: List[List[Tuple[Tuple[int, int], np.ndarray]]] = [list() for _ in range(len(chunk_specs))]
        if image_workers <= 1 or len(chunk_specs) <= 1:
            for idx, (chunk_urls, chunk_meta) in enumerate(chunk_specs):
                chunk_results[idx] = embed_chunk(chunk_urls, chunk_meta)
                if progress_bar is not None:
                    progress_bar.update(len(chunk_urls))
        else:
            with ThreadPoolExecutor(max_workers=image_workers) as executor:
                future_to_idx = {
                    executor.submit(embed_chunk, chunk_urls, chunk_meta): idx
                    for idx, (chunk_urls, chunk_meta) in enumerate(chunk_specs)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    chunk_urls, _ = chunk_specs[idx]
                    try:
                        chunk_results[idx] = future.result()
                    except Exception as exc:
                        logger.exception("Root image embedding worker failed: %s", exc)
                        chunk_results[idx] = []
                    if progress_bar is not None:
                        progress_bar.update(len(chunk_urls))

        for meta_vec_pairs in chunk_results:
            for (entry_idx, local_idx), vec in meta_vec_pairs:
                per_entry_vectors[entry_idx].append((local_idx, vec.astype(np.float32)))

        for idx, entry in enumerate(root_entries):
            candidate_info = per_entry_vectors.get(idx)
            if not candidate_info:
                continue
            if idx >= len(alignment_vectors):
                continue
            text_vec = alignment_vectors[idx]
            if np.linalg.norm(text_vec) < 1e-12:
                continue
            candidate_info.sort(key=lambda item: item[0])
            valid_indices = [local_idx for local_idx, _ in candidate_info]
            valid_vectors = np.vstack([vec for _, vec in candidate_info])
            scores = valid_vectors @ text_vec
            best_local = int(scores.argmax())
            best_candidate_idx = valid_indices[best_local]
            vectors[idx] = valid_vectors[best_local].astype(np.float32)
            candidates = entry.get("image_candidates") or []
            if not candidates or best_candidate_idx >= len(candidates):
                continue
            chosen_url = candidates[best_candidate_idx]
            entry["image_uri"] = chosen_url
            metadata = dict(entry.get("metadata") or {})
            metadata.setdefault("image_source", "wikipedia")
            metadata["alignment_best_score"] = float(scores[best_local])
            metadata["alignment_scores"] = [
                {"url": candidates[i], "score": float(score)}
                for i, score in zip(valid_indices, scores.tolist())
                if i < len(candidates)
            ]
            metadata["image_selection_strategy"] = "ingest_alignment"
            entry["metadata"] = metadata
        return vectors

    def _select_event_image_vectors(
        self,
        event_entries: Sequence[Dict[str, Any]],
        embedder: ClipEmbedding,
        event_text_vectors: np.ndarray,
        root_image_vectors_by_tree: Dict[str, np.ndarray],
        root_image_uris_by_tree: Dict[str, Optional[str]],
        *,
        image_batch_size: int,
        image_workers: int = 1,
        progress_bar: Optional[Any] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[str]]]:
        event_vectors: Dict[str, np.ndarray] = {}
        event_uris: Dict[str, Optional[str]] = {}
        if not event_entries:
            return event_vectors, event_uris

        per_event_candidates: Dict[int, List[str]] = {}
        flat_urls: List[str] = []
        flat_meta: List[Tuple[int, int]] = []
        for idx, entry in enumerate(event_entries):
            metadata = entry.get("metadata") or {}
            candidates = metadata.get("section_images") or []
            if isinstance(candidates, str):
                candidates = [candidates]
            candidates = [c for c in candidates if isinstance(c, str) and c]
            if not candidates:
                continue
            per_event_candidates[idx] = candidates
            for local_idx, url in enumerate(candidates):
                flat_urls.append(url)
                flat_meta.append((idx, local_idx))

        per_event_vectors: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        if flat_urls:
            batch_size = max(1, image_batch_size)
            chunk_specs: List[Tuple[List[str], List[Tuple[int, int]]]] = []
            for start in range(0, len(flat_urls), batch_size):
                chunk_urls = flat_urls[start : start + batch_size]
                chunk_meta = flat_meta[start : start + batch_size]
                chunk_specs.append((chunk_urls, chunk_meta))

            def embed_chunk(
                urls: List[str],
                meta: List[Tuple[int, int]],
            ) -> List[Tuple[Tuple[int, int], np.ndarray]]:
                results: List[Tuple[Tuple[int, int], np.ndarray]] = []
                try:
                    batch_vectors = embedder.embed_images(urls)
                    results = list(zip(meta, batch_vectors))
                except Exception as exc:
                    logger.warning("Batch event image embedding failed: %s", exc)
                    for meta_item, url in zip(meta, urls):
                        try:
                            vec = embedder.embed_images([url])[0]
                            results.append((meta_item, vec))
                        except Exception as inner_exc:
                            logger.error("Failed to embed event image %s: %s", url, inner_exc)
                return results

            chunk_results: List[List[Tuple[Tuple[int, int], np.ndarray]]] = [
                list() for _ in range(len(chunk_specs))
            ]
            if image_workers <= 1 or len(chunk_specs) <= 1:
                for idx, (chunk_urls, chunk_meta) in enumerate(chunk_specs):
                    chunk_results[idx] = embed_chunk(chunk_urls, chunk_meta)
                    if progress_bar is not None:
                        progress_bar.update(len(chunk_urls))
            else:
                with ThreadPoolExecutor(max_workers=image_workers) as executor:
                    future_to_idx = {
                        executor.submit(embed_chunk, chunk_urls, chunk_meta): idx
                        for idx, (chunk_urls, chunk_meta) in enumerate(chunk_specs)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        chunk_urls, _ = chunk_specs[idx]
                        try:
                            chunk_results[idx] = future.result()
                        except Exception as exc:
                            logger.exception("Event image embedding worker failed: %s", exc)
                            chunk_results[idx] = []
                        if progress_bar is not None:
                            progress_bar.update(len(chunk_urls))

            for meta_vec_pairs in chunk_results:
                for (event_idx, local_idx), vec in meta_vec_pairs:
                    per_event_vectors[event_idx].append((local_idx, vec.astype(np.float32)))

        for idx, entry in enumerate(event_entries):
            event_id = entry.get("event_id")
            tree_id = entry.get("tree_id")
            root_vec = root_image_vectors_by_tree.get(str(tree_id)) if tree_id is not None else None
            root_uri = root_image_uris_by_tree.get(str(tree_id)) if tree_id is not None else None

            metadata = dict(entry.get("metadata") or {})
            candidates = per_event_candidates.get(idx, [])
            candidate_info = per_event_vectors.get(idx, [])

            chosen_vec: Optional[np.ndarray] = None
            chosen_uri: Optional[str] = None
            source = None
            alignment_scores: List[Dict[str, Any]] = []

            if candidates and candidate_info:
                candidate_info.sort(key=lambda item: item[0])
                valid_indices: List[int] = []
                valid_vectors: List[np.ndarray] = []
                for local_idx, vec in candidate_info:
                    if float(np.linalg.norm(vec)) > 1e-12:
                        valid_indices.append(local_idx)
                        valid_vectors.append(vec)
                if valid_vectors:
                    vectors = np.vstack(valid_vectors)
                    if len(valid_vectors) == 1:
                        chosen_vec = vectors[0].astype(np.float32)
                        chosen_uri = candidates[valid_indices[0]]
                        source = "section"
                    else:
                        text_vec = event_text_vectors[idx] if idx < len(event_text_vectors) else None
                        if text_vec is not None and float(np.linalg.norm(text_vec)) > 1e-12:
                            scores = vectors @ text_vec
                            best_local = int(scores.argmax())
                            chosen_vec = vectors[best_local].astype(np.float32)
                            chosen_uri = candidates[valid_indices[best_local]]
                            source = "section"
                            alignment_scores = [
                                {"url": candidates[i], "score": float(score)}
                                for i, score in zip(valid_indices, scores.tolist())
                                if i < len(candidates)
                            ]
                        else:
                            chosen_vec = vectors[0].astype(np.float32)
                            chosen_uri = candidates[valid_indices[0]]
                            source = "section"

            if chosen_vec is None and root_vec is not None:
                chosen_vec = root_vec.astype(np.float32)
                chosen_uri = root_uri
                source = "root"

            if event_id is not None and chosen_vec is not None:
                event_vectors[str(event_id)] = chosen_vec
                event_uris[str(event_id)] = chosen_uri
            if source:
                metadata["event_image_uri"] = chosen_uri
                metadata["event_image_source"] = source
                if alignment_scores:
                    metadata["event_alignment_scores"] = alignment_scores
                entry["metadata"] = metadata

        return event_vectors, event_uris

    def _compose_root_vector(
        self,
        *,
        tree: MemoryTree,
        text_vec: Optional[np.ndarray],
        image_vec: Optional[np.ndarray],
        alpha_override: Optional[float],
        fusion_fn: Optional[
            Callable[[np.ndarray, Optional[np.ndarray], MemoryTree], np.ndarray]
        ],
    ) -> np.ndarray:
        if text_vec is None and image_vec is None:
            raise ValueError(f"Tree {tree.tree_id} has neither text nor image embeddings for root node.")
        if fusion_fn is not None and text_vec is not None:
            try:
                return fusion_fn(text_vec, image_vec, tree).astype(np.float32)
            except Exception as exc:
                logger.exception("Custom fusion_fn failed for tree=%s: %s", tree.tree_id, exc)
                raise
        if image_vec is None:
            return text_vec.astype(np.float32)  # type: ignore[return-value]
        if text_vec is None:
            return image_vec.astype(np.float32)

        alpha_value = self._resolve_alpha(tree, alpha_override)
        fused = alpha_value * text_vec + (1.0 - alpha_value) * image_vec
        norm = float(np.linalg.norm(fused))
        if norm > 1e-12:
            fused = fused / norm
        return fused.astype(np.float32)

    def _resolve_alpha(self, tree: MemoryTree, alpha_override: Optional[float]) -> float:
        if alpha_override is not None:
            return float(np.clip(alpha_override, 0.0, 1.0))
        meta = getattr(tree.root, "metadata", None)
        if isinstance(meta, dict) and "fusion_alpha" in meta:
            try:
                return float(np.clip(meta["fusion_alpha"], 0.0, 1.0))
            except Exception:
                pass
        return float(np.clip(self.fusion_alpha, 0.0, 1.0))

    def _upsert_roots(self, vectors: np.ndarray, root_entries: Sequence[Dict[str, Any]]) -> None:
        payloads: List[Dict[str, Any]] = []
        ids: List[str] = []
        for entry in root_entries:
            payload: Dict[str, Any] = {
                "modality": "multimodal" if entry["image_uri"] is not None else "text",
                "node_type": NodeRole.ROOT.value,
                "tree_id": entry["tree_id"],
                "topic": entry["topic"],
            }
            if entry["image_uri"]:
                payload["image_uri"] = entry["image_uri"]
            if entry["metadata"]:
                payload["metadata"] = entry["metadata"]
            payloads.append(payload)
            ids.append(entry["root_id"])
        self.upsert(vectors, payloads, ids=ids)

    def _upsert_events(self, vectors: np.ndarray, event_entries: Sequence[Dict[str, Any]]) -> None:
        if not event_entries:
            return
        payloads: List[Dict[str, Any]] = []
        ids: List[str] = []
        for entry in event_entries:
            payload: Dict[str, Any] = {
                "modality": "text",
                "node_type": NodeRole.EVENT.value,
                "tree_id": entry["tree_id"],
                "parent_id": entry["parent_id"],
                "summary": entry["summary"],
            }
            if entry["metadata"]:
                payload["metadata"] = entry["metadata"]
            payloads.append(payload)
            ids.append(entry["event_id"])
        self._upsert_to_collection(self.event_collection, vectors, payloads, ids)

    def _upsert_leaves(self, vectors: np.ndarray, leaf_entries: Sequence[Dict[str, Any]]) -> None:
        if not leaf_entries:
            return
        payloads: List[Dict[str, Any]] = []
        ids: List[str] = []
        for entry in leaf_entries:
            payload: Dict[str, Any] = {
                "modality": "text",
                "node_type": NodeRole.LEAF.value,
                "tree_id": entry["tree_id"],
                "parent_id": entry["parent_id"],
                "content": entry["text"],
            }
            if entry.get("text_hash"):
                payload["text_hash"] = entry["text_hash"]
            if entry["metadata"]:
                payload["metadata"] = entry["metadata"]
            payloads.append(payload)
            ids.append(entry["leaf_id"])
        self._upsert_to_collection(self.leaf_collection, vectors, payloads, ids)

    def _upsert_leaves_fused(
        self,
        vectors: np.ndarray,
        leaf_entries: Sequence[Dict[str, Any]],
    ) -> None:
        if not leaf_entries:
            return
        payloads: List[Dict[str, Any]] = []
        ids: List[str] = []
        for entry in leaf_entries:
            image_uri = entry.get("image_uri")
            payload: Dict[str, Any] = {
                "modality": "multimodal" if image_uri else "text",
                "node_type": NodeRole.LEAF.value,
                "tree_id": entry["tree_id"],
                "parent_id": entry["parent_id"],
                "content": entry["text"],
            }
            if image_uri:
                payload["image_uri"] = image_uri
                if entry.get("image_source"):
                    payload["image_source"] = entry.get("image_source")
            if entry.get("text_hash"):
                payload["text_hash"] = entry["text_hash"]
            if entry["metadata"]:
                payload["metadata"] = entry["metadata"]
            payloads.append(payload)
            ids.append(entry["leaf_id"])
        self._upsert_to_collection(self.leaf_collection, vectors, payloads, ids)

    def _upsert_to_collection(
        self,
        collection_name: str,
        vectors: np.ndarray,
        payloads: Sequence[Dict[str, Any]],
        ids: Sequence[str],
    ) -> None:
        if len(vectors) != len(payloads) or len(vectors) != len(ids):
            raise ValueError("Vectors, payloads and ids must have identical lengths.")
        points: List[qmodels.PointStruct] = []
        for vector, payload, point_id in zip(vectors, payloads, ids):
            points.append(
                qmodels.PointStruct(
                    id=self._normalize_id(point_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        # self.client.upsert(collection_name=collection_name, points=points, wait=False)
        batch_size = max(1, getattr(self.cfg, "batch_size", 10000) or 10000)  # smaller batch to avoid timeouts
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch, wait=False)

    # ------------------------------------------------------------------ retrieval
    def retrieve(
        self,
        embedder: ClipEmbedding,
        *,
        query_text: Optional[str] = None,
        query_image: Optional[Union[str, bytes, Any]] = None,
        alpha: Optional[float] = None,
        fusion_fn: Optional[
            Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
        ] = None,
        root_top_k: int = 3,
        event_top_k: int = 3,
        leaf_top_k: int = 5,
        root_filter: Optional[qmodels.Filter] = None,
        root_score_threshold: Optional[float] = None,
        event_score_threshold: Optional[float] = None,
        leaf_score_threshold: Optional[float] = None,
        event_filter_builder: Optional[
            Callable[[RetrievalHit], Optional[qmodels.Filter]]
        ] = None,
        leaf_filter_builder: Optional[
            Callable[[RetrievalHit, RetrievalHit], Optional[qmodels.Filter]]
        ] = None,
    ) -> List[TreeRetrievalResult]:
        """
        Retrieve hierarchical context given text and/or image query.
        """
        text_vec = (
            embedder.embed_texts([query_text])[0]
            if query_text is not None
            else None
        )
        if query_image is not None:
            query_image = _resolve_query_image_path(query_image)
        image_vec = (
            embedder.embed_images([query_image])[0]
            if query_image is not None
            else None
        )
        if text_vec is None and image_vec is None:
            raise ValueError("At least one of query_text or query_image must be provided.")

        root_query = self._build_query_vector(text_vec, image_vec, alpha, fusion_fn)
        root_points = super().search(
            root_query,
            top_k=root_top_k,
            filter_=root_filter,
            score_threshold=root_score_threshold,
        )
        results: List[TreeRetrievalResult] = []
        for root_point in root_points:
            root_hit = self._to_hit(root_point, NodeRole.ROOT, self.cfg.collection)
            tree_id = root_hit.payload.get("tree_id")
            if not tree_id:
                continue

            event_hits = self._retrieve_events(
                text_vec=text_vec,
                fallback_vec=root_query,
                root_hit=root_hit,
                top_k=event_top_k,
                score_threshold=event_score_threshold,
                filter_builder=event_filter_builder,
            )
            leaves = self._retrieve_leaves(
                text_vec=text_vec,
                fallback_vec=root_query,
                root_hit=root_hit,
                event_hits=event_hits,
                top_k=leaf_top_k,
                score_threshold=leaf_score_threshold,
                filter_builder=leaf_filter_builder,
            )
            results.append(
                TreeRetrievalResult(
                    tree_id=str(tree_id),
                    root=root_hit,
                    events=event_hits,
                    leaves=leaves,
                )
            )
        return results

    def collapsed_retrieve(
        self,
        embedder: ClipEmbedding,
        *,
        query_text: Optional[str] = None,
        query_image: Optional[Union[str, bytes, Any]] = None,
        alpha: Optional[float] = None,
        fusion_fn: Optional[
            Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
        ] = None,
        leaf_top_k: int = 20,
        leaf_filter: Optional[qmodels.Filter] = None,
        leaf_score_threshold: Optional[float] = None,
    ) -> List[CollapsedRetrievalResult]:
        """
        Collapsed retrieval: search the leaf collection directly with a fused query,
        then fold top-k leaf hits into their parent events.
        """
        text_vec = (
            embedder.embed_texts([query_text])[0]
            if query_text is not None
            else None
        )
        if query_image is not None:
            query_image = _resolve_query_image_path(query_image)
        image_vec = (
            embedder.embed_images([query_image])[0]
            if query_image is not None
            else None
        )
        if text_vec is None and image_vec is None:
            raise ValueError("At least one of query_text or query_image must be provided.")
        if leaf_top_k <= 0:
            return []

        root_query = self._build_query_vector(text_vec, image_vec, alpha, fusion_fn)
        leaf_points = self._search_collection(
            collection_name=self.leaf_collection,
            query_vec=root_query,
            top_k=leaf_top_k,
            filter_=leaf_filter,
            score_threshold=leaf_score_threshold,
        )
        leaf_hits = [self._to_hit(point, NodeRole.LEAF, self.leaf_collection) for point in leaf_points]

        tree_order: List[str] = []
        tree_to_event_leaves: Dict[str, Dict[str, List[RetrievalHit]]] = defaultdict(dict)
        event_scores: Dict[str, float] = {}
        event_ids: List[str] = []

        for hit in leaf_hits:
            tree_id = hit.payload.get("tree_id")
            raw_event_id = hit.payload.get("parent_id")
            if not tree_id or not raw_event_id:
                continue
            tree_id = str(tree_id)
            event_id = str(self._normalize_id(raw_event_id))
            if tree_id not in tree_order:
                tree_order.append(tree_id)
            event_map = tree_to_event_leaves.setdefault(tree_id, {})
            event_map.setdefault(event_id, []).append(hit)
            event_scores[event_id] = max(event_scores.get(event_id, 0.0), float(hit.score))
            if event_id not in event_ids:
                event_ids.append(event_id)

        event_payloads: Dict[str, Any] = {}
        if event_ids:
            try:
                points = self.client.retrieve(
                    collection_name=self.event_collection,
                    ids=event_ids,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.exception("Failed to retrieve event payloads: %s", exc)
                points = []
            for point in points:
                point_id = point.id if isinstance(point.id, str) else str(point.id)
                payload = getattr(point, "payload", {}) or {}
                event_payloads[point_id] = dict(payload) if isinstance(payload, dict) else payload

        results: List[CollapsedRetrievalResult] = []
        for tree_id in tree_order:
            event_map = tree_to_event_leaves.get(tree_id, {})
            if not event_map:
                continue
            event_hits: List[RetrievalHit] = []
            leaves_by_event: Dict[str, List[RetrievalHit]] = {}
            for event_id, leaves in event_map.items():
                leaves.sort(key=lambda hit: hit.score, reverse=True)
                payload = event_payloads.get(event_id, {"summary": ""})
                event_hits.append(
                    RetrievalHit(
                        id=event_id,
                        score=event_scores.get(event_id, 0.0),
                        payload=payload,
                        role=NodeRole.EVENT,
                        collection=self.event_collection,
                    )
                )
                leaves_by_event[event_id] = leaves
            event_hits.sort(key=lambda hit: hit.score, reverse=True)
            results.append(
                CollapsedRetrievalResult(
                    tree_id=tree_id,
                    events=event_hits,
                    leaves=leaves_by_event,
                )
            )
        return results

    def _build_query_vector(
        self,
        text_vec: Optional[np.ndarray],
        image_vec: Optional[np.ndarray],
        alpha: Optional[float],
        fusion_fn: Optional[
            Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
        ],
    ) -> np.ndarray:
        if fusion_fn is not None and text_vec is not None:
            vector = fusion_fn(text_vec, image_vec)
            return vector.astype(np.float32)
        if image_vec is None:
            return text_vec.astype(np.float32)  # type: ignore[return-value]
        if text_vec is None:
            return image_vec.astype(np.float32)
        alpha_value = float(np.clip(alpha if alpha is not None else self.fusion_alpha, 0.0, 1.0))
        fused = alpha_value * text_vec + (1.0 - alpha_value) * image_vec
        norm = float(np.linalg.norm(fused))
        if norm > 1e-12:
            fused = fused / norm
        return fused.astype(np.float32)

    def _retrieve_events(
        self,
        *,
        text_vec: Optional[np.ndarray],
        fallback_vec: np.ndarray,
        root_hit: RetrievalHit,
        top_k: int,
        score_threshold: Optional[float],
        filter_builder: Optional[Callable[[RetrievalHit], Optional[qmodels.Filter]]],
    ) -> List[RetrievalHit]:
        if top_k <= 0:
            return []
        query_vec = text_vec if text_vec is not None else fallback_vec
        base_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="tree_id",
                    match=qmodels.MatchValue(value=root_hit.payload.get("tree_id")),
                ),
                qmodels.FieldCondition(
                    key="parent_id",
                    match=qmodels.MatchValue(value=root_hit.id),
                ),
            ]
        )
        extra_filter = filter_builder(root_hit) if filter_builder else None
        event_filter = self._merge_filters(base_filter, extra_filter)
        points = self._search_collection(
            collection_name=self.event_collection,
            query_vec=query_vec,
            top_k=top_k,
            filter_=event_filter,
            score_threshold=score_threshold,
        )
        return [self._to_hit(point, NodeRole.EVENT, self.event_collection) for point in points]

    def _retrieve_leaves(
        self,
        *,
        text_vec: Optional[np.ndarray],
        fallback_vec: np.ndarray,
        root_hit: RetrievalHit,
        event_hits: Sequence[RetrievalHit],
        top_k: int,
        score_threshold: Optional[float],
        filter_builder: Optional[
            Callable[[RetrievalHit, RetrievalHit], Optional[qmodels.Filter]]
        ],
    ) -> Dict[str, List[RetrievalHit]]:
        if top_k <= 0 or not event_hits:
            return {}
        query_vec = text_vec if text_vec is not None else fallback_vec
        leaves: Dict[str, List[RetrievalHit]] = {}
        for event_hit in event_hits:
            base_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="tree_id",
                        match=qmodels.MatchValue(value=root_hit.payload.get("tree_id")),
                    ),
                    qmodels.FieldCondition(
                        key="parent_id",
                        match=qmodels.MatchValue(value=event_hit.id),
                    ),
                ]
            )
            extra_filter = filter_builder(event_hit, root_hit) if filter_builder else None
            leaf_filter = self._merge_filters(base_filter, extra_filter)
            points = self._search_collection(
                collection_name=self.leaf_collection,
                query_vec=query_vec,
                top_k=top_k,
                filter_=leaf_filter,
                score_threshold=score_threshold,
            )
            leaf_hits = [self._to_hit(point, NodeRole.LEAF, self.leaf_collection) for point in points]
            if leaf_hits:
                leaves[event_hit.id] = leaf_hits
        return leaves

    def _search_collection(
        self,
        *,
        collection_name: str,
        query_vec: np.ndarray,
        top_k: int,
        filter_: Optional[qmodels.Filter],
        score_threshold: Optional[float],
    ) -> List[qmodels.ScoredPoint]:
        assert query_vec.ndim == 1, "query_vec must be 1D for search"
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vec.astype(np.float32).tolist(),
            query_filter=filter_,
            limit=top_k,
            score_threshold=score_threshold,
        )

    def _to_hit(self, point: qmodels.ScoredPoint, role: NodeRole, collection: str) -> RetrievalHit:
        payload = getattr(point, "payload", {}) or {}
        point_id = point.id if isinstance(point.id, str) else str(point.id)
        score = float(getattr(point, "score", 0.0))
        return RetrievalHit(
            id=point_id,
            score=score,
            payload=dict(payload) if isinstance(payload, dict) else payload,
            role=role,
            collection=collection,
        )

    def _merge_filters(
        self,
        base: qmodels.Filter,
        extra: Optional[qmodels.Filter],
    ) -> qmodels.Filter:
        if extra is None:
            return base
        merged = qmodels.Filter(
            must=list((base.must or [])) + list((extra.must or [])),
            should=list((base.should or [])) + list((extra.should or [])),
            must_not=list((base.must_not or [])) + list((extra.must_not or [])),
        )
        return merged


# ------------------------------------------------------------------ tree construction helpers
def iter_wiki_dict(store: Dict[str, Any], limit: Optional[int] = None):
    """
    Yield (doc_id, payload) pairs from a wiki dict JSON with optional limit.
    """
    for idx, (doc_id, payload) in enumerate(store.items()):
        if limit is not None and idx >= limit:
            break
        yield doc_id, payload


def chunk_paragraphs(text: str, *, chunk_size: int = 1024, overlap: int = 120) -> Iterable[str]:
    """
    Chunk a long paragraph into overlapping word windows for leaf nodes.
    """
    tokens = text.split()
    if not tokens:
        return []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        yield " ".join(tokens[start:end])
        start += step


def summarize_event(paragraph: str) -> str:
    """
    Lightweight fallback summarizer using the first sentence / 256 chars.
    """
    stripped = paragraph.strip()
    if not stripped:
        return ""
    sentence = stripped.split(".", 1)[0]
    return (sentence if sentence else stripped)[:256]


def summarize_event_with_llm(
    leaf_texts: Sequence[str],
    *,
    llm: Any,
    model: str,
    section_title: str = "",
    max_leaf_texts: Optional[int] = None,
    max_chars_per_leaf: int = 1200,
    request_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate an event summary using an OpenAI-compatible client (e.g., DashScope compatible-mode).

    Notes:
        - The input MUST be the leaf node texts for that section (per design).
        - ``llm`` is expected to be an ``openai.OpenAI(...)`` client with
          ``llm.chat.completions.create(model=..., messages=[...], ...)``.
    """
    texts = [t.strip() for t in (leaf_texts or []) if isinstance(t, str) and t.strip()]
    if not texts:
        return ""

    if max_leaf_texts is not None and max_leaf_texts > 0:
        texts = texts[:max_leaf_texts]

    clipped: List[str] = []
    for text in texts:
        clipped.append(text[:max_chars_per_leaf] if max_chars_per_leaf and max_chars_per_leaf > 0 else text)

    title_part = f"Section title: {section_title.strip()}\n" if section_title and section_title.strip() else ""
    joined = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(clipped))
    prompt = (
        "You are a careful summarizer.\n"
        "Given the following wiki paragraphs (leaf node texts), write a concise summary.\n"
        "Requirements:\n"
        "- Output ONLY the summary text, no bullet points, no quotes.\n\n"
        f"{title_part}"
        "Paragraphs:\n"
        f"{joined}\n"
    )

    messages = [
        {"role": "system", "content": "You are a helpful summarizer."},
        {"role": "user", "content": prompt},
    ]
    kwargs: Dict[str, Any] = {
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    if request_kwargs:
        kwargs.update(request_kwargs)

    resp = llm.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    content = ""
    try:
        content = resp.choices[0].message.content or ""
    except Exception:
        content = str(resp)
    return str(content).strip()


def build_image_index(
    image_urls: Sequence[str],
    image_section_indices: Sequence[int],
    num_sections: int,
) -> Dict[Optional[int], List[int]]:
    """
    Map section index -> list of indices in image_urls. Unmatched images fall under None.
    """
    mapping: Dict[Optional[int], List[int]] = defaultdict(list)
    for idx, _ in enumerate(image_urls):
        section_idx: Optional[int] = None
        if image_section_indices and idx < len(image_section_indices):
            candidate = image_section_indices[idx]
            if isinstance(candidate, int) and 0 <= candidate < num_sections:
                section_idx = candidate
        mapping[section_idx].append(idx)
    return mapping


def build_tree(
    wiki_url: str,
    payload: Dict[str, Any],
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 120,
    max_summary_sections: Optional[int] = None,
    llm: Optional[Any] = None,
    llm_model: str = "qwen-plus",
    llm_request_kwargs: Optional[Dict[str, Any]] = None,
    llm_workers: int = 1,
    show_progress: bool = True,
) -> MemoryTree:
    """
    Build a MemoryTree from a wiki-style payload (title, section_texts, images).

    If ``llm`` is provided (e.g., ``openai.OpenAI(...)`` client for DashScope compatible-mode),
    each event summary is generated via ``llm.chat.completions.create(...)`` using the corresponding leaf texts as input. The
    ``max_summary_sections`` value caps how many leaf texts are included in that prompt
    (use ``None`` for unlimited).

    Set ``llm_workers>1`` to summarize sections concurrently.

    Set ``show_progress=True`` to see a tqdm progress bar over sections (if tqdm is installed).
    """
    logger.info("Building tree for %s", wiki_url)
    topic = (payload.get("title") or "").strip()
    section_titles = list(payload.get("section_titles") or [])
    section_texts = list(payload.get("section_texts") or [])
    image_urls = list(payload.get("image_urls") or [])
    image_section_indices = list(payload.get("image_section_indices") or [])
    section_count = max(len(section_titles), len(section_texts))
    section_titles_clean = [
        (section_titles[idx] if idx < len(section_titles) else "").strip()
        for idx in range(section_count)
    ]
    section_texts_clean = [
        (section_texts[idx] if idx < len(section_texts) else "").strip()
        for idx in range(section_count)
    ]

    root_metadata: Dict[str, Any] = {
        "source_url": payload.get("url", wiki_url),
        "num_sections": section_count,
        "num_images": len(image_urls),
    }
    if image_urls:
        root_metadata["image_source"] = "wikipedia"

    root_image_url: Optional[str] = None
    if image_urls:
        root_image_url = image_urls[0]

    root = RootNode(
        topic=topic or wiki_url,
        root_id=wiki_url,
        image_uri=root_image_url,
        metadata=root_metadata,
        image_candidates=tuple(image_urls),
    )

    section_image_index = build_image_index(image_urls, image_section_indices, section_count)
    event_specs: List[Dict[str, Any]] = []
    leaves: List[LeafNode] = []

    section_iter: Iterable[int] = range(section_count)
    if show_progress and tqdm is not None:
        progress_position = 0
        instances = getattr(tqdm, "_instances", None)
        if instances is not None:
            try:
                progress_position = len(instances)
            except TypeError:
                progress_position = 0
        section_iter = tqdm(
            section_iter,
            desc=f"Sections[{wiki_url}]",
            leave=False,
            position=progress_position,
        )

    use_llm_concurrency = llm is not None and llm_workers > 1

    for sec_idx in section_iter:
        title_clean = section_titles_clean[sec_idx] if sec_idx < len(section_titles_clean) else ""
        text_clean = section_texts_clean[sec_idx] if sec_idx < len(section_texts_clean) else ""
        if not title_clean and not text_clean:
            continue

        event_id = f"{wiki_url}::sec{sec_idx}"
        event_metadata: Dict[str, Any] = {
            "section_index": sec_idx,
            "section_title": title_clean,
            "source_url": payload.get("url", wiki_url),
        }
        if text_clean:
            event_metadata["section_preview"] = text_clean[:512]
        image_indices = section_image_index.get(sec_idx, [])
        if image_indices:
            event_metadata.setdefault("section_images", [image_urls[idx] for idx in image_indices])

        section_leaf_ids: List[str] = []
        section_leaf_texts: List[str] = []
        if text_clean:
            paragraphs = list(chunk_paragraphs(text_clean, chunk_size=chunk_size, overlap=chunk_overlap))
            if not paragraphs:
                paragraphs = [text_clean]
            for para_idx, paragraph in enumerate(paragraphs):
                paragraph_clean = paragraph.strip()
                if not paragraph_clean:
                    continue
                leaf_id = f"{event_id}::leaf{para_idx}"
                section_leaf_ids.append(leaf_id)
                section_leaf_texts.append(paragraph_clean)
                text_hash = hashlib.sha1(paragraph_clean.encode("utf-8")).hexdigest()
                leaves.append(
                    LeafNode(
                        text=paragraph_clean,
                        parent_id=event_id,
                        leaf_id=leaf_id,
                        metadata={
                            "section_index": sec_idx,
                            "section_title": title_clean,
                            "paragraph_index": para_idx,
                            "source_url": payload.get("url", wiki_url),
                        },
                        text_hash=text_hash,
                    )
                )

        summary = "" if use_llm_concurrency else summarize_event(text_clean)
        if llm is not None and section_leaf_texts and llm_workers <= 1:
            try:
                llm_summary = summarize_event_with_llm(
                    section_leaf_texts,
                    llm=llm,
                    model=llm_model,
                    section_title=title_clean,
                    max_leaf_texts=max_summary_sections,
                    request_kwargs=llm_request_kwargs,
                )
                if llm_summary:
                    summary = llm_summary
            except Exception:
                logger.exception("LLM summarization failed for %s sec=%s; falling back.", wiki_url, sec_idx)

        event_specs.append(
            {
                "sec_idx": sec_idx,
                "event_id": event_id,
                "event_metadata": event_metadata,
                "leaf_ids": tuple(section_leaf_ids),
                "leaf_texts": section_leaf_texts,
                "fallback_summary": "" if use_llm_concurrency else summary,
                "fallback_text": text_clean,
                "section_title": title_clean,
            }
        )

    summaries: Dict[int, str] = {int(spec["sec_idx"]): str(spec["fallback_summary"] or "") for spec in event_specs}
    if llm is not None and llm_workers > 1:
        summary_progress = None
        if show_progress and tqdm is not None:
            total_summaries = sum(1 for spec in event_specs if spec.get("leaf_texts"))
            if total_summaries > 0:
                progress_position = 0
                instances = getattr(tqdm, "_instances", None)
                if instances is not None:
                    try:
                        progress_position = len(instances)
                    except TypeError:
                        progress_position = 0
                summary_progress = tqdm(
                    total=total_summaries,
                    desc=f"Summaries[{wiki_url}]",
                    leave=False,
                    position=progress_position,
                )
        max_workers = max(1, int(llm_workers))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_sec: Dict[Any, int] = {}
                for spec in event_specs:
                    leaf_texts = spec.get("leaf_texts") or []
                    if not leaf_texts:
                        continue
                    sec_idx = int(spec["sec_idx"])
                    future = executor.submit(
                        summarize_event_with_llm,
                        leaf_texts,
                        llm=llm,
                        model=llm_model,
                        section_title=str(spec.get("section_title") or ""),
                        max_leaf_texts=max_summary_sections,
                        request_kwargs=llm_request_kwargs,
                    )
                    future_to_sec[future] = sec_idx

                for future in as_completed(future_to_sec):
                    sec_idx = future_to_sec[future]
                    try:
                        llm_summary = future.result()
                        if llm_summary:
                            summaries[sec_idx] = llm_summary
                    except Exception:
                        logger.exception("LLM summarization failed for %s sec=%s; falling back.", wiki_url, sec_idx)
                    if summary_progress is not None:
                        summary_progress.update(1)
        finally:
            if summary_progress is not None:
                summary_progress.close()

        for spec in event_specs:
            sec_idx = int(spec["sec_idx"])
            if summaries.get(sec_idx):
                continue
            fallback_text = str(spec.get("fallback_text") or "")
            if fallback_text:
                summaries[sec_idx] = summarize_event(fallback_text)

    events: List[EventNode] = [
        EventNode(
            summary=summaries.get(int(spec["sec_idx"]), ""),
            parent_id=wiki_url,
            event_id=str(spec["event_id"]),
            metadata=dict(spec["event_metadata"]),
            leaf_ids=spec["leaf_ids"],
        )
        for spec in event_specs
    ]

    return MemoryTree(tree_id=wiki_url, root=root, events=events, leaves=leaves)
