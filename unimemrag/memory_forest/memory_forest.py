from __future__ import annotations

import hashlib
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from qdrant_client.http import models as qmodels

from unimemrag.embedding.models.ClipEmbedding import ClipEmbedding
from unimemrag.vector_store.qdrant import QdrantStore


logger = logging.getLogger(__name__)


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
        fusion_alpha: float = 0.6,
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
    max_summary_sections: int = 3,
    show_progress: bool = False,
) -> MemoryTree:
    """
    Build a MemoryTree from a wiki-style payload (title, section_texts, images).

    The ``embedder`` argument is retained for backward compatibility but is no longer used;
    all embedding/vectorization happens during ingestion.

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
    events: List[EventNode] = []
    leaves: List[LeafNode] = []

    section_iter: Iterable[int] = range(section_count)
    if show_progress and tqdm is not None:
        section_iter = tqdm(
            section_iter,
            desc=f"Sections[{wiki_url}]",
            leave=False,
        )

    for sec_idx in section_iter:
        title_clean = section_titles_clean[sec_idx] if sec_idx < len(section_titles_clean) else ""
        text_clean = section_texts_clean[sec_idx] if sec_idx < len(section_texts_clean) else ""
        if not title_clean and not text_clean:
            continue

        summary = title_clean or summarize_event(text_clean)
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

        events.append(
            EventNode(
                summary=summary,
                parent_id=wiki_url,
                event_id=event_id,
                metadata=event_metadata,
                leaf_ids=tuple(section_leaf_ids),
            )
        )

    return MemoryTree(tree_id=wiki_url, root=root, events=events, leaves=leaves)
