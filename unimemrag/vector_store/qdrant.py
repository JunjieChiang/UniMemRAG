import numpy as np
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from config import Config
from unimemrag.embedding.models.ClipEmbedding import ClipEmbedding
from qdrant_client.http import models as qmodels
from typing import List, Dict, Any, Iterable, Optional, Tuple, Union, Callable, Sequence

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _batched(iterable: Iterable, batch_size: int) -> Iterable[List[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    iterator = iter(iterable)
    while True:
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(iterator))
        except StopIteration:
            if batch:
                yield batch
            break
        else:
            yield batch

# -----------------------------
# QdrantStore: 建库、入库、检索
# 设计：单向量字段 `clip`，点可代表“图像”或“文本”
# payload 记录 modality / content / uri / meta
# -----------------------------
class QdrantStore:
    def __init__(self, cfg: Config, vector_size: int):
        self.cfg = cfg
        self.client = QdrantClient(
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
            grpc_port=cfg.qdrant_grpc_port,
            prefer_grpc=cfg.prefer_grpc,
            timeout=cfg.timeout_sec,        # 可以是 float 或 dict
            )
        self.vector_name = "clip"
        self._ensure_collection(vector_size)

    def clear_collection(self):
        self.client.delete_collection(self.cfg.collection)
        
    @staticmethod
    def _default_image_payload(image_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(image_path)
        return {
            "uri": path.as_posix(),
            "modality": "image",
            "content": path.stem,
        }

    @staticmethod
    def default_caption_from_name(image_path: Union[str, Path]) -> Optional[str]:
        name = Path(image_path).stem
        if "_gt_" in name:
            return name.split("_gt_", 1)[1].replace("_", " ").replace("-", " ")
        return None

    def ingest_images(
        self,
        embedder: ClipEmbedding,
        image_items: Iterable[Union[str, Path]],
        payload_builder: Optional[Callable[[Union[str, Path]], Dict[str, Any]]] = None,
        caption_builder: Optional[Callable[[Union[str, Path]], Optional[str]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        batch_size = batch_size or self.cfg.batch_size or 100
        payload_builder = payload_builder or self._default_image_payload
        failures: List[Dict[str, Any]] = []
        images_indexed = 0
        texts_indexed = 0

        items = image_items
        if show_progress and tqdm is not None:
            if isinstance(image_items, Sequence):
                items = tqdm(image_items, desc="Embedding images", leave=False)
            else:
                buffered = list(image_items)
                items = tqdm(buffered, desc="Embedding images", leave=False)
        for batch in _batched(items, batch_size):
            targets: List[Tuple[str, Dict[str, Any]]] = []
            text_candidates: List[Tuple[str, str]] = []
            for raw_item in batch:
                path_str = Path(raw_item).as_posix()
                try:
                    payload = payload_builder(raw_item)
                except Exception as exc:
                    failures.append({"uri": path_str, "error": f"payload_builder: {exc}"})
                    continue
                if payload is None:
                    continue
                payload.setdefault("uri", path_str)
                payload.setdefault("modality", "image")
                targets.append((path_str, payload))
                if caption_builder:
                    try:
                        caption = caption_builder(raw_item)
                    except Exception as exc:
                        failures.append({"uri": path_str, "error": f"caption_builder: {exc}"})
                        caption = None
                    if caption:
                        text_candidates.append((path_str, caption))
            if not targets:
                continue

            image_paths = [p for p, _ in targets]
            image_payloads = [pl for _, pl in targets]
            try:
                vectors = embedder.embed_images(image_paths)
                self.upsert(vectors, image_payloads, ids=[pl["uri"] for pl in image_payloads])
                images_indexed += len(image_payloads)
            except Exception as exc:
                for path_str, payload in targets:
                    try:
                        vector = embedder.embed_images([path_str])[0]
                        self.upsert(vector.reshape(1, -1), [payload], ids=[payload["uri"]])
                        images_indexed += 1
                    except Exception as inner_exc:
                        failures.append({"uri": path_str, "error": str(inner_exc)})

            if text_candidates:
                captions = [caption for _, caption in text_candidates]
                text_payloads = [{"uri": uri, "modality": "text", "content": caption} for uri, caption in text_candidates]
                text_ids = [f"{uri}::text" for uri, _ in text_candidates]
                try:
                    text_vectors = embedder.embed_texts(captions)
                    self.upsert(text_vectors, text_payloads, ids=text_ids)
                    texts_indexed += len(text_payloads)
                except Exception:
                    for (uri, caption), payload, text_id in zip(text_candidates, text_payloads, text_ids):
                        try:
                            vector = embedder.embed_texts([caption])[0]
                            self.upsert(vector.reshape(1, -1), [payload], ids=[text_id])
                            texts_indexed += 1
                        except Exception as inner_exc:
                            failures.append({"uri": uri, "error": f"text embedding: {inner_exc}"})

        return {
            "images_indexed": images_indexed,
            "texts_indexed": texts_indexed,
            "failures": failures,
        }

    def _normalize_id(self, raw_id: Union[str, int, uuid.UUID]) -> Union[str, int]:
        if isinstance(raw_id, int):
            return raw_id
        if isinstance(raw_id, uuid.UUID):
            return str(raw_id)
        if isinstance(raw_id, str):
            try:
                uuid.UUID(raw_id)
                return raw_id
            except ValueError:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
        return str(uuid.uuid4())

    def _ensure_collection(self, vector_size: int):
        collections = {c.name for c in self.client.get_collections().collections}
        if self.cfg.collection not in collections:
            self.client.create_collection(
                collection_name=self.cfg.collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=self.cfg.distance,
                    on_disk=self.cfg.on_disk,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=20000
                ),
            )
        # 打开 payload 索引（常用字段）
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        # 为常用筛选字段建索引
        for field in ["modality", "uri"]:
            try:
                self.client.create_payload_index(
                    collection_name=self.cfg.collection,
                    field_name=field,
                    field_type="keyword",
                )
            except Exception:
                pass  # 已存在则忽略

    def upsert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: Optional[List[Union[str, int]]] = None,
    ):
        assert len(vectors) == len(payloads), "vectors/payloads length mismatch"
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        else:
            ids = [self._normalize_id(x) for x in ids]

        points = []
        for _id, vec, pl in zip(ids, vectors, payloads):
            points.append(
                qmodels.PointStruct(
                    id=_id,
                    vector=vec.tolist(),
                    payload=pl,
                )
            )
        # self.client.upsert(
        #     collection_name=self.cfg.collection,
        #     points=points,
        #     wait=False,
        # )
        batch_size = max(1, getattr(self.cfg, "batch_size", 10000) or 10000)
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.cfg.collection,
                points=batch,
                wait=True,
            )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        filter_: Optional[qmodels.Filter] = None,
        score_threshold: Optional[float] = None,
    ) -> List[qmodels.ScoredPoint]:
        assert query_vec.ndim == 1, "query_vec must be 1D"
        res = self.client.search(
            collection_name=self.cfg.collection,
            query_vector=query_vec.astype(np.float32).tolist(),
            query_filter=filter_,
            limit=top_k,
            score_threshold=score_threshold,
        )
        return res

    def delete_by_filter(self, filter_: qmodels.Filter):
        self.client.delete(
            collection_name=self.cfg.collection,
            points_selector=qmodels.FilterSelector(filter=filter_),
            wait=True,
        )
