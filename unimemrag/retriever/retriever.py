from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
from qdrant_client.http import models as qmodels
from unimemrag.embedding.models import ClipEmbedding
from unimemrag.vector_store.qdrant import QdrantStore
from ..utils.image_node import ImageNode, as_image_node
from ..utils.text_node import TextNode, as_text_node
from unimemrag.utils.log_config import setup_logger

if TYPE_CHECKING:
    from unimemrag.vlm.QwenVL import QwenVL


logger = setup_logger()


# -----------------------------
# Retriever: 文搜图 / 图搜文 / 文搜文 / 图搜图
# -----------------------------
class Retriever:
    def __init__(self, embedder: ClipEmbedding, store: QdrantStore, top_k: int = 3, image_top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.image_top_k = image_top_k

    def _modality_filter(self, modality: Optional[str]) -> Optional[qmodels.Filter]:
        if modality is None:
            return None
        return qmodels.Filter(
            must=[qmodels.FieldCondition(key="modality", match=qmodels.MatchValue(value=modality))]
        )

    def _wrap_results(
        self,
        points: List[qmodels.ScoredPoint],
        as_nodes: bool,
        target_modality: Optional[str],
    ) -> List[Union[qmodels.ScoredPoint, ImageNode, TextNode]]:
        if not as_nodes:
            return points
        nodes: List[Union[ImageNode, TextNode]] = []
        modality_hint = (target_modality or "").lower() if target_modality else ""
        for point in points:
            converter = None
            if modality_hint == "image":
                converter = as_image_node
            elif modality_hint == "text":
                converter = as_text_node
            else:
                payload_modality = ""
                payload = getattr(point, "payload", None)
                if isinstance(payload, dict):
                    payload_modality = (payload.get("modality") or "").lower()
                elif hasattr(payload, "dict"):
                    try:
                        payload_modality = (payload.dict().get("modality") or "").lower()
                    except Exception:
                        payload_modality = ""
                else:
                    try:
                        payload_modality = (payload.get("modality") or "").lower()  # type: ignore[attr-defined]
                    except Exception:
                        payload_modality = ""
                if payload_modality == "image":
                    converter = as_image_node
                elif payload_modality == "text":
                    converter = as_text_node
            if converter is None:
                converter = as_text_node if modality_hint != "image" else as_image_node
            try:
                nodes.append(converter(point))
            except ValueError:
                fallback = as_image_node if converter is as_text_node else as_text_node
                try:
                    nodes.append(fallback(point))
                except ValueError:
                    continue
        return nodes

    def _log_vlm_input(
        self,
        *,
        query: str,
        prompt: str,
        nodes: List[Union[ImageNode, TextNode]],
        note: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        summary = [
            {
                "id": node.id,
                "uri": node.uri,
                "modality": node.modality,
                "content": node.content,
                "score": float(node.score),
            }
            for node in nodes
        ]
        meta: Dict[str, Any] = {
            "query": query,
            "note": note,
            "node_count": len(nodes),
            "nodes": summary,
        }
        if extra:
            meta.update(extra)
        logger.info(
            "VLM input prepared | meta={meta}\nPrompt:\n{prompt}",
            meta=meta,
            prompt=prompt,
        )

    # 用文本 query 去搜“图像”/“文本”
    def search_by_text(
        self,
        query: str,
        target_modality: Optional[str] = None,
        as_nodes: bool = True,
    ) -> List[Union[qmodels.ScoredPoint, ImageNode, TextNode]]:
        qv = self.embedder.embed_texts([query])[0]
        points = self.store.search(qv, top_k=self.top_k, filter_=self._modality_filter(target_modality))
        return self._wrap_results(points, as_nodes=as_nodes, target_modality=target_modality)

    # 用图像去搜“文本”/“图像”
    def search_by_image(
        self,
        image: Union[Image.Image, str, bytes],
        target_modality: Optional[str] = None,
        *,
        image_top_k: Optional[int] = None,
        top_k: Optional[int] = None,
        as_nodes: bool = True,
    ) -> List[Union[qmodels.ScoredPoint, ImageNode, TextNode]]:
        qv = self.embedder.embed_images([image])[0]
        if target_modality:
            target_modality_value = target_modality.lower()
            if target_modality_value == "image":
                search_limit = image_top_k if image_top_k is not None else self.image_top_k
            elif target_modality_value == "text":
                search_limit = top_k if top_k is not None else self.top_k
            else:
                search_limit = top_k if top_k is not None else self.top_k
            points = self.store.search(
                qv,
                top_k=search_limit,
                filter_=self._modality_filter(target_modality),
            )
            return self._wrap_results(points, as_nodes=as_nodes, target_modality=target_modality)

        image_limit = image_top_k if image_top_k is not None else self.image_top_k
        text_limit = top_k if top_k is not None else self.top_k

        combined: List[Union[qmodels.ScoredPoint, ImageNode, TextNode]] = []
        seen_ids: Set[str] = set()

        if image_limit:
            image_points = self.store.search(
                qv,
                top_k=image_limit,
                filter_=self._modality_filter("image"),
            )
            image_items = self._wrap_results(image_points, as_nodes=as_nodes, target_modality="image")
            for item in image_items:
                item_id: Optional[str]
                if as_nodes:
                    item_id = getattr(item, "id", None)
                else:
                    item_id = str(getattr(item, "id", None)) if getattr(item, "id", None) is not None else None
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                combined.append(item)

        if text_limit:
            text_points = self.store.search(
                qv,
                top_k=text_limit,
                filter_=self._modality_filter("text"),
            )
            text_items = self._wrap_results(text_points, as_nodes=as_nodes, target_modality="text")
            for item in text_items:
                item_id: Optional[str]
                if as_nodes:
                    item_id = getattr(item, "id", None)
                else:
                    item_id = str(getattr(item, "id", None)) if getattr(item, "id", None) is not None else None
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                combined.append(item)

        return combined

    def answer_with_vlm(
        self,
        query: str,
        vlm: "QwenVL",
        nodes: Sequence[Union[ImageNode, TextNode]],
        *,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Union[ImageNode, TextNode]]]:
        """Use previously retrieved nodes to answer the query with the provided VLM."""

        node_list: List[Union[ImageNode, TextNode]] = []
        for node in nodes:
            if isinstance(node, (ImageNode, TextNode)):
                node_list.append(node)
            else:
                # Skip scored points or unexpected types; callers should pass nodes
                continue

        gen_kwargs = gen_kwargs or {}
        if not node_list:
            prompt = (
                "No related context was retrieved, but please answer the question.\n"
                f"Question: {query}"
            )
            self._log_vlm_input(
                query=query,
                prompt=prompt,
                nodes=[],
                note="no_nodes",
            )
            answer = vlm.complete(prompt, images=None, **gen_kwargs)
            return answer, []

        image_nodes: List[ImageNode] = []
        text_nodes: List[TextNode] = []
        for node in node_list:
            if isinstance(node, TextNode):
                text_nodes.append(node)
            elif isinstance(node, ImageNode):
                image_nodes.append(node)

        prompt_parts = ["You are a helpful assistant."]
        text_contexts = [n.content for n in text_nodes if n.content]
        if text_contexts:
            prompt_parts.append("Context:\n" + "\n".join(text_contexts))
        prompt_parts.append(f"Question: {query}")
        prompt = "\n\n".join(prompt_parts)

        vlm_images: List[Union[str, Image.Image]] = []
        for node in image_nodes:
            try:
                vlm_images.append(node.load_image())
            except Exception:
                if node.uri:
                    vlm_images.append(node.uri)

        self._log_vlm_input(
            query=query,
            prompt=prompt,
            nodes=node_list,
            note="vlm_with_retrieved_context",
            extra={
                "image_count": len(image_nodes),
                "text_count": len(text_nodes),
                "image_inputs": [str(img) for img in vlm_images],
            },
        )
        answer = vlm.complete(prompt, images=vlm_images or None, **gen_kwargs)
        return answer, node_list
