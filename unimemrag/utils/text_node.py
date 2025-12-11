from dataclasses import dataclass
from typing import Any, Dict, Optional

from qdrant_client.http import models as qmodels


def _payload_to_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "dict"):
        return payload.dict()
    try:
        return dict(payload)
    except Exception:
        return {"value": payload}


@dataclass
class TextNode:
    """Represent a text result stored in Qdrant."""

    id: str
    score: float
    content: str
    uri: Optional[str] = None
    modality: Optional[str] = None
    payload: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {}


def as_text_node(point: qmodels.ScoredPoint) -> TextNode:
    payload = _payload_to_dict(point.payload)
    content = payload.get("content")
    if content is None:
        raise ValueError("ScoredPoint payload missing 'content'")

    point_id = str(point.id)
    uri = payload.get("uri") or payload.get("path")
    modality = payload.get("modality") or "text"
    return TextNode(
        id=point_id,
        score=point.score,
        content=content,
        uri=uri,
        modality=modality,
        payload=payload,
    )

