
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from qdrant_client.http import models as qmodels

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - running outside notebooks
    display = None


def _payload_to_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, 'dict'):
        return payload.dict()
    try:
        return dict(payload)
    except Exception:
        return {"value": payload}


@dataclass
class ImageNode:
    """Represent an image result stored in Qdrant."""

    id: str
    score: float
    uri: str
    content: Optional[str] = None
    modality: Optional[str] = None
    payload: Dict[str, Any] = None

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}

    def resolve_path(self, base_dir: Optional[Path] = None) -> Path:
        uri = self.uri
        if uri.startswith(('http://', 'https://')):
            return Path(uri)
        path = Path(uri)
        if path.is_absolute() and path.exists():
            return path
        candidates = []
        if base_dir:
            candidates.append(Path(base_dir) / path)
        candidates.append(Path.cwd() / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path

    def load_image(self, base_dir: Optional[Path] = None) -> Image.Image:
        if self.uri.startswith(('http://', 'https://')):
            import requests

            response = requests.get(self.uri, stream=True, timeout=20)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')

        resolved = self.resolve_path(base_dir)
        return Image.open(resolved).convert('RGB')


def as_image_node(point: qmodels.ScoredPoint) -> ImageNode:
    payload = _payload_to_dict(point.payload)
    uri = payload.get('uri') or payload.get('path')
    if not uri:
        raise ValueError("ScoredPoint payload missing 'uri' or 'path'")

    point_id = str(point.id)
    modality = payload.get('modality')
    content = payload.get('content')
    return ImageNode(
        id=point_id,
        score=point.score,
        uri=uri,
        content=content,
        modality=modality,
        payload=payload,
    )


def show(node: ImageNode, base_dir: Optional[Path] = None) -> Image.Image:
    """Display the image represented by the ImageNode and return the PIL image."""

    image = node.load_image(base_dir=base_dir)
    if display is not None:
        display(image)
    else:  # pragma: no cover - fallback for CLI environments
        image.show()
    return image
