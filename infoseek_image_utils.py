from pathlib import Path
from typing import Optional, Union

# Allowed image extensions for InfoSeek images.
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg"}


def resolve_images_root(images_root: Optional[Union[str, Path]]) -> Path:
    """Return the absolute images root, defaulting to benchmark/oven relative to this file."""
    if images_root is None:
        base_dir = Path(__file__).resolve().parent
        images_root = (base_dir / "../benchmark/oven/").resolve()
    else:
        images_root = Path(images_root).expanduser().resolve()
    return images_root
