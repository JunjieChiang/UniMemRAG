from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


logger = logging.getLogger(__name__)

_THREAD_LOCAL = threading.local()
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0 Safari/537.36"
    )
}
_KNOWN_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug for doc ids/URLs."""
    slug = re.sub(r"[^\w\-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "doc"


def _guess_suffix(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix in _KNOWN_SUFFIXES else ".jpg"


def _get_session(headers: Optional[Mapping[str, str]] = None) -> requests.Session:
    session: Optional[requests.Session] = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(_DEFAULT_HEADERS)
        if headers:
            session.headers.update(headers)
        setattr(_THREAD_LOCAL, "session", session)
    return session


def _download_single_image(
    *,
    doc_id: str,
    index: int,
    url: str,
    cache_root: Path,
    timeout: Tuple[int, int],
    headers: Optional[Mapping[str, str]],
    resume: bool,
) -> Tuple[str, Optional[str]]:
    """
    Worker helper. Returns (url, local_path or None).
    """
    if not isinstance(url, str) or not url:
        return url, None
    if not url.startswith("http://") and not url.startswith("https://"):
        return url, None

    suffix = _guess_suffix(url)
    doc_dir = cache_root / _slugify(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{index:04d}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}{suffix}"
    target_path = doc_dir / filename
    if resume and target_path.exists() and target_path.stat().st_size > 0:
        return url, str(target_path)

    try:
        session = _get_session(headers)
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        return url, str(target_path)
    except Exception as exc:  # pragma: no cover - network failures unavoidable
        # logger.warning("Failed to download %s: %s", url, exc)
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        return url, None


def download_images_for_kb(
    kb_store: Mapping[str, Dict[str, Any]],
    cache_root: Path,
    *,
    max_workers: int = 8,
    timeout: Tuple[int, int] = (5, 30),
    headers: Optional[Mapping[str, str]] = None,
    resume: bool = True,
    show_progress: bool = True,
) -> Dict[str, str]:
    """
    Download all image URLs contained in a KB dict to a local cache directory.

    Returns a mapping {original_url: local_path}. Re-running the function
    skips files that already exist when ``resume`` is True.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    download_tasks = []
    for doc_id, payload in kb_store.items():
        image_urls = payload.get("image_urls") or []
        for idx, url in enumerate(image_urls):
            if not isinstance(url, str) or not url:
                continue
            download_tasks.append((doc_id, idx, url))

    if not download_tasks:
        return {}

    url_to_local: Dict[str, str] = {}
    progress = tqdm(total=len(download_tasks), desc="Downloading images", leave=False) if show_progress and tqdm else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _download_single_image,
                doc_id=doc_id,
                index=idx,
                url=url,
                cache_root=cache_root,
                timeout=timeout,
                headers=headers,
                resume=resume,
            )
            for doc_id, idx, url in download_tasks
        ]
        for future in as_completed(futures):
            url, local_path = future.result()
            if local_path:
                url_to_local[url] = local_path
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    return url_to_local


def replace_payload_image_urls(
    payload: Dict[str, Any],
    url_map: Mapping[str, str],
    used_for_tree: bool = False,
) -> Dict[str, Any]:
    """
    Replace image URLs in payload using a {url: local_path} map.

    - used_for_tree = False (default):
        Only replace `image_urls` if it exists (wiki KB).
    - used_for_tree = True:
        Replace `image_urls`, `image_candidates`, and `section_images`
        if they exist (tree KB).
    """

    def _replace_list_in(d: Dict[str, Any], key: str):
        if key in d and isinstance(d[key], list):
            d[key] = [url_map.get(u, u) for u in d[key]]

    def _replace_str_in(d: Dict[str, Any], key: str):
        if key in d and isinstance(d[key], str):
            d[key] = url_map.get(d[key], d[key])

    if not used_for_tree:
        # 原始 wiki KB：只处理 image_urls
        _replace_list_in(payload, "image_urls")
        return payload
    else:
        # tree KB：三个字段独立替换
        root = payload.get("root")
        if isinstance(root, dict):
            _replace_str_in(root, "image_uri")
            _replace_list_in(root, "image_candidates")
            _replace_list_in(root, "image_urls")  # 兼容：如果 tree 里也有

        events = payload.get("events")
        if isinstance(events, list):
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                md = ev.get("metadata")
                if isinstance(md, dict):
                    _replace_list_in(md, "section_images")

        return payload


def save_image_cache(mapping: Mapping[str, str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)


def load_image_cache(path: Path) -> Dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): str(v) for k, v in data.items()}
