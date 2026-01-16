#!/usr/bin/env python3
"""
Utility script for preparing a compact InfoSeek knowledge base split.

The script performs three key tasks:
1. Samples a fixed number of entries from the 6M InfoSeek Wikipedia dump while
   guaranteeing coverage of all entities referenced by the validation split.
   When multiple rows exist for the same entity, versions with
   `wikipedia_image_url` are preferred whenever available.
2. Optionally downloads Wikipedia images referenced by the sampled entries so
   the resulting KB can be used for multi-modal RAG experiments.
3. Emits a JSONL file compatible with UniMemRAG indexing utilities. The output
   rows contain both the original text fields and, when available, a
   `local_image_path` value pointing to the downloaded image.

Example usage (text-only KB):
    python examples/prepare_infoseek_kb.py \\
        --kb benchmark/infoseek/wiki_text/Wiki6M_ver_1_0.jsonl \\
        --val-mapping benchmark/infoseek/kb_mapping/infoseek_val_withkb.jsonl \\
        --output benchmark/infoseek/wiki_text/infoseek_kb_100ƒk.jsonl

Example usage (multi-modal KB with image downloads):
    python examples/prepare_infoseek_kb.py \\
        --kb benchmark/infoseek/wiki_text/Wiki6M_ver_1_0.jsonl \\
        --val-mapping benchmark/infoseek/kb_mapping/infoseek_val_withkb.jsonl \\
        --output benchmark/infoseek/wiki_text/infoseek_kb_100k_mm.jsonl \\
        --image-dir benchmark/infoseek/images_100k \\
        --download-images --max-workers 16

Example usage (E-VQA image downloads):
    python examples/prepare_infoseek_kb.py \
        --evqa-kb /home/mobuser/jjj/benchmark/encyclopedic_vqa/encyclopedic_kb_wiki.json \
        --image-dir benchmark/encyclopedic_vqa/images_kb \
        --download-images --max-workers 16

Optional (write E-VQA KB with local image paths):
    python examples/prepare_infoseek_kb.py \\
        --evqa-kb /home/mobuser/jjj/benchmark/encyclopedic_vqa/encyclopedic_kb_wiki.json \\
        --image-dir benchmark/encyclopedic_vqa/images \\
        --download-images \\
        --evqa-output benchmark/encyclopedic_vqa/encyclopedic_kb_wiki_local.json
"""

from __future__ import annotations
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import shutil
import subprocess
import argparse
import hashlib
import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from socket import timeout as SocketTimeout

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress bar
    tqdm = None

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None


def _resolve_path(path: str) -> Path:
    """Resolve relative paths with respect to the repository root when possible."""

    candidate = Path(path).expanduser()
    if candidate.exists() or candidate.parent.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / path).expanduser()


def read_jsonl(path: Path) -> Iterator[Dict]:
    """Yield JSONL records without loading the entire file into memory."""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def iter_json_object_items(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[Tuple[str, object]]:
    """Stream key/value pairs from a large JSON object without loading it fully."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buf = ""
        pos = 0
        eof = False

        def fill() -> None:
            nonlocal buf, eof
            if eof:
                return
            chunk = handle.read(chunk_size)
            if chunk:
                buf += chunk
            else:
                eof = True

        fill()
        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos < len(buf):
                if buf[pos] != "{":
                    raise ValueError(f"Expected JSON object in {path}")
                pos += 1
                break
            if eof:
                raise ValueError(f"Empty JSON file: {path}")
            fill()

        while True:
            while True:
                while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ","):
                    pos += 1
                if pos < len(buf):
                    break
                if eof:
                    raise ValueError(f"Unexpected end of JSON object: {path}")
                fill()

            if buf[pos] == "}":
                break

            while True:
                try:
                    key, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
                    continue
                if not isinstance(key, str):
                    raise ValueError(f"Expected string key in {path}")
                pos = end
                break

            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf):
                    break
                if eof:
                    raise ValueError(f"Unexpected end after key in {path}")
                fill()

            if buf[pos] != ":":
                raise ValueError(f"Expected ':' after key in {path}")
            pos += 1

            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf):
                    break
                if eof:
                    raise ValueError(f"Unexpected end after ':' in {path}")
                fill()

            while True:
                try:
                    value, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
                    continue
                pos = end
                break

            yield key, value

            if pos > chunk_size:
                buf = buf[pos:]
                pos = 0
                if not eof and len(buf) < chunk_size:
                    fill()


def load_required_entities(mapping_path: Path) -> Dict[str, List[str]]:
    """Return a mapping from entity id to the validation data ids that require it."""

    coverage: Dict[str, List[str]] = {}
    for row in read_jsonl(mapping_path):
        entity_id = row.get("entity_id")
        data_id = row.get("data_id")
        if not entity_id:
            raise ValueError(f"Missing entity_id in mapping row: {row}")
        coverage.setdefault(entity_id, []).append(data_id or "")
    return coverage


def _pick_image_extension(url: str) -> str:
    """Infer an image extension from the URL, defaulting to .jpg when ambiguous."""

    allowed = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in allowed:
        return ".jpg" if suffix == ".jpeg" else suffix
    # Try format query parameter (occasionally used in Wikimedia URLs)
    params = parse_qs(parsed.query.lower())
    fmt = params.get("format", [])
    if fmt:
        potential = f".{fmt[0]}"
        if potential in allowed:
            return potential
    return ".jpg"


def _filename_from_url(url: str) -> str:
    """Generate a stable filename for a URL using a hash and inferred extension."""

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return f"{digest}{_pick_image_extension(url)}"


def _relative_to(path: Path, root: Path) -> str:
    """Return a POSIX-style relative path when possible, otherwise the absolute path."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _has_image(record: Dict) -> bool:
    """Check whether a KB record provides a downloadable image URL."""

    url = record.get("wikipedia_image_url")
    return bool(url and isinstance(url, str) and url.strip())


def sample_kb_entries(
    kb_path: Path,
    required_entities: Set[str],
    sample_size: int,
    *,
    seed: Optional[int] = None,
    annotate_source: bool = False,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Sample KB entries guaranteeing coverage for the provided entity ids.

    Returns:
        sampled_entries: The combined list of required and randomly sampled entries.
        stats: Sampling statistics for logging.
    """

    rng = random.Random(seed)
    required_total = len(required_entities)
    if sample_size < required_total:
        raise ValueError(
            f"Sample size {sample_size} is smaller than the number of required entities {required_total}"
        )

    required_records: Dict[str, Dict] = {}
    reservoir_target = sample_size - required_total
    reservoir: List[Dict] = []
    candidates_seen = 0
    reservoir_entities: Dict[str, int] = {}

    for record in read_jsonl(kb_path):
        entity_id = record.get("wikidata_id")
        record_has_image = _has_image(record)

        if entity_id in required_entities and entity_id not in required_records:
            required_records[entity_id] = record
            continue
        elif entity_id in required_entities:
            # Prefer entries with images for required entities when possible.
            existing = required_records[entity_id]
            if record_has_image and not _has_image(existing):
                required_records[entity_id] = record
            continue

        if reservoir_target <= 0:
            continue

        if entity_id and entity_id in reservoir_entities:
            idx = reservoir_entities[entity_id]
            existing = reservoir[idx]
            if record_has_image and not _has_image(existing):
                reservoir[idx] = record
            continue

        candidates_seen += 1
        if len(reservoir) < reservoir_target:
            reservoir.append(record)
            if entity_id:
                reservoir_entities[entity_id] = len(reservoir) - 1
        else:
            idx = rng.randint(0, candidates_seen - 1)
            if idx < reservoir_target:
                displaced = reservoir[idx]
                old_id = displaced.get("wikidata_id")
                if old_id:
                    reservoir_entities.pop(old_id, None)
                reservoir[idx] = record
                if entity_id:
                    reservoir_entities[entity_id] = idx

    missing = required_entities.difference(required_records.keys())
    if missing:
        raise RuntimeError(
            f"Failed to locate {len(missing)} required entities in KB: {sorted(list(missing))[:5]} ..."
        )

    sampled = list(required_records.values()) + reservoir
    if len(sampled) < sample_size:
        raise RuntimeError(
            f"Unable to reach the requested sample size {sample_size}; only {len(sampled)} unique entries collected."
        )
    if annotate_source:
        for rec in sampled:
            entity_id = rec.get("wikidata_id")
            rec["sampled_from"] = "validation_entity" if entity_id in required_entities else "random_pool"
    rng.shuffle(sampled)

    stats = {
        "total_sampled": len(sampled),
        "required_entities": required_total,
        "random_sampled": len(reservoir),
        "random_candidates_seen": candidates_seen,
    }
    return sampled, stats


def _download_single_image(
    url: str,
    destination: Path,
    *,
    timeout: int,
    user_agent: str,
    retries: int,
    retry_delay: float,
    chunk_size: int,  # kept for API compatibility; unused by wget
) -> Tuple[bool, Optional[str]]:
    """Download a single image using wget, returning (success, error_message)."""

    wget_path = shutil.which("wget")
    if not wget_path:
        return False, "wget not found in PATH. Please install wget or use requests/urlopen mode."

    destination.parent.mkdir(parents=True, exist_ok=True)

    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        tmp_path = destination.with_suffix(destination.suffix + ".part")

        # wget options:
        # -q: quiet (you can remove to debug)
        # --timeout: network timeout per operation (seconds)
        # --tries=1: we handle retries ourselves
        # -U: User-Agent
        # -O: output file
        cmd = [
            wget_path,
            "-q",
            "--timeout",
            str(timeout),
            "--tries",
            "1",
            "-U",
            user_agent,
            "-O",
            str(tmp_path),
            url,
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                tmp_path.replace(destination)
                return True, None

            # Non-zero exit: capture stderr for debugging
            error_msg = (proc.stderr or proc.stdout or "").strip()
            if not error_msg:
                error_msg = f"wget failed with return code {proc.returncode}"
        except Exception as exc:  # pragma: no cover
            error_msg = f"Unexpected error running wget: {exc}"
        finally:
            # Clean partial file if exists and download failed
            if tmp_path.exists() and not destination.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        if attempt == attempts:
            return False, error_msg

        time.sleep(retry_delay * attempt)

    return False, "Unknown download error"



def iter_evqa_image_urls(kb_path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """Yield image URLs from the E-VQA encyclopedic KB JSON."""

    for _, payload in iter_json_object_items(kb_path, chunk_size=chunk_size):
        if not isinstance(payload, dict):
            continue
        urls = payload.get("image_urls")
        if not isinstance(urls, list):
            continue
        for url in urls:
            if isinstance(url, str):
                stripped = url.strip()
                if stripped:
                    yield stripped


def count_evqa_image_urls(kb_path: Path, *, chunk_size: int = 1024 * 1024) -> int:
    """Count image URLs in the E-VQA encyclopedic KB JSON."""

    total = 0
    for _ in iter_evqa_image_urls(kb_path, chunk_size=chunk_size):
        total += 1
    return total


def download_images_for_urls(
    urls: Iterable[str],
    *,
    image_dir: Path,
    max_workers: int = 8,
    timeout: int = 10,
    skip_existing: bool = True,
    user_agent: str = "UniMemRAG-ImageFetcher/1.0",
    retries: int = 3,
    retry_delay: float = 1.5,
    chunk_size: int = 64 * 1024,
    show_progress: bool = True,
    progress_desc: str = "Downloading images",
    total: Optional[int] = None,
) -> Dict[str, int]:
    """Download images from an iterable of URLs."""

    image_dir.mkdir(parents=True, exist_ok=True)
    total_urls = 0
    queued = 0
    skipped_existing = 0
    duplicates = 0
    successes = 0
    failures = 0
    seen: Set[str] = set()

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=total, desc=progress_desc, leave=False)

    def handle_future(future) -> None:
        nonlocal successes, failures
        success, _error = future.result()
        if success:
            successes += 1
        else:
            failures += 1
        if progress is not None:
            progress.update(1)

    futures = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for url in urls:
            total_urls += 1
            filename = _filename_from_url(url)
            if filename in seen:
                duplicates += 1
                if progress is not None:
                    progress.update(1)
                continue
            seen.add(filename)

            destination = image_dir / filename
            if skip_existing and destination.exists():
                skipped_existing += 1
                if progress is not None:
                    progress.update(1)
                continue

            future = executor.submit(
                _download_single_image,
                url,
                destination,
                timeout=timeout,
                user_agent=user_agent,
                retries=retries,
                retry_delay=retry_delay,
                chunk_size=chunk_size,
            )
            futures.add(future)
            queued += 1

            if len(futures) >= max_workers * 4:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for finished in done:
                    handle_future(finished)

        if futures:
            done, _ = wait(futures)
            for finished in done:
                handle_future(finished)

    if progress is not None:
        progress.close()

    return {
        "urls": total_urls,
        "unique_urls": len(seen),
        "queued": queued,
        "downloaded": successes,
        "failed": failures,
        "skipped_existing": skipped_existing,
        "duplicates": duplicates,
    }


def download_images_for_entries(
    entries: Sequence[Dict],
    *,
    image_dir: Path,
    max_workers: int = 8,
    timeout: int = 10,
    skip_existing: bool = True,
    user_agent: str = "UniMemRAG-ImageFetcher/1.0",
    retries: int = 3,
    retry_delay: float = 1.5,
    chunk_size: int = 64 * 1024,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    Download Wikipedia images referenced by the sampled entries.

    The function updates each entry in-place with a `local_image_path` when the
    download succeeds or when a matching file already exists on disk.
    """

    image_dir.mkdir(parents=True, exist_ok=True)
    total_with_url = 0
    queued = []

    for entry in entries:
        url = entry.get("wikipedia_image_url")
        if not url:
            continue
        total_with_url += 1
        ext = _pick_image_extension(url)
        filename = f"{entry.get('wikidata_id', 'unknown')}{ext}"
        destination = image_dir / filename

        if skip_existing and destination.exists():
            entry["local_image_path"] = _relative_to(destination, image_dir)
            continue

        queued.append((entry, url, destination))

    if not queued:
        return {
            "urls": total_with_url,
            "queued": 0,
            "downloaded": 0,
            "failed": 0,
            "skipped_existing": total_with_url,
        }

    successes = 0
    failures = 0
    skipped_existing = total_with_url - len(queued)
    future_to_payload = {}

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(queued), desc="Downloading images", leave=False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for entry, url, destination in queued:
            future = executor.submit(
                _download_single_image,
                url,
                destination,
                timeout=timeout,
                user_agent=user_agent,
                retries=retries,
                retry_delay=retry_delay,
                chunk_size=chunk_size,
            )
            future_to_payload[future] = (entry, url, destination)

        for future in as_completed(future_to_payload):
            entry, url, destination = future_to_payload[future]
            success, error = future.result()
            if success:
                entry["local_image_path"] = _relative_to(destination, image_dir)
                successes += 1
            else:
                failures += 1
                entry.setdefault("image_download_error", error)
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    return {
        "urls": total_with_url,
        "queued": len(queued),
        "downloaded": successes,
        "failed": failures,
        "skipped_existing": skipped_existing,
    }


def annotate_existing_images(entries: Sequence[Dict], *, image_dir: Optional[Path]) -> None:
    """Fill in `local_image_path` for entries when the corresponding file already exists."""

    if not image_dir:
        return
    if not image_dir.exists():
        return

    for entry in entries:
        url = entry.get("wikipedia_image_url")
        if not url:
            continue
        ext = _pick_image_extension(url)
        filename = f"{entry.get('wikidata_id', 'unknown')}{ext}"
        destination = image_dir / filename
        if destination.exists():
            entry["local_image_path"] = _relative_to(destination, image_dir)


def write_jsonl(entries: Iterable[Dict], output_path: Path) -> None:
    """Write the sampled entries to a JSONL file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            # Avoid leaking helper keys used during processing.
            entry = {k: v for k, v in entry.items() if not k.startswith("_")}
            handle.write(json.dumps(entry, ensure_ascii=True))
            handle.write("\n")


def write_evqa_kb_with_local_paths(
    kb_path: Path,
    output_path: Path,
    *,
    image_dir: Path,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Write an E-VQA KB JSON with local image paths added."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("{\n")
        first = True
        for key, payload in iter_json_object_items(kb_path, chunk_size=chunk_size):
            if isinstance(payload, dict):
                urls = payload.get("image_urls")
                if isinstance(urls, list):
                    local_paths = []
                    for url in urls:
                        if isinstance(url, str):
                            stripped = url.strip()
                        else:
                            stripped = ""
                        if stripped:
                            destination = image_dir / _filename_from_url(stripped)
                            if destination.exists():
                                local_paths.append(_relative_to(destination, image_dir))
                            else:
                                local_paths.append(None)
                        else:
                            local_paths.append(None)
                    payload = dict(payload)
                    payload["local_image_paths"] = local_paths

            if not first:
                handle.write(",\n")
            handle.write(json.dumps(key, ensure_ascii=True))
            handle.write(": ")
            handle.write(json.dumps(payload, ensure_ascii=True))
            first = False
        handle.write("\n}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample InfoSeek KB entries or download images for E-VQA KBs."
    )
    parser.add_argument(
        "--kb",
        help="Path to Wiki6M_ver_1_0.jsonl (required unless --evqa-kb is provided)",
    )
    parser.add_argument(
        "--val-mapping",
        help="Path to infoseek_val_withkb.jsonl containing (data_id, entity_id)",
    )
    parser.add_argument("--output", help="Destination JSONL file (InfoSeek mode)")
    parser.add_argument(
        "--evqa-kb",
        help="Path to encyclopedic_kb_wiki.json to download E-VQA images",
    )
    parser.add_argument(
        "--evqa-output",
        help="Optional output JSON path to add local_image_paths for E-VQA entries",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of KB entries to sample (default: 100000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--annotate-source",
        action="store_true",
        help="Add a `sampled_from` field describing whether an entry covers validation entities.",
    )
    parser.add_argument(
        "--image-dir",
        help="Directory where downloaded images will be stored. Required for --download-images or --evqa-kb.",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download images for InfoSeek or E-VQA entries (requires --image-dir).",
    )
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel workers for image download.")
    parser.add_argument("--timeout", type=int, default=10, help="Per-request timeout (seconds) for image download.")
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="Number of retry attempts per image download (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.5,
        help="Initial backoff delay in seconds between download retries (default: 1.5).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64 * 1024,
        help="Image download chunk size in bytes (default: 65536).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download images even if a file already exists at the destination path.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar during image downloads.",
    )
    parser.add_argument(
        "--download-backend",
        choices=["wget", "requests"],
        default="wget",
        help="Image download backend (default: wget).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = _resolve_path(args.image_dir) if args.image_dir else None

    if args.evqa_kb:
        if args.kb or args.val_mapping or args.output:
            raise ValueError("Use either InfoSeek inputs or --evqa-kb, not both.")
        if args.evqa_output and not image_dir:
            raise ValueError("--evqa-output requires --image-dir")
        if args.download_images and image_dir is None:
            raise ValueError("--download-images requires --image-dir")
        if not args.download_images and not args.evqa_output:
            raise ValueError("E-VQA mode requires --download-images and/or --evqa-output.")

        evqa_kb_path = _resolve_path(args.evqa_kb)
        if not evqa_kb_path.is_file():
            raise FileNotFoundError(f"E-VQA KB JSON not found: {evqa_kb_path}")
        if image_dir is None:
            raise ValueError("--evqa-kb requires --image-dir")

        if args.download_images:
            total_urls = None
            if not args.no_progress and tqdm is not None:
                print("Counting E-VQA image URLs for progress bar...")
                total_urls = count_evqa_image_urls(evqa_kb_path)
            download_stats = download_images_for_urls(
                iter_evqa_image_urls(evqa_kb_path),
                image_dir=image_dir,
                max_workers=args.max_workers,
                timeout=args.timeout,
                skip_existing=not args.no_skip_existing,
                retries=max(0, args.download_retries),
                retry_delay=max(0.1, args.retry_delay),
                chunk_size=max(1024, args.chunk_size),
                show_progress=not args.no_progress,
                progress_desc="Downloading E-VQA images",
                total=total_urls,
            )
            print(
                "E-VQA image download stats: "
                f"urls={download_stats['urls']} unique={download_stats['unique_urls']} "
                f"queued={download_stats['queued']} downloaded={download_stats['downloaded']} "
                f"failed={download_stats['failed']} skipped_existing={download_stats['skipped_existing']} "
                f"duplicates={download_stats['duplicates']}"
            )

        if args.evqa_output:
            evqa_output_path = _resolve_path(args.evqa_output)
            write_evqa_kb_with_local_paths(
                evqa_kb_path,
                evqa_output_path,
                image_dir=image_dir,
            )
            print(f"Wrote E-VQA KB with local image paths to {evqa_output_path}")

        if image_dir:
            print(f"Image directory: {image_dir}")
        return

    if args.evqa_output:
        raise ValueError("--evqa-output requires --evqa-kb")

    if not args.kb or not args.val_mapping or not args.output:
        raise ValueError("--kb, --val-mapping, and --output are required for InfoSeek mode.")
    kb_path = _resolve_path(args.kb)
    mapping_path = _resolve_path(args.val_mapping)
    output_path = _resolve_path(args.output)

    if not kb_path.is_file():
        raise FileNotFoundError(f"Knowledge base JSONL not found: {kb_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Validation mapping JSONL not found: {mapping_path}")
    if args.download_images and image_dir is None:
        raise ValueError("--download-images requires --image-dir")

    entity_coverage = load_required_entities(mapping_path)
    required_entities = set(entity_coverage.keys())
    sampled_entries, stats = sample_kb_entries(
        kb_path,
        required_entities,
        args.sample_size,
        seed=args.seed,
        annotate_source=args.annotate_source,
    )

    print(
        f"Sampled {stats['total_sampled']} entries "
        f"(required={stats['required_entities']} random={stats['random_sampled']}) "
        f"from {kb_path}"
    )

    if args.download_images and image_dir is not None:
        download_stats = download_images_for_entries(
            sampled_entries,
            image_dir=image_dir,
            max_workers=args.max_workers,
            timeout=args.timeout,
            skip_existing=not args.no_skip_existing,
            retries=max(0, args.download_retries),
            retry_delay=max(0.1, args.retry_delay),
            chunk_size=max(1024, args.chunk_size),
            show_progress=not args.no_progress,
        )
        print(
            "Image download stats: "
            f"urls={download_stats['urls']} queued={download_stats['queued']} "
            f"downloaded={download_stats['downloaded']} failed={download_stats['failed']} "
            f"skipped_existing={download_stats['skipped_existing']}"
        )
    else:
        annotate_existing_images(sampled_entries, image_dir=image_dir)

    write_jsonl(sampled_entries, output_path)
    print(f"Wrote sampled KB to {output_path}")
    if image_dir:
        print(f"Image directory: {image_dir}")


if __name__ == "__main__":
    main()
