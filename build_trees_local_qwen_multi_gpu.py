'''
python build_trees_local_qwen_multi_gpu.py \
  --kb-path ../benchmark/infoseek/wiki_100_dict_v4.json \
  --output-dir examples/tree/ \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 1 \
  --shard-dir examples/kb_shards \
  --llm-model ../ckpts/Qwen-8B \
  --llm-batch-size 4 \
  --max-new-tokens 256 \
  --show-progress \
  --overwrite

CUDA_VISIBLE_DEVICES=0 python build_trees_local_qwen_multi_gpu.py \
  --mode worker \
  --kb-path ../benchmark/infoseek/wiki_100_dict_v4.json \
  --output-dir examples \
  --llm-model ../ckpts/Qwen-8B \
  --llm-batch-size 8 \
  --max-new-tokens 512 \
  --shard-index 0 \
  --num-shards 4 \
  --gpu-id 0 \
  --worker-id 0 \
  --shard-path examples/kb_shards/kb_shard_0.json \
  --tqdm-position 0 \
  --show-progress \
  --overwrite

'''


import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm.auto import tqdm

from unimemrag.memory_forest.memory_forest import build_tree, iter_wiki_dict
from unimemrag.utils.image_cache import (
    download_images_for_kb,
    load_image_cache,
    replace_payload_image_urls,
    save_image_cache,
)


def _parse_gpu_ids(value: str) -> List[int]:
    if not value:
        return []
    parts = re.split(r"[,\s]+", value.strip())
    ids: List[int] = []
    for part in parts:
        if not part:
            continue
        ids.append(int(part))
    return ids


def _resolve_image_cache_paths(
    cache_dir: Optional[str],
    cache_index: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    if not cache_dir and not cache_index:
        return None, None
    if cache_dir:
        cache_dir_path = Path(cache_dir).expanduser()
    else:
        cache_dir_path = Path(cache_index).expanduser().parent  # type: ignore[arg-type]
    cache_index_path = Path(cache_index).expanduser() if cache_index else cache_dir_path / "image_cache_index.json"
    return cache_dir_path, cache_index_path


def _load_kb(path: str) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict-style KB JSON; got {type(data)!r}")
    return data


def _prepare_image_cache(
    kb_path: str,
    *,
    cache_dir: Optional[str],
    cache_index: Optional[str],
    download_images: bool,
    image_workers: int,
) -> Tuple[Optional[Path], Optional[Path]]:
    cache_dir_path, cache_index_path = _resolve_image_cache_paths(cache_dir, cache_index)
    if not cache_dir_path or not cache_index_path:
        return None, None

    if download_images:
        kb = _load_kb(kb_path)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        image_cache = download_images_for_kb(
            kb,
            cache_dir_path,
            max_workers=image_workers,
            resume=True,
            show_progress=True,
        )
        save_image_cache(image_cache, cache_index_path)
    return cache_dir_path, cache_index_path


def _prepare_kb_shards(
    kb_path: str,
    *,
    shard_dir: str,
    shard_count: int,
    limit: Optional[int],
    overwrite: bool,
) -> List[Path]:
    shard_root = Path(shard_dir).expanduser()
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_root / f"kb_shard_{idx}.json" for idx in range(shard_count)]

    if not overwrite and all(path.exists() for path in shard_paths):
        return shard_paths

    kb = _load_kb(kb_path)
    shards: List[Dict[str, Any]] = [dict() for _ in range(shard_count)]
    for idx, (doc_id, payload) in enumerate(iter_wiki_dict(kb)):
        if limit is not None and idx >= limit:
            break
        shard_idx = idx % shard_count
        shards[shard_idx][doc_id] = payload

    for shard_idx, path in enumerate(shard_paths):
        with path.open("w", encoding="utf-8") as fh:
            json.dump(shards[shard_idx], fh, ensure_ascii=False)

    return shard_paths


def _count_shard_total(total: int, shard_index: int, shard_count: int) -> int:
    if total <= 0 or shard_count <= 0:
        return 0
    base = total // shard_count
    extra = total % shard_count
    return base + (1 if shard_index < extra else 0)


def _iter_shard(
    kb: Dict[str, Any],
    *,
    shard_index: int,
    shard_count: int,
    limit: Optional[int],
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for idx, (doc_id, payload) in enumerate(iter_wiki_dict(kb)):
        if limit is not None and idx >= limit:
            break
        if shard_count > 1 and idx % shard_count != shard_index:
            continue
        yield doc_id, payload


def _worker_main(args: argparse.Namespace) -> None:
    if args.gpu_id is not None:
        if args.worker_id is not None:
            gpu_label = f"{args.gpu_id}_w{args.worker_id}"
        else:
            gpu_label = str(args.gpu_id)
    else:
        gpu_label = str(args.shard_index)
    if args.gpu_id is not None and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from unimemrag.llm.QwenText import QwenText

    llm = QwenText(
        args.llm_model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    image_cache: Dict[str, str] = {}
    if args.image_cache_index:
        image_cache = load_image_cache(Path(args.image_cache_index))

    def localize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not image_cache:
            return dict(payload)
        return replace_payload_image_urls(dict(payload), image_cache)

    kb_path = args.shard_path or args.kb_path
    kb = _load_kb(kb_path)
    total_items = len(kb)
    effective_limit = None if args.shard_path else args.limit
    if effective_limit is not None and effective_limit >= 0:
        total_items = min(total_items, effective_limit)
    if args.shard_path:
        shard_total = total_items
    else:
        shard_total = _count_shard_total(total_items, args.shard_index, args.num_shards)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.output_prefix}_gpu{gpu_label}_shard{args.shard_index}.jsonl"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace.")

    if args.shard_path:
        iterator = iter_wiki_dict(kb, limit=effective_limit)
    else:
        iterator = _iter_shard(
            kb,
            shard_index=args.shard_index,
            shard_count=args.num_shards,
            limit=effective_limit,
        )

    if args.show_progress:
        iterator = tqdm(
            iterator,
            total=shard_total if shard_total > 0 else None,
            desc=f"GPU{gpu_label}",
            leave=False,
            position=args.tqdm_position,
        )

    request_kwargs: Dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.temperature is not None:
        request_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        request_kwargs["top_p"] = args.top_p
    if args.do_sample or args.temperature is not None or args.top_p is not None:
        request_kwargs["do_sample"] = True

    with output_path.open("w", encoding="utf-8") as fh:
        for wiki_url, payload in iterator:
            payload = localize_payload(payload)
            tree = build_tree(
                wiki_url,
                payload,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                max_summary_sections=args.max_summary_sections,
                llm=llm,
                llm_model=args.llm_model,
                llm_request_kwargs=request_kwargs,
                llm_workers=1,
                llm_batch_size=args.llm_batch_size,
                show_progress=False,
            )
            fh.write(json.dumps(asdict(tree), ensure_ascii=False) + "\n")


def _build_worker_cmd(
    script_path: Path,
    args: argparse.Namespace,
    *,
    shard_index: int,
    shard_count: int,
    gpu_id: int,
    tqdm_position: int,
    worker_id: int,
    shard_path: Optional[Path],
) -> List[str]:
    cmd = [
        sys.executable,
        script_path.as_posix(),
        "--mode",
        "worker",
        "--kb-path",
        args.kb_path,
        "--output-dir",
        args.output_dir,
        "--output-prefix",
        args.output_prefix,
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(shard_count),
        "--gpu-id",
        str(gpu_id),
        "--worker-id",
        str(worker_id),
        "--tqdm-position",
        str(tqdm_position),
        "--llm-model",
        args.llm_model,
        "--torch-dtype",
        str(args.torch_dtype),
        "--device-map",
        str(args.device_map),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--llm-batch-size",
        str(args.llm_batch_size),
        "--chunk-size",
        str(args.chunk_size),
        "--chunk-overlap",
        str(args.chunk_overlap),
    ]

    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.temperature is not None:
        cmd += ["--temperature", str(args.temperature)]
    if args.top_p is not None:
        cmd += ["--top-p", str(args.top_p)]
    if args.do_sample:
        cmd.append("--do-sample")
    if args.max_summary_sections is not None:
        cmd += ["--max-summary-sections", str(args.max_summary_sections)]
    if shard_path is not None:
        cmd += ["--shard-path", shard_path.as_posix()]
    if args.image_cache_dir:
        cmd += ["--image-cache-dir", args.image_cache_dir]
    if args.image_cache_index:
        cmd += ["--image-cache-index", args.image_cache_index]
    if args.download_images:
        cmd.append("--download-images")
    if args.image_workers is not None:
        cmd += ["--image-workers", str(args.image_workers)]
    if args.show_progress:
        cmd.append("--show-progress")

    return cmd


def _launch_workers(
    args: argparse.Namespace,
    gpu_ids: List[int],
    shard_paths: Optional[List[Path]],
) -> None:
    script_path = Path(__file__).resolve()
    shard_count = len(gpu_ids) * max(1, int(args.workers_per_gpu))
    processes: List[subprocess.Popen] = []

    shard_index = 0
    for gpu_id in gpu_ids:
        for worker_id in range(max(1, int(args.workers_per_gpu))):
            shard_path = shard_paths[shard_index] if shard_paths else None
            cmd = _build_worker_cmd(
                script_path,
                args,
                shard_index=shard_index,
                shard_count=shard_count,
                gpu_id=gpu_id,
                tqdm_position=shard_index,
                worker_id=worker_id,
                shard_path=shard_path,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(cmd, env=env)
            processes.append(proc)
            print(f"Worker {shard_index} -> GPU {gpu_id} (local {worker_id})")
            shard_index += 1

    for proc in processes:
        proc.wait()
        if proc.returncode != 0:
            raise SystemExit(f"Worker failed with exit code {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build memory trees with local Qwen3 across multiple GPUs.")
    parser.add_argument("--kb-path", required=True, help="Path to wiki-style KB JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to write shard outputs.")
    parser.add_argument("--output-prefix", default="trees", help="Prefix for output shard files.")
    parser.add_argument("--mode", choices=["launch", "worker"], default="launch", help="Launch or worker mode.")
    parser.add_argument("--gpu-ids", default="0", help="Comma/space-separated GPU ids.")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Number of workers per GPU.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N items (global index).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output shards if they exist.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for worker mode.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--shard-dir", default=None, help="Pre-shard KB into this directory.")
    parser.add_argument("--overwrite-shards", action="store_true", help="Recreate KB shards if they exist.")
    parser.add_argument("--shard-path", default=None, help="Worker-only: path to a pre-sharded KB JSON.")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id label for worker output naming.")
    parser.add_argument("--worker-id", type=int, default=None, help="Worker id label for worker output naming.")
    parser.add_argument("--tqdm-position", type=int, default=0, help="tqdm position for worker progress bar.")

    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B", help="Local Qwen model name or path.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype (e.g., auto, bfloat16).")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per summary.")
    parser.add_argument("--llm-batch-size", type=int, default=8, help="Batch size for llm.chat_batch.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for local LLM.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p for local LLM sampling.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for local LLM.")

    parser.add_argument("--chunk-size", type=int, default=1024, help="Leaf chunk size.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Leaf chunk overlap.")
    parser.add_argument("--max-summary-sections", type=int, default=None, help="Max leaf texts per summary.")

    parser.add_argument("--image-cache-dir", default=None, help="Directory for cached images.")
    parser.add_argument("--image-cache-index", default=None, help="JSON index for cached images.")
    parser.add_argument("--download-images", action="store_true", help="Download images before building trees.")
    parser.add_argument("--image-workers", type=int, default=64, help="Workers for image download.")

    parser.add_argument("--show-progress", action="store_true", help="Show per-GPU progress bars.")

    args = parser.parse_args()

    if args.mode == "launch":
        gpu_ids = _parse_gpu_ids(args.gpu_ids)
        if not gpu_ids:
            raise ValueError("No GPU ids provided. Use --gpu-ids 0,1,...")

        shard_paths = None
        if args.shard_dir:
            total_workers = len(gpu_ids) * max(1, int(args.workers_per_gpu))
            shard_paths = _prepare_kb_shards(
                args.kb_path,
                shard_dir=args.shard_dir,
                shard_count=total_workers,
                limit=args.limit,
                overwrite=args.overwrite_shards,
            )

        cache_dir_path, cache_index_path = _prepare_image_cache(
            args.kb_path,
            cache_dir=args.image_cache_dir,
            cache_index=args.image_cache_index,
            download_images=args.download_images,
            image_workers=args.image_workers,
        )
        args.image_cache_dir = cache_dir_path.as_posix() if cache_dir_path else None
        args.image_cache_index = cache_index_path.as_posix() if cache_index_path else None

        _launch_workers(args, gpu_ids, shard_paths)
        return

    if args.download_images and not args.image_cache_index:
        cache_dir_path, cache_index_path = _prepare_image_cache(
            args.kb_path,
            cache_dir=args.image_cache_dir,
            cache_index=args.image_cache_index,
            download_images=True,
            image_workers=args.image_workers,
        )
        args.image_cache_dir = cache_dir_path.as_posix() if cache_dir_path else None
        args.image_cache_index = cache_index_path.as_posix() if cache_index_path else None

    _worker_main(args)


if __name__ == "__main__":
    main()
