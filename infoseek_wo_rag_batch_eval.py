import json
import re
from typing import Any, Dict, List, Union, Optional, Sequence
from pathlib import Path
from tqdm import tqdm
import torch

from unimemrag.vlm.QwenVL import QwenVL

vlm = QwenVL(
      model_path="../ckpts/Qwen2.5-VL-7B-Instruct",
      torch_dtype="auto",
      device_map="auto",
      # device_map={"": "cuda:0"}
      attn_implementation="flash_attention_2",
      # min_pixels=256*28*28, max_pixels=1280*28*28,  # 如需控制视觉Token范围可解开
)

dataset = []
with open('../benchmark/infoseek/annotations/infoseek_val.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line))

def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _quick_match(prediction: str, gold_answers: Sequence[str]) -> bool:
    if not prediction:
        return False
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    for ans in gold_answers:
        if not ans:
            continue
        gold_norm = normalize_answer(ans)
        if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
            return True
    return False


def match_answer(
    prediction: str,
    gold_answers: Sequence[str],
    *,
    question: Optional[str] = None,
    judge_vlm: Optional[object] = None,
    max_new_tokens: int = 64,
) -> bool:
    # Unused parameters retained for call-site compatibility
    _ = (question, judge_vlm, max_new_tokens)

    if not gold_answers:
        return False

    return _quick_match(prediction, gold_answers)

def build_infoseek_message(
    question: str,
    image_id: Union[str, Path],
    images_root: Optional[Union[str, Path]] = None,
):
    image_stem = Path(image_id).stem
    id_parts = image_stem.split("_")
    if len(id_parts) != 2 or len(id_parts[1]) < 2:
        raise ValueError(f"Unexpected image id format: {image_id!r}")

    bucket = id_parts[1][:2]

    if images_root is None:
        base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        images_root = (base_dir / "../benchmark/oven_eval/image_downloads/oven_images").resolve()
    else:
        images_root = Path(images_root).expanduser().resolve()

    image_file = images_root / bucket / f"{image_stem}.JPEG"
    if not image_file.exists():
        image_file = images_root / bucket / f"{image_stem}.jpg"
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

    return [
        {"role": "system", "content": "You are a helpful assistant. Please answer the user's question without explanation."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_file.as_posix()},
                {"type": "text", "text": question},
            ],
        },
    ]


def evaluate(
    vlm,
    question: str,
    image_id: Union[str, Path],
    images_root: Optional[Union[str, Path]] = None,
    *,
    max_new_tokens: int = 4096,
):
    messages = build_infoseek_message(
        question,
        image_id=image_id,
        images_root=images_root,
    )
    return vlm.chat(messages, max_new_tokens=max_new_tokens)

def run_infoseek_evaluation(
    vlm,
    dataset,
    *,
    limit: Optional[int] = None,
    images_root: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    max_new_tokens: int = 4096,
    batch_size: int = 4,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    records: List[Dict[str, Any]] = []
    skipped = 0
    iterable = dataset if limit is None else dataset[:limit]
    iterator = tqdm(iterable, desc="Evaluating InfoSeek", leave=False) if show_progress else iterable

    pending_examples: List[Dict[str, Any]] = []
    pending_messages: List[List[Dict[str, Any]]] = []

    def _log(message: str) -> None:
        if show_progress and hasattr(iterator, "write"):
            iterator.write(message)
        else:
            print(message)

    def _flush_batch() -> None:
        nonlocal pending_examples, pending_messages, skipped
        if not pending_messages:
            return

        predictions: List[Optional[str]] = []

        try:
            if len(pending_messages) > 1 and hasattr(vlm, "chat_batch"):
                predictions = vlm.chat_batch(pending_messages, max_new_tokens=max_new_tokens)
            else:
                predictions = [vlm.chat(pending_messages[0], max_new_tokens=max_new_tokens)]
        except Exception as exc:
            _log(f"Batch inference failed ({exc}); falling back to sequential execution.")
            predictions = []
            for example, messages in zip(pending_examples, pending_messages):
                try:
                    predictions.append(vlm.chat(messages, max_new_tokens=max_new_tokens))
                except Exception as inner_exc:
                    data_id = example.get("data_id") or example.get("image_id") or "unknown"
                    _log(f"Skipping {data_id}: {inner_exc}")
                    skipped += 1
                    predictions.append(None)

        # predictions = vlm.chat_batch(pending_messages, max_new_tokens=max_new_tokens)

        if len(predictions) != len(pending_examples):
            _log("Mismatch between predictions and examples; skipping incomplete results.")
            skipped += len(pending_examples)
            pending_examples = []
            pending_messages = []
            return

        for example, prediction in zip(pending_examples, predictions):
            if prediction is None:
                continue

            gold_candidates = example.get("answer_eval") or example.get("answer") or []
            correct = match_answer(
                prediction,
                gold_candidates,
                question=example.get("question"),
                judge_vlm=vlm,
            )

            record = dict(example)
            record["model_answer"] = prediction
            record["is_correct"] = bool(correct)
            records.append(record)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pending_examples = []
        pending_messages = []
        

    for example in iterator:
        try:
            messages = build_infoseek_message(
                example["question"],
                image_id=example["image_id"],
                images_root=images_root,
            )
        except FileNotFoundError as exc:
            data_id = example.get("data_id") or example.get("image_id") or "unknown"
            _log(f"Skipping {data_id}: {exc}")
            skipped += 1
            continue

        pending_examples.append(example)
        pending_messages.append(messages)

        if len(pending_messages) >= batch_size:
            _flush_batch()

    _flush_batch()

    total = len(records)
    num_correct = sum(1 for record in records if record["is_correct"])
    accuracy = num_correct / total if total else 0.0

    if total:
        tp = num_correct
        fp = 0
        fn = total - num_correct
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    else:
        f1 = 0.0

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "total": total,
        "correct": num_correct,
        "skipped": skipped,
    }

    if save_path and records:
        output_path = Path(save_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            for record in records:
                json.dump(record, fout, ensure_ascii=False)
                fout.write("\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records, metrics


if __name__ == "__main__":
    results, metrics = run_infoseek_evaluation(
        vlm,
        dataset,
        limit=None,  # Adjust or set to None to evaluate the full validation split
        save_path=Path("infoseek_wo_rag_predictions.jsonl"),
        batch_size=50,
    )
    print(metrics)
