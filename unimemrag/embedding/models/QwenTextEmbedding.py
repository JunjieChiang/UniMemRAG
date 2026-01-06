from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _resolve_device(device: Optional[str]) -> str:
    if device == "cuda:auto":
        device = "auto"
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        best_idx = 0
        best_free = -1
        for idx in range(torch.cuda.device_count()):
            try:
                free, _total = torch.cuda.mem_get_info(idx)
            except Exception:
                return "cuda"
            if free > best_free:
                best_free = free
                best_idx = idx
        return f"cuda:{best_idx}"
    return device


class QwenTextEmbedding:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        prompt_name: Optional[str] = "query",
        normalize: bool = True,
    ) -> None:
        resolved_device = _resolve_device(device)
        model_kwargs = dict(model_kwargs or {})
        tokenizer_kwargs = dict(tokenizer_kwargs or {})
        device_arg = None if "device_map" in model_kwargs else resolved_device
        self.model = SentenceTransformer(
            model_name,
            device=device_arg,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.device = device_arg or str(self.model.device)
        self.prompt_name = prompt_name
        self.normalize = normalize

    def embed_texts(self, texts: List[str], *, prompt_name: Optional[str] = None) -> np.ndarray:
        encode_kwargs: Dict[str, Any] = {
            "normalize_embeddings": self.normalize,
            "show_progress_bar": False,
        }
        final_prompt = self.prompt_name if prompt_name is None else prompt_name
        if final_prompt is not None:
            encode_kwargs["prompt_name"] = final_prompt
        embeddings = self.model.encode(texts, **encode_kwargs)
        return np.asarray(embeddings, dtype=np.float32)
