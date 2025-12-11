import io
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from transformers import CLIPModel, CLIPProcessor
from urllib3.util.retry import Retry


class ClipEmbedding:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.request_headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0 Safari/537.36"
            )
        }
        self.request_timeout: Tuple[int, int] = (5, 20)  # (connect timeout, read timeout)
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"GET"},
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # dynamically access the dim
        self.dim = int(self.model.text_projection.out_features)

    @torch.no_grad()
    def embed_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        inputs = self.processor(text=texts, images=None, return_tensors="pt", padding=True, truncation=True)
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        feats = self.model.get_text_features(**inputs).detach().cpu().numpy()
        if normalize:
            feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
        return feats.astype(np.float32)

    @torch.no_grad()
    def embed_images(
        self,
        images: List[Union[Image.Image, str, bytes]],
        normalize: bool = True,
    ) -> np.ndarray:
        result = np.zeros((len(images), self.dim), dtype=np.float32)
        pil_images: List[Image.Image] = []
        valid_indices: List[int] = []

        for idx, x in enumerate(images):
            if isinstance(x, Image.Image):
                pil_images.append(x.convert("RGB"))
                valid_indices.append(idx)
            elif isinstance(x, str):
                try:
                    if x.startswith("http://") or x.startswith("https://"):
                        response = self.session.get(
                            x,
                            timeout=self.request_timeout,
                            headers=self.request_headers,
                        )
                        response.raise_for_status()
                        img = Image.open(io.BytesIO(response.content)).convert("RGB")
                    else:
                        img = Image.open(x).convert("RGB")
                    pil_images.append(img)
                    valid_indices.append(idx)
                except Exception as exc:
                    # raise f"[ClipEmbedding] Skipping image {x}: {exc}"
                    pass
            elif isinstance(x, (bytes, bytearray)):
                try:
                    img = Image.open(io.BytesIO(x)).convert("RGB")
                    pil_images.append(img)
                    valid_indices.append(idx)
                except Exception as exc:
                    print(f"[ClipEmbedding] Skipping in-memory image bytes: {exc}")
            else:
                raise TypeError(f"Unsupported image type: {type(x)}")

        if not pil_images:
            return result

        inputs = self.processor(text=None, images=pil_images, return_tensors="pt", padding=True)
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        feats = self.model.get_image_features(**inputs).detach().cpu().numpy()
        if normalize:
            feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
        feats = feats.astype(np.float32)
        for local_idx, original_idx in enumerate(valid_indices):
            result[original_idx] = feats[local_idx]
        return result
