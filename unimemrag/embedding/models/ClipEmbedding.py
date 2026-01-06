from typing import Optional, Union, List, Tuple
import torch
import numpy as np
from PIL import Image
import io, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPProcessor,
    CLIPTokenizer,
)
from transformers.image_utils import PILImageResampling

class ClipEmbedding:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: Optional[str] = None,
        device_map: Optional[Union[str, dict]] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        trust_remote_code: bool = False,
        tokenizer_name_or_path: Optional[str] = None,
        image_processor_name_or_path: Optional[str] = None,
    ):
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        # 1) 用 device_map="auto" 时：不要再 .to(device)
        if device_map is not None:
            # 需要安装 accelerate：pip install accelerate
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,  # 可选：torch.float16 / "auto" 等
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            self.text_device = self._infer_submodule_device(["text_model", "text"])
            self.vision_device = self._infer_submodule_device(["vision_model", "vision", "visual"])

        else:
            # 2) 不用 device_map 时：按你原来的单设备逻辑
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            ).to(self.device)
            self.text_device = self.device
            self.vision_device = self.device

        self.processor = self._load_processor(
            model_name=model_name,
            tokenizer_name_or_path=tokenizer_name_or_path,
            image_processor_name_or_path=image_processor_name_or_path,
        )
        self.model.eval()
        self.max_text_length = self._infer_text_max_length(default=77)

        self.request_headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0 Safari/537.36"
            )
        }
        self.request_timeout: Tuple[int, int] = (5, 20)
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

        self.dim = self._infer_embedding_dim()

    def _infer_submodule_device(self, candidate_attr_names: List[str]) -> torch.device:
        for attr_name in candidate_attr_names:
            submodule = getattr(self.model, attr_name, None)
            if submodule is None:
                continue
            try:
                return next(submodule.parameters()).device
            except StopIteration:
                continue
        return next(self.model.parameters()).device

    def _infer_image_size(self, model_name: str) -> int:
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
        except Exception:
            return 224

        vision_cfg = getattr(cfg, "vision_config", None)
        for obj in [vision_cfg, cfg]:
            if obj is None:
                continue
            for attr in ["image_size", "image_resolution"]:
                val = getattr(obj, attr, None)
                if isinstance(val, int) and val > 0:
                    return val
        return 224

    def _infer_text_max_length(self, *, default: int) -> int:
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        for obj in [text_cfg, cfg]:
            if obj is None:
                continue
            val = getattr(obj, "max_position_embeddings", None)
            if isinstance(val, int) and val > 0:
                return int(val)
        return int(default)

    def _build_default_image_processor(self, image_size: int) -> CLIPImageProcessor:
        return CLIPImageProcessor(
            do_resize=True,
            size={"shortest_edge": image_size},
            resample=PILImageResampling.BICUBIC,
            do_center_crop=True,
            crop_size={"height": image_size, "width": image_size},
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )

    def _load_processor(
        self,
        *,
        model_name: str,
        tokenizer_name_or_path: Optional[str],
        image_processor_name_or_path: Optional[str],
    ) -> CLIPProcessor:
        if tokenizer_name_or_path is None and image_processor_name_or_path is None:
            try:
                return CLIPProcessor.from_pretrained(model_name)
            except OSError:
                pass

        tokenizer_source = tokenizer_name_or_path or model_name
        image_processor_source = image_processor_name_or_path or model_name

        try:
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=self.trust_remote_code,
            )

        try:
            image_processor = CLIPImageProcessor.from_pretrained(image_processor_source)
        except Exception:
            image_size = self._infer_image_size(model_name)
            image_processor = self._build_default_image_processor(image_size=image_size)

        return CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

    def _infer_embedding_dim(self) -> int:
        for attr_name in ["text_projection", "visual_projection"]:
            proj = getattr(self.model, attr_name, None)
            if proj is None:
                continue
            out_features = getattr(proj, "out_features", None)
            if isinstance(out_features, int) and out_features > 0:
                return int(out_features)
            weight = getattr(proj, "weight", None)
            if hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[0])

        cfg = getattr(self.model, "config", None)
        for attr_name in ["projection_dim", "embed_dim", "hidden_size"]:
            val = getattr(cfg, attr_name, None)
            if isinstance(val, int) and val > 0:
                return int(val)

        raise ValueError(
            "Failed to infer embedding dim; please file an issue with your model_name and config, "
            "or extend ClipEmbedding._infer_embedding_dim()."
        )

    @torch.no_grad()
    def embed_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )
        for k in inputs:
            inputs[k] = inputs[k].to(self.text_device)  # 注意：放到 text_device
        feats_t = self._encode_text(inputs)
        feats = feats_t.detach().cpu().numpy()
        if normalize:
            feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
        return feats.astype(np.float32)

    @torch.no_grad()
    def embed_images(self, images: List[Union[Image.Image, str, bytes]], normalize: bool = True) -> np.ndarray:
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
                        response = self.session.get(x, timeout=self.request_timeout, headers=self.request_headers)
                        response.raise_for_status()
                        img = Image.open(io.BytesIO(response.content)).convert("RGB")
                    else:
                        img = Image.open(x).convert("RGB")
                    pil_images.append(img)
                    valid_indices.append(idx)
                except Exception:
                    pass
            elif isinstance(x, (bytes, bytearray)):
                try:
                    img = Image.open(io.BytesIO(x)).convert("RGB")
                    pil_images.append(img)
                    valid_indices.append(idx)
                except Exception:
                    pass
            else:
                raise TypeError(f"Unsupported image type: {type(x)}")

        if not pil_images:
            return result

        inputs = self.processor(images=pil_images, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.vision_device)  # 注意：放到 vision_device
        feats_t = self._encode_image(inputs)
        feats = feats_t.detach().cpu().numpy()

        if normalize:
            feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)

        feats = feats.astype(np.float32)
        for local_idx, original_idx in enumerate(valid_indices):
            result[original_idx] = feats[local_idx]
        return result

    def _encode_text(self, inputs: dict) -> torch.Tensor:
        if hasattr(self.model, "get_text_features"):
            return self.model.get_text_features(**inputs)
        if hasattr(self.model, "encode_text"):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            try:
                return self.model.encode_text(input_ids, attention_mask=attention_mask)
            except TypeError:
                return self.model.encode_text(input_ids)
        raise AttributeError("Model does not implement get_text_features() or encode_text().")

    def _encode_image(self, inputs: dict) -> torch.Tensor:
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(**inputs)
        if hasattr(self.model, "encode_image"):
            pixel_values = inputs.get("pixel_values")
            try:
                return self.model.encode_image(pixel_values)
            except TypeError:
                return self.model.encode_image(pixel_values=pixel_values)
        raise AttributeError("Model does not implement get_image_features() or encode_image().")
