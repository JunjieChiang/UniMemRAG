# qwen_vl_wrapper.py
import io
import torch
import asyncio
from PIL import Image
from typing import List, Dict, Optional, Any, Sequence, Tuple, Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from unimemrag.utils.log_config import setup_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio


Message = Dict[str, Any]  # {"role": "user"/"assistant"/"system", "content":[{"type":"text"/"image"/"video", ...}]}
Messages = List[Message]


logger = setup_logger()


class QwenVL:
    """
    轻量封装的 Qwen VL 推理类
    - 支持 chat / complete / 批量 / 并发
    - 保持与官方 messages 结构一致
    """

    def __init__(
        self,
        model_path: str = "../ckpts/Qwen2.5-VL-3B-Instruct",
        *,
        model_type: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        device_map: Union[str, Dict[str, int]] = "auto",
        attn_implementation: Optional[str] = "flash_attention_2",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        compile: bool = False,
        bf16_fallback_to_fp16: bool = True,
    ) -> None:
        """
        Args:
            model_path: 模型或权重目录/Hub名
            model_type: "qwen2.5" / "qwen3" / None(自动根据路径推断)
            torch_dtype: "auto" / torch.bfloat16 / torch.float16 ...
            device_map: "auto" 或者 显卡映射
            attn_implementation: "flash_attention_2" / "sdpa" / None
            min_pixels, max_pixels: 控制视觉Token数（像素->token自动映射）
            compile: 是否对 model 做 torch.compile（可能提升吞吐，但首次耗时更久）
            bf16_fallback_to_fp16: 若设备不支持 bf16，则回退到 fp16
        """
        self._model_type = self._resolve_model_type(model_path, model_type)

        # 选择 dtype
        if torch_dtype == "auto":
            dtype = "auto"
        else:
            dtype = torch_dtype

        # 加载模型
        model_cls = self._get_model_class(self._model_type)     # get corresponding model class Qwen2.5-VL/Qwen3-VL
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )

        # 加载处理器（可配置像素范围）
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

        # 设备
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 若 bf16 不可用，且用户想用 bf16，则回退 fp16
        if dtype == torch.bfloat16 and bf16_fallback_to_fp16:
            if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
                # 安全回退
                self.model.to(dtype=torch.float16)

        if compile:
            # 仅推理路径图编译；对 generate 友好（依环境而定）
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass  # 环境不支持就忽略

        self.model.eval()

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        normalized = model_type.lower()
        normalized = normalized.replace('_', '').replace('-', '').replace('.', '')
        if normalized.startswith('qwen25'):
            return 'qwen2.5'
        if normalized.startswith('qwen3'):
            return 'qwen3'
        raise ValueError(f"Unsupported model_type {model_type!r}. Expected something like 'qwen2.5' or 'qwen3'.")

    @classmethod
    def _resolve_model_type(cls, model_path: str, model_type: Optional[str]) -> str:
        """Return normalized model type, optionally inferred from path."""
        if model_type:
            return cls._normalize_model_type(model_type)
        path_lower = model_path.lower()
        if 'qwen3' in path_lower:
            return 'qwen3'
        return 'qwen2.5'

    @classmethod
    def _get_model_class(cls, model_type: str):
        normalized = cls._normalize_model_type(model_type)
        mapping = {
            'qwen2.5': Qwen2_5_VLForConditionalGeneration,
            'qwen3': Qwen3VLForConditionalGeneration,
        }
        return mapping[normalized]

    # --------- 工具函数 ---------
    def _apply_template(self, messages: Messages) -> str:
        """将 messages 应用 chat 模板为单轮文本提示（由 processor 管理 BOS/EOS 等）。"""
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _prepare_inputs(
        self, messages: Messages, prompt_text: Optional[str] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        根据 messages 组装多模态输入张量，并移动到正确 device。
        返回：inputs, input_ids（用于截断前缀）
        """
        text = prompt_text if prompt_text is not None else self._apply_template(messages)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs, inputs["input_ids"]

    def _decode(
        self, generated_ids: torch.Tensor, input_ids: torch.Tensor
    ) -> List[str]:
        """
        仅保留新生成部分并解码为字符串列表。
        """
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _normalize_messages(self, messages: Messages) -> Messages:
        normalized: Messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", [])

            if isinstance(content, str):
                normalized_content: List[Dict[str, Any]] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_content = []
                for item in content:
                    if isinstance(item, str):
                        normalized_content.append({"type": "text", "text": item})
                    elif isinstance(item, dict):
                        normalized_content.append(dict(item))
                    else:
                        raise TypeError(f"Unsupported message content item type: {type(item)!r}")
            elif content is None:
                normalized_content = []
            else:
                raise TypeError(f"Unsupported message content container: {type(content)!r}")

            normalized.append({
                "role": role,
                "content": normalized_content,
            })
        return normalized

    def _sanitize_messages(self, messages: Messages) -> Messages:
        normalized = self._normalize_messages(messages)
        sanitized: Messages = []
        for msg in normalized:
            raw_iterable = msg.get("content", [])
            sanitized_content: List[Dict[str, Any]] = []
            for item in raw_iterable:
                entry = dict(item)
                if entry.get("type") == "image":
                    image_value = entry.get("image")
                    if isinstance(image_value, Image.Image):
                        entry["image"] = f"<PIL.Image mode={image_value.mode} size={image_value.size}>"
                    else:
                        entry["image"] = str(image_value)
                elif entry.get("type") == "video":
                    entry["video"] = str(entry.get("video"))
                sanitized_content.append(entry)
            sanitized.append({
                "role": msg.get("role"),
                "content": sanitized_content,
            })
        return sanitized

    def _log_generation_input(
        self,
        *,
        note: str,
        messages: Messages,
        prompt: str,
        gen_kwargs: Dict[str, Any],
    ) -> None:
        logger.info(
            "QwenVL generate | note={note} | gen_kwargs={gen_kwargs}\nPrompt:\n{prompt}\nMessages={messages}",
            note=note,
            gen_kwargs=gen_kwargs,
            prompt=prompt,
            messages=self._sanitize_messages(messages),
        )

    # --------- 对外 API ---------
    @torch.inference_mode()
    def chat(self, messages: Messages, **gen_kwargs) -> str:
        """
        与 Qwen2.5-VL 进行多模态对话。
        Args:
            messages: 官方多模态 messages 结构
            **gen_kwargs: 透传给 model.generate 的参数，比如
                max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True ...
        Returns:
            单条字符串回复
        """
        prompt_text = self._apply_template(messages)
        # self._log_generation_input(
        #     note="chat",
        #     messages=messages,
        #     prompt=prompt_text,
        #     gen_kwargs=dict(gen_kwargs),
        # )
        inputs, in_ids = self._prepare_inputs(messages, prompt_text=prompt_text)
        generated = self.model.generate(**inputs, **gen_kwargs)
        outs = self._decode(generated, in_ids)
        return outs[0]

    @torch.inference_mode()
    def complete(
        self,
        prompt: str,
        *,
        images: Optional[Sequence[Union[str, "PIL.Image.Image"]]] = None,
        videos: Optional[Sequence[str]] = None,
        **gen_kwargs,
    ) -> str:
        """
        便捷补全：把纯文本/图像/视频包装成单轮 user 消息后调用 chat。
        """
        content: List[Dict[str, Any]] = []
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        if videos:
            for v in videos:
                content.append({"type": "video", "video": v})
        content.append({"type": "text", "text": prompt})

        messages: Messages = [{"role": "user", "content": content}]
        return self.chat(messages, **gen_kwargs)

    @torch.inference_mode()
    def chat_batch(self, batch_of_messages: List[Messages], **gen_kwargs) -> List[str]:
        """
        批量推理（单进程顺序，适合小批量/可复用padding）。
        """
        if not batch_of_messages:
            return []

        normalized_messages = [self._normalize_messages(msgs) for msgs in batch_of_messages]
        sanitized_messages = [self._sanitize_messages(msgs) for msgs in normalized_messages]
        # logger.info(
        #     "QwenVL batch generate | batch_size={batch_size} | gen_kwargs={gen_kwargs}\nMessages={messages}",
        #     batch_size=len(batch_of_messages),
        #     gen_kwargs=dict(gen_kwargs),
        #     messages=sanitized_messages,
        # )

        tokenizer = getattr(self.processor, "tokenizer", None)
        original_padding_side: Optional[str] = None
        if tokenizer is not None:
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"

        try:
            processed = self.processor.apply_chat_template(
                normalized_messages,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            if tokenizer is not None and original_padding_side is not None:
                tokenizer.padding_side = original_padding_side

        processed = processed.to(self.device)
        in_ids = processed["input_ids"]

        generated = self.model.generate(**processed, **gen_kwargs)
        trimmed = [
            out_ids[len(in_ids_row):]
            for in_ids_row, out_ids in zip(in_ids, generated)
        ]
        outs = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outs

    @torch.inference_mode()
    def chat_concurrent(
        self,
        batch_of_messages: List[Messages],
        *,
        max_workers: int = 4,
        **gen_kwargs,
    ) -> List[str]:
        """
        线程并发推理：对多路会话各自单独 forward。适合 I/O 准备多、显卡能并行排队的场景。
        注意：同一张卡上并发大多是“排队+重叠”，吞吐是否提升取决于负载。
        """
        def _single(msgs: Messages) -> str:
            return self.chat(msgs, **gen_kwargs)

        results: List[Optional[str]] = [None] * len(batch_of_messages)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_single, m): i for i, m in enumerate(batch_of_messages)}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        # 类型忽略：此时不存在 None
        return [r for r in results if r is not None]

        
    
# # ---------------- 使用示例 ----------------
# if __name__ == "__main__":
#     qwen = QwenVL(
#         model_path="../ckpts/Qwen2.5-VL-3B-Instruct",
#         torch_dtype=torch.bfloat16,           # 自动也可： "auto"
#         device_map="auto",
#         attn_implementation="flash_attention_2",
#         # min_pixels=256*28*28, max_pixels=1280*28*28,  # 如需控制视觉Token范围可解开
#     )

#     # 单轮 chat（与你给的示例一致）
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": "test/1/image_1.jpeg"},
#                 {"type": "text", "text": "How many children does this man have?"},
#             ],
#         }
#     ]
#     out = qwen.chat(messages, max_new_tokens=128)
#     print("chat:", out)

#     # 便捷 complete（文本+可选多图/视频）
#     out2 = qwen.complete(
#         "Describe the scene in one short sentence.",
#         images=["test/1/image_1.jpeg"],
#         max_new_tokens=64,
#     )
#     print("complete:", out2)

#     # 批量
#     batch = [messages, messages]
#     outs = qwen.chat_batch(batch, max_new_tokens=64)
#     print("batch:", outs)

#     # 并发
#     outs_conc = qwen.chat_concurrent(batch, max_workers=2, max_new_tokens=64)
#     print("concurrent:", outs_conc)
