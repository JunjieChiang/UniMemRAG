from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen:
    def __init__(
        self,
        model_name: str,
        *,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def chat(self, messages: List[Dict[str, Any]], *, max_new_tokens: int) -> str:
        return self.chat_batch([messages], max_new_tokens=max_new_tokens)[0]

    def chat_batch(
        self,
        messages_list: List[List[Dict[str, Any]]],
        *,
        max_new_tokens: int,
    ) -> List[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in messages_list
        ]
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)
        attention_mask = model_inputs.get("attention_mask")
        input_ids = model_inputs["input_ids"]

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        outputs: List[str] = []
        # for idx in range(generated_ids.size(0)):
        #     if attention_mask is not None:
        #         input_len = int(attention_mask[idx].sum().item())
        #     else:
        #         input_len = int(input_ids.shape[1])
        #     output_ids = generated_ids[idx][input_len:].tolist()
        #     outputs.append(self.tokenizer.decode(output_ids, skip_special_tokens=True))
        prompt_len = input_ids.shape[1]
        for idx in range(generated_ids.size(0)):
            output_ids = generated_ids[idx][prompt_len:].tolist()
            outputs.append(self.tokenizer.decode(output_ids, skip_special_tokens=True).strip())
        return outputs
