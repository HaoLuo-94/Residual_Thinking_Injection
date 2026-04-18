from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.05


@dataclass
class ResidualGenerationConfig:
    enabled: bool = False
    alpha: float = 0.001
    layer_start: int = 0
    layer_end: int = -1
    entropy_threshold: float = 0.01
    low_entropy_patience: int = 5
    min_think_steps: int = 8
    max_think_steps: int = 32
    think_end_token: str = "</think>"
    topk: int = 32


class SimpleDeltaBuilder(nn.Module):
    def __init__(self, hidden_size: int, topk: int = 16) -> None:
        super().__init__()
        self.topk = topk
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, logits: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        k = min(self.topk, logits.shape[-1])
        vals, idx = torch.topk(logits, k=k, dim=-1)
        probs = torch.softmax(vals, dim=-1)
        emb = embed_weight[idx]
        soft_emb = (probs.unsqueeze(-1) * emb).sum(dim=1)
        soft_emb = soft_emb.to(embed_weight.dtype)
        return self.proj(soft_emb)


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(dim=-1)


class BatchModelRunner:
    """HuggingFace wrapper with optional residual injection decoding."""

    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        residual_config: Optional[ResidualGenerationConfig] = None,
    ) -> None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        self.residual_config = residual_config or ResidualGenerationConfig()
        self.residual_state = None
        self.delta_builder: Optional[SimpleDeltaBuilder] = None

        if self.residual_config.enabled:
            self._setup_residual_injection()

    def _setup_residual_injection(self) -> None:
        from model.residual import InjectionConfig, InjectionState, apply_qwen3_patch

        cfg = InjectionConfig(
            enabled=self.residual_config.enabled,
            alpha=self.residual_config.alpha,
            layer_start=self.residual_config.layer_start,
            layer_end=self.residual_config.layer_end,
            entropy_threshold=self.residual_config.entropy_threshold,
            low_entropy_patience=self.residual_config.low_entropy_patience,
            min_think_steps=self.residual_config.min_think_steps,
            max_think_steps=self.residual_config.max_think_steps,
            think_end_token=self.residual_config.think_end_token,
            hide_think_output=True,
        )
        self.residual_state = InjectionState()
        apply_qwen3_patch(self.model, cfg, self.residual_state)

        embed_weight = self.model.get_input_embeddings().weight
        self.delta_builder = SimpleDeltaBuilder(
            hidden_size=self.model.config.hidden_size,
            topk=self.residual_config.topk,
        ).to(device=embed_weight.device, dtype=embed_weight.dtype)

    def build_prompts(
        self,
        prompts: Sequence[str],
        prompt_format: str = "chat",
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        if prompt_format in {"completion", "generation"}:
            return list(prompts)

        if prompt_format == "chat":
            if not hasattr(self.tokenizer, "apply_chat_template") or self.tokenizer.chat_template is None:
                return list(prompts)

            rendered_prompts: List[str] = []
            for prompt in prompts:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                rendered_prompts.append(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            return rendered_prompts

        raise ValueError(f"Unsupported prompt_format: {prompt_format}")

    @torch.no_grad()
    def _generate_single_with_residual(
        self,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> str:
        if self.residual_state is None or self.delta_builder is None:
            raise RuntimeError("Residual generation is not initialized.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.input_device)

        eos_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
            eos_token_ids.append(im_end_id)

        state = self.residual_state
        state.delta = None
        state.thinking_phase = True
        state.low_entropy_counter = 0
        state.think_step_count = 0
        state.think_tokens = []
        state.answer_tokens = []

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        device = input_ids.device
        embed_weight = self.model.get_input_embeddings().weight

        past_key_values = None
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        finished = torch.zeros(1, dtype=torch.bool, device=device)
        think_end_ids = self.tokenizer.encode(
            self.residual_config.think_end_token,
            add_special_tokens=False,
        )

        cfg = self.residual_config

        for step in range(generation_config.max_new_tokens):
            outputs = self.model(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            entropy = compute_entropy_from_logits(logits)[0].item()

            if state.thinking_phase:
                state.think_step_count += 1
                
                if entropy < self.residual_config.entropy_threshold:
                    state.low_entropy_counter += 1
                else:
                    state.low_entropy_counter = 0

            next_token = torch.argmax(logits, dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_token_ids[0]),
                next_token,
            )
            token_id = next_token.item()

            if state.thinking_phase:
                state.think_tokens.append(token_id)

                should_end_think = False

                # 条件1：连续低熵达到 patience，但至少已经 think 了 min_think_steps
                if (
                    state.think_step_count >= cfg.min_think_steps
                    and state.low_entropy_counter >= cfg.low_entropy_patience
                ):
                    should_end_think = True

                # 条件2：强制保护，超过 max_think_steps 直接结束
                if state.think_step_count >= cfg.max_think_steps:
                    should_end_think = True

                if should_end_think:
                    state.thinking_phase = False
                    state.low_entropy_counter = 0

                    if len(think_end_ids) > 0:
                        injected_ids = torch.tensor(
                            [think_end_ids],
                            dtype=cur_input_ids.dtype,
                            device=device,
                        )
                        cur_input_ids = injected_ids
                        cur_attention_mask = torch.cat(
                            [
                                cur_attention_mask,
                                torch.ones(
                                    (batch_size, injected_ids.shape[1]),
                                    dtype=cur_attention_mask.dtype,
                                    device=cur_attention_mask.device,
                                ),
                            ],
                            dim=1,
                        )

                        delta = self.delta_builder(logits, embed_weight)
                        state.delta = delta.detach()
                        continue
            else:
                state.answer_tokens.append(token_id)

            for eos_id in eos_token_ids:
                finished |= next_token == eos_id

            state.delta = self.delta_builder(logits, embed_weight).detach()

            if finished.all():
                break

            cur_input_ids = next_token.unsqueeze(-1)
            cur_attention_mask = torch.cat(
                [
                    cur_attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=cur_attention_mask.dtype,
                        device=cur_attention_mask.device,
                    ),
                ],
                dim=1,
            )

            # print(
            #     f"step={step}, entropy={entropy:.4f}, "
            #     f"thinking={state.thinking_phase}, "
            #     f"low_entropy_counter={state.low_entropy_counter}, "
            #     f"think_step_count={state.think_step_count}"
            # )

        if not state.answer_tokens:
            return ""

        generated_ids = torch.tensor(
            [state.answer_tokens],
            dtype=input_ids.dtype,
            device=device,
        )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: Sequence[str],
        generation_config: GenerationConfig,
        prompt_format: str = "chat",
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        rendered_prompts = self.build_prompts(
            prompts=prompts,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )

        if self.residual_config.enabled:
            return [
                self._generate_single_with_residual(prompt, generation_config)
                for prompt in rendered_prompts
            ]

        inputs = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.input_device)

        eos_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
            eos_token_ids.append(im_end_id)

        generation_kwargs = {
            "max_new_tokens": generation_config.max_new_tokens,
            "repetition_penalty": generation_config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_token_ids,
            "do_sample": generation_config.do_sample,
        }

        if generation_config.do_sample:
            generation_kwargs.update(
                {
                    "temperature": generation_config.temperature,
                    "top_p": generation_config.top_p,
                }
            )

        outputs = self.model.generate(**inputs, **generation_kwargs)

        input_seq_len = inputs["input_ids"].shape[1]
        responses: List[str] = []
        for output_ids in outputs:
            generated_ids = output_ids[input_seq_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            responses.append(text)

        return responses