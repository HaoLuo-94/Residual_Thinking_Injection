from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from transformers.models.qwen3 import modeling_qwen3


@dataclass
class InjectionConfig:
    enabled: bool = True
    alpha: float = 0.01
    layer_start: int = 0
    layer_end: int = -1

    # soft thinking control
    entropy_threshold: float = 1.5
    low_entropy_patience: int = 3
    think_end_token: str = "</think>"
    hide_think_output: bool = True
    
    min_think_steps: int = 8
    max_think_steps: int = 32


class InjectionState:
    """
    全局共享 delta（由外部 decode loop 更新）
    """
    def __init__(self):
        self.delta: Optional[torch.Tensor] = None

        self.thinking_phase: bool = True
        self.low_entropy_counter: int = 0
        self.think_step_count: int = 0

        self.think_tokens = []
        self.answer_tokens = []

class PatchedQwen3DecoderLayer(modeling_qwen3.Qwen3DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self._injection_state: Optional[InjectionState] = None
        self._injection_cfg: Optional[InjectionConfig] = None
        self._layer_idx: int = layer_idx
        self._num_layers: int = 1

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # post-attention residual
        hidden_states = residual + hidden_states

        if (
            self._injection_cfg is not None
            and self._injection_cfg.enabled
            and self._injection_state is not None
            and self._injection_state.delta is not None
        ):
            if self._layer_idx >= self._injection_cfg.layer_start:
                if self._injection_cfg.layer_end < 0 or self._layer_idx <= self._injection_cfg.layer_end:
                    delta = self._injection_state.delta.to(
                        hidden_states.device,
                        dtype=hidden_states.dtype,
                    )

                    while delta.dim() < hidden_states.dim():
                        delta = delta.unsqueeze(1)

                    scale = (self._layer_idx + 1) / self._num_layers
                    hidden_states = hidden_states + self._injection_cfg.alpha * scale * delta

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

def apply_qwen3_patch(model, cfg: InjectionConfig, state: InjectionState):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("This model does not expose model.layers, not a supported Qwen3 layout.")

    layers = model.model.layers
    num_layers = len(layers)
    model_config = model.config

    for i in range(num_layers):
        old_layer = layers[i]

        new_layer = PatchedQwen3DecoderLayer(model_config, i)
        new_layer.load_state_dict(old_layer.state_dict())

        new_layer._injection_state = state
        new_layer._injection_cfg = cfg
        new_layer._layer_idx = i
        new_layer._num_layers = num_layers

        old_param = next(old_layer.parameters())
        new_layer = new_layer.to(device=old_param.device, dtype=old_param.dtype)

        layers[i] = new_layer

def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, V]
    return: [B]
    """
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ===== 你的 delta 构造器 =====
    class SimpleDeltaBuilder(nn.Module):
        def __init__(self, hidden_size: int, topk: int = 16):
            super().__init__()
            self.topk = topk
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, logits: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
            # logits: [B, V]
            k = min(self.topk, logits.shape[-1])
            vals, idx = torch.topk(logits, k=k, dim=-1)      # [B, K]
            probs = torch.softmax(vals, dim=-1)              # [B, K]
            emb = embed_weight[idx]                          # [B, K, H]
            soft_emb = (probs.unsqueeze(-1) * emb).sum(dim=1)  # [B, H]
            soft_emb = soft_emb.to(embed_weight.dtype)
            delta = self.proj(soft_emb)
            return delta

    @torch.no_grad()
    def simple_generate(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        state,
        delta_builder,
        cfg,
        max_new_tokens=64,
        eos_token_id=None,
    ):
        device = input_ids.device
        embed_weight = model.get_input_embeddings().weight

        batch_size = input_ids.shape[0]
        if batch_size != 1:
            raise ValueError("Current soft-thinking implementation only supports batch_size=1.")

        past_key_values = None
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if eos_token_id is None:
            eos_ids = []
        elif isinstance(eos_token_id, int):
            eos_ids = [eos_token_id]
        else:
            eos_ids = list(eos_token_id)

        # 初始化状态
        state.delta = None
        state.thinking_phase = True
        state.low_entropy_counter = 0
        state.think_tokens = []
        state.answer_tokens = []

        # think_end token id（如果 tokenizer 能识别）
        think_end_ids = tokenizer.encode(cfg.think_end_token, add_special_tokens=False)

        for step in range(max_new_tokens):
            outputs = model(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :]   # [1, V]
            past_key_values = outputs.past_key_values

            # ===== 1) 计算 entropy =====
            entropy = compute_entropy_from_logits(logits)[0].item()

            if state.thinking_phase:
                if entropy < cfg.entropy_threshold:
                    state.low_entropy_counter += 1
                else:
                    state.low_entropy_counter = 0

            # ===== 2) 采样/贪心取 token =====
            next_token = torch.argmax(logits, dim=-1)  # [1]

            if eos_ids:
                fill_eos = eos_ids[0]
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, fill_eos),
                    next_token,
                )

            token_id = next_token.item()

            # ===== 3) think阶段 / answer阶段 状态切换 =====
            if state.thinking_phase:
                state.think_tokens.append(token_id)

                if state.low_entropy_counter >= cfg.low_entropy_patience:
                    # 触发结束 think
                    state.thinking_phase = False
                    state.low_entropy_counter = 0

                    # 可选：把 </think> 也送进上下文，但不显示
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

                        # 用当前 logits 更新 delta，供下一步用
                        delta = delta_builder(logits, embed_weight)
                        state.delta = delta.detach()

                        continue
            else:
                state.answer_tokens.append(token_id)

            # ===== 4) 更新 finished =====
            for eid in eos_ids:
                finished |= (next_token == eid)

            # ===== 5) 更新 delta（下一步生效） =====
            delta = delta_builder(logits, embed_weight)
            state.delta = delta.detach()

            if finished.all():
                break

            # ===== 6) 下一个 step 只输入一个 token =====
            cur_input_ids = next_token.unsqueeze(-1)
            cur_attention_mask = torch.cat(
                [
                    cur_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=cur_attention_mask.dtype,
                        device=cur_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            # print(
            #     f"step={step}, entropy={entropy:.4f}, "
            #     f"thinking_phase={state.thinking_phase}, "
            #     f"low_entropy_counter={state.low_entropy_counter}"
            # )

        # 只返回 answer_tokens，不显示 think_tokens
        if len(state.answer_tokens) == 0:
            return torch.empty((1, 0), dtype=input_ids.dtype, device=device)

        return torch.tensor(
            [state.answer_tokens],
            dtype=input_ids.dtype,
            device=device,
        )

    # ===== 配置 =====
    model_name = "/kpfs-llm-text/models/Qwen3-4B"
    prompt = (
        "Natalia sold clips to 48 of her friends in April, "
        "and then she sold half as many clips in May. "
        "How many clips did she sell altogether?"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # ===== patch 配置 =====
    cfg = InjectionConfig(
        enabled=True,
        alpha=0.001,
        layer_start=0,
        layer_end=-1,
        entropy_threshold=0.01,
        low_entropy_patience=5,
        think_end_token="</think>",
        hide_think_output=True,
    )
    state = InjectionState()

    # 应用 patch
    apply_qwen3_patch(model, cfg, state)

    # delta builder
    delta_builder = SimpleDeltaBuilder(
        hidden_size=model.config.hidden_size,
        topk=32,
    ).to(device=device, dtype=model.get_input_embeddings().weight.dtype)

    # prompt
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    print("\n=== Input Prompt ===")
    print(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_token_ids.append(im_end_id)

    generated_ids = simple_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        state=state,
        delta_builder=delta_builder,
        cfg=cfg,
        max_new_tokens=1024,
        eos_token_id=eos_token_ids,
    )

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n=== Generated Output ===")
    print(output_text)

    if state.delta is not None:
        print("\n=== Debug ===")
        print("delta shape:", tuple(state.delta.shape))
        print("delta mean norm:", state.delta.norm(dim=-1).mean().item())