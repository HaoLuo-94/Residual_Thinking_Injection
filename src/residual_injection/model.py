"""Custom models: explicit residual + αΔ injection, and Δ updates in compute_logits."""
from __future__ import annotations

import torch.nn as nn

import types
from typing import Callable

from .config import CONFIG
from .runtime import InjectionRuntime


# ----------------------------------------------------------------------------
# Bind "explicit residual + injection" onto constructed model / layer instances
# (relies on stable attribute names; more version-tolerant than overriding __init__)
# ----------------------------------------------------------------------------
def _make_layer_forward(runtime: InjectionRuntime):
    def layer_forward(self, positions, hidden_states, inject):
        # Non-fused explicit pre-norm residual, matching the formula exactly:
        #   h_res    = h_prev + Attn(LN(h_prev))
        #   h_inject = h_res + αΔ          <-- injection
        #   h_next   = h_inject + MLP(LN(h_inject))
        h_prev = hidden_states
        normed = self.input_layernorm(h_prev)
        attn_out = self.self_attn(positions, normed)
        h_res = h_prev + attn_out
        # Inject only on selected layers (inject already includes α; _ri_do_inject set by _install_injection)
        h_inject = h_res + inject if getattr(self, "_ri_do_inject", True) else h_res
        normed2 = self.post_attention_layernorm(h_inject)
        mlp_out = self.mlp(normed2)
        return h_inject + mlp_out
    return layer_forward


def _make_glm4_layer_forward(runtime: InjectionRuntime):
    # GLM-4 (0414 series, Glm4DecoderLayer) adds an extra sandwich norm on each sub-block output
    # vs Llama/Qwen3:
    #   h_res    = h_prev + post_self_attn_layernorm( Attn(LN(h_prev)) )
    #   h_inject = h_res + αΔ                                            <-- injection
    #   h_next   = h_inject + post_mlp_layernorm( MLP(post_attention_layernorm(h_inject)) )
    def layer_forward(self, positions, hidden_states, inject):
        h_prev = hidden_states
        normed = self.input_layernorm(h_prev)
        attn_out = self.self_attn(positions, normed)
        attn_out = self._ri_post_attn_norm(attn_out)        # sandwich (identity if missing)
        h_res = h_prev + attn_out
        h_inject = h_res + inject if getattr(self, "_ri_do_inject", True) else h_res
        normed2 = self.post_attention_layernorm(h_inject)
        mlp_out = self.mlp(normed2)
        mlp_out = self._ri_post_mlp_norm(mlp_out)            # sandwich (identity if missing)
        return h_inject + mlp_out
    return layer_forward


def _make_glm4_per_layer_setup():
    # Sandwich norm may be absent on some GLM-4 variants; fall back to identity and probe on first layer.
    state = {"printed": False}

    def setup(layer):
        a = getattr(layer, "post_self_attn_layernorm", None)
        m = getattr(layer, "post_mlp_layernorm", None)
        layer._ri_post_attn_norm = a if a is not None else nn.Identity()
        layer._ri_post_mlp_norm = m if m is not None else nn.Identity()
        if not state["printed"]:
            print(f"[residual] GLM4 sandwich norm: "
                  f"post_self_attn={a is not None}, post_mlp={m is not None}")
            state["printed"] = True

    return setup


def _make_model_forward(runtime: InjectionRuntime):
    def model_forward(self, input_ids, positions,
                      intermediate_tensors=None, inputs_embeds=None):
        runtime.observe_input_tokens(input_ids)   # update per-request state in think mode
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)   # h^0

        inject = runtime.build_inject_tensor(
            num_tokens=hidden_states.shape[0],
            hidden_size=hidden_states.shape[-1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, inject)

        hidden_states = self.norm(hidden_states)
        return hidden_states
    return model_forward


def _install_injection(causal_lm, runtime: InjectionRuntime, *,
                       layer_fwd_factory: Callable = _make_layer_forward,
                       per_layer_setup: Callable | None = None):
    model = causal_lm.model
    layer_fwd = layer_fwd_factory(runtime)

    real_layers = [
        l for l in model.layers if l.__class__.__name__ != "PPMissingLayer"
    ]
    inject_set = runtime.resolve_inject_layers(len(real_layers))

    idx = 0
    for layer in model.layers:
        if layer.__class__.__name__ == "PPMissingLayer":   # PP placeholder layer (none when PP=1)
            continue
        layer.forward = types.MethodType(layer_fwd, layer)
        layer._ri_do_inject = (inject_set is None) or (idx in inject_set)
        if per_layer_setup is not None:
            per_layer_setup(layer)
        idx += 1
    model.forward = types.MethodType(_make_model_forward(runtime), model)

    if inject_set is None:
        print(f"[residual] inject layers: ALL ({len(real_layers)})")
    elif not inject_set:
        print(
            f"[residual] inject layers: NONE "
            f"(inject_layer={runtime.config.inject_layer} out of range 0..{len(real_layers) - 1})"
        )
    else:
        print(f"[residual] inject layers: {sorted(inject_set)} / total {len(real_layers)}")


def _wrap_causal_lm(base_cls: type, name: str, *,
                    install_kwargs: dict | None = None) -> type:
    """Build a residual-injection model class from a vLLM CausalLM subclass."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    install_kwargs = install_kwargs or {}

    class ResidualInjectionModel(base_cls):
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            if get_tensor_model_parallel_world_size() > 1:
                raise RuntimeError(
                    "Residual Injection currently assumes TP=1 (embeddings are not sharded).")
            self._inj_runtime = InjectionRuntime(self, CONFIG)
            _install_injection(self, self._inj_runtime, **install_kwargs)

        def compute_logits(self, hidden_states):
            logits = super().compute_logits(hidden_states)
            self._inj_runtime.update_deltas_from_logits(logits)
            return logits

    ResidualInjectionModel.__name__ = name
    ResidualInjectionModel.__qualname__ = name
    return ResidualInjectionModel


def build_llama_model_class():
    from vllm.model_executor.models.llama import LlamaForCausalLM
    return _wrap_causal_lm(LlamaForCausalLM, "LlamaForResidualInjection")


def build_qwen3_model_class():
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    return _wrap_causal_lm(Qwen3ForCausalLM, "Qwen3ForResidualInjection")


def build_glm4_model_class():
    # GLM-4 dense, 0414 series (e.g. GLM-4-9B-0414 / GLM-4-32B-0414).
    # Note: not GLM-4-9B-Chat (ChatGLM architecture; see build_chatglm_model_class).
    from vllm.model_executor.models.glm4 import Glm4ForCausalLM
    return _wrap_causal_lm(
        Glm4ForCausalLM, "Glm4ForResidualInjection",
        install_kwargs=dict(
            layer_fwd_factory=_make_glm4_layer_forward,
            per_layer_setup=_make_glm4_per_layer_setup(),
        ),
    )


# ----------------------------------------------------------------------------
# ChatGLM architecture (GLM-4-9B-Chat / GLM-4-9B-Chat-1M / chatglm3, etc.)
#   - Layers: transformer.encoder.layers; block class name "GLMBlock"
#   - Attention submodule is self_attention; block returns a single tensor (non-fused residual)
#   - Residual path controlled by apply_residual_connection_post_layernorm (9B-Chat default False)
#   - Injection point: after post-attention residual, before MLP (same as dense path)
# inject tensor uses hybrid mechanism: built in wrapper.forward and stored in _ri_inject;
# blocks read it via back-reference to causal_lm; no need to rewrite GLMTransformer.
# ----------------------------------------------------------------------------
def _make_glm_block_forward(causal_lm):
    def block_forward(self, hidden_states, position_ids):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attention(
            hidden_states=layernorm_output,
            position_ids=position_ids,
        )
        post_ln = getattr(self, "apply_residual_connection_post_layernorm", False)
        residual = layernorm_output if post_ln else hidden_states
        layernorm_input = residual + attention_output

        inj = getattr(causal_lm, "_ri_inject", None)
        if inj is not None and getattr(self, "_ri_do_inject", True):
            layernorm_input = layernorm_input + inj

        layernorm_output = self.post_attention_layernorm(layernorm_input)
        residual = layernorm_output if post_ln else layernorm_input
        return self.mlp(layernorm_output) + residual
    return block_forward


def _install_chatglm_injection(causal_lm, runtime: InjectionRuntime):
    blocks = [m for _, m in causal_lm.named_modules()
              if m.__class__.__name__ == "GLMBlock"]
    if not blocks:
        raise RuntimeError("GLMBlock layers not found (confirm ChatGLM architecture?)")

    inject_set = runtime.resolve_inject_layers(len(blocks))
    block_fwd = _make_glm_block_forward(causal_lm)
    for idx, blk in enumerate(blocks):
        blk.forward = types.MethodType(block_fwd, blk)
        blk._ri_do_inject = (inject_set is None) or (idx in inject_set)

    if inject_set is None:
        print(f"[residual] inject layers (chatglm): ALL ({len(blocks)})")
    elif not inject_set:
        print(f"[residual] inject layers (chatglm): NONE "
              f"(inject_layer={runtime.config.inject_layer} out of range 0..{len(blocks) - 1})")
    else:
        print(f"[residual] inject layers (chatglm): {sorted(inject_set)} / total {len(blocks)}")


def _wrap_chatglm(base_cls: type, name: str) -> type:
    from vllm.distributed import get_tensor_model_parallel_world_size

    class ChatGLMResidualInjectionModel(base_cls):
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            if get_tensor_model_parallel_world_size() > 1:
                raise RuntimeError("Residual Injection assumes TP=1.")
            cfg = vllm_config.model_config.hf_config
            text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
            self._ri_hidden = text_cfg.hidden_size
            self._ri_inject = None
            self._inj_runtime = InjectionRuntime(self, CONFIG)
            _install_chatglm_injection(self, self._inj_runtime)

        def forward(self, input_ids=None, positions=None, *args, **kwargs):
            n_tok, dev = 0, None
            if input_ids is not None:
                self._inj_runtime.observe_input_tokens(input_ids)
                n_tok, dev = input_ids.shape[0], input_ids.device
            else:
                emb = kwargs.get("inputs_embeds")
                if emb is not None:
                    n_tok, dev = emb.shape[0], emb.device
            self._ri_inject = (
                self._inj_runtime.build_inject_tensor(
                    num_tokens=n_tok, hidden_size=self._ri_hidden,
                    device=dev, dtype=next(self.parameters()).dtype)
                if n_tok > 0 else None
            )
            return super().forward(input_ids, positions, *args, **kwargs)

        def compute_logits(self, hidden_states, *args, **kwargs):
            logits = super().compute_logits(hidden_states, *args, **kwargs)
            self._inj_runtime.update_deltas_from_logits(logits)
            return logits

    ChatGLMResidualInjectionModel.__name__ = ChatGLMResidualInjectionModel.__qualname__ = name
    return ChatGLMResidualInjectionModel


def build_chatglm_model_class():
    # GLM-4-9B-Chat and other ChatGLM variants (config architectures: ChatGLMModel /
    # ChatGLMForConditionalGeneration).
    from vllm.model_executor.models.chatglm import ChatGLMForCausalLM
    return _wrap_chatglm(ChatGLMForCausalLM, "ChatGLMForResidualInjection")


def _install_injection_hook(causal_lm, runtime: InjectionRuntime, n_layers: int):
    # Locate text decoder layers: under language_model in hybrid models; match by len==n_layers
    layers = longest = None
    for _, m in causal_lm.named_modules():
        if isinstance(m, nn.ModuleList):
            if len(m) == n_layers:
                layers = m; break
            if longest is None or len(m) > len(longest):
                longest = m
    layers = layers or longest
    if layers is None:
        raise RuntimeError("decoder layer ModuleList not found")

    real_idx = [i for i, l in enumerate(layers)
                if l.__class__.__name__ != "PPMissingLayer"]
    inject_set = runtime.resolve_inject_layers(len(real_idx))

    def make_hook(causal):
        def hook(module, args, output):
            inj = getattr(causal, "_ri_inject", None)
            if inj is None:
                return output
            if not getattr(causal, "_ri_probed", False):   # one-time probe
                kind = type(output).__name__
                ln = len(output) if isinstance(output, tuple) else "-"
                print(f"[residual] layer output: type={kind} len={ln}")
                causal._ri_probed = True
            if isinstance(output, tuple):
                # vLLM fused residual is usually (hidden_states, residual); inject into residual stream.
                lst = list(output)
                if len(lst) >= 2 and lst[-1] is not None:
                    lst[-1] = lst[-1] + inj
                else:
                    lst[0] = lst[0] + inj
                return tuple(lst)
            return output + inj
        return hook

    installed = []
    for slot, gi in enumerate(real_idx):
        if inject_set is None or slot in inject_set:
            layers[gi].register_forward_hook(make_hook(causal_lm))
            installed.append(slot)
    tag = "ALL" if inject_set is None else (installed or "NONE")
    print(f"[residual] inject layers (hook): {tag} / total {len(real_idx)}")


def _wrap_hybrid_causal_lm(base_cls: type, name: str) -> type:
    from vllm.distributed import get_tensor_model_parallel_world_size

    class HybridResidualInjectionModel(base_cls):
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            if get_tensor_model_parallel_world_size() > 1:
                raise RuntimeError("Residual Injection assumes TP=1.")
            text_cfg = vllm_config.model_config.hf_config.get_text_config()
            self._ri_hidden = text_cfg.hidden_size
            self._ri_inject = None
            self._inj_runtime = InjectionRuntime(self, CONFIG)
            _install_injection_hook(self, self._inj_runtime, text_cfg.num_hidden_layers)

        def forward(self, input_ids=None, positions=None, *args, **kwargs):
            n_tok, dev = 0, None
            if input_ids is not None:
                self._inj_runtime.observe_input_tokens(input_ids)
                n_tok, dev = input_ids.shape[0], input_ids.device
            else:
                emb = kwargs.get("inputs_embeds")
                if emb is not None:
                    n_tok, dev = emb.shape[0], emb.device
            self._ri_inject = (
                self._inj_runtime.build_inject_tensor(
                    num_tokens=n_tok, hidden_size=self._ri_hidden,
                    device=dev, dtype=next(self.parameters()).dtype)
                if n_tok > 0 else None
            )
            return super().forward(input_ids, positions, *args, **kwargs)

        def compute_logits(self, hidden_states, *args, **kwargs):
            logits = super().compute_logits(hidden_states, *args, **kwargs)
            self._inj_runtime.update_deltas_from_logits(logits)
            return logits

    HybridResidualInjectionModel.__name__ = HybridResidualInjectionModel.__qualname__ = name
    return HybridResidualInjectionModel


def build_qwen3_5_model_class():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    return _wrap_hybrid_causal_lm(Qwen3_5ForConditionalGeneration, "Qwen3_5ForResidualInjection")

def build_qwen3_5_moe_model_class():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
    return _wrap_hybrid_causal_lm(
        Qwen3_5MoeForConditionalGeneration,
        "Qwen3_5MoeForResidualInjection",
    )


def build_glm4_moe_model_class():
    # GLM-4.5 / 4.6 (Glm4MoeForCausalLM): MoE + initial dense layers; hook path is most reliable
    # (inject into layer output residual stream, same as Qwen3.5).
    from vllm.model_executor.models.glm4_moe import Glm4MoeForCausalLM
    return _wrap_hybrid_causal_lm(Glm4MoeForCausalLM, "Glm4MoeForResidualInjection")

def build_glm_model_class():
    from vllm.model_executor.models.glm import GlmForCausalLM  # glm.py, not glm4.py
    return _wrap_hybrid_causal_lm(GlmForCausalLM, "GlmForResidualInjection")


_MODEL_BUILDERS: dict[str, Callable[[], type]] = {
    "LlamaForResidualInjection": build_llama_model_class,
    "Qwen3ForResidualInjection": build_qwen3_model_class,
    "Qwen3_5ForResidualInjection": build_qwen3_5_model_class,
    "Qwen3_5MoeForResidualInjection": build_qwen3_5_moe_model_class,
    "GlmForResidualInjection": build_glm_model_class,   # GLM-4-9B-Chat
    "Glm4ForResidualInjection": build_glm4_model_class,         # GLM-4-*-0414 dense
    "Glm4MoeForResidualInjection": build_glm4_moe_model_class,  # GLM-4.5/4.6 MoE
    "ChatGLMForResidualInjection": build_chatglm_model_class,
}


def register_residual_models() -> None:
    from vllm import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    for arch_name, builder in _MODEL_BUILDERS.items():
        if arch_name not in supported:
            ModelRegistry.register_model(arch_name, builder())


# Legacy API compatibility
def build_model_class():
    return build_llama_model_class()
