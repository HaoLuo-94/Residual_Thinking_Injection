"""Injection runtime: batch layout context + per-request Δ maintenance."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import InjectionConfig


# ----------------------------------------------------------------------------
# Per-step batch layout (published by patch.py before forward)
# ----------------------------------------------------------------------------
@dataclass
class BatchLayout:
    req_ids: List[str]          # Request order in the current batch (matches flat token layout)
    num_scheduled: List[int]    # Number of tokens scheduled per request this step
    # The following three are optional: per-request sampling params aligned with req_ids.
    # Published by patch.py from sampling_metadata; if None, runtime falls back to CONFIG broadcast.
    temperature: Optional[List[float]] = None
    top_p: Optional[List[float]] = None
    top_k: Optional[List[int]] = None


class _BatchContext:
    """Thread-local storage for the current step's batch layout."""
    def __init__(self):
        self._local = threading.local()

    def set(self, layout: Optional[BatchLayout]):
        self._local.value = layout

    def get(self) -> Optional[BatchLayout]:
        return getattr(self._local, "value", None)


BATCH_CTX = _BatchContext()


# ----------------------------------------------------------------------------
# Injection runtime
# ----------------------------------------------------------------------------
class InjectionRuntime:
    def __init__(self, causal_lm: torch.nn.Module, config: InjectionConfig):
        self.lm = causal_lm                 # LlamaForCausalLM instance
        self.config = config
        self.deltas: Dict[str, torch.Tensor] = {}   # req_id -> Δ [D]
        self.in_think: Dict[str, bool] = {}         # req_id -> inside <think>...</think>
        # committed-anchor only: compute_logits stores e_soft; observe finalizes next step.
        # Lifetime is exactly one step (compute_logits[t] write -> observe[t+1] consume and clear).
        self.pending_soft: Dict[str, torch.Tensor] = {}

    @property
    def embed_weight(self) -> torch.Tensor:
        w = getattr(self, "_embed_w_cache", None)
        if w is not None:
            return w
        lm = self.lm
        # 1) dense legacy path (Llama/Qwen3)
        w = getattr(getattr(getattr(lm, "model", None), "embed_tokens", None), "weight", None)
        # 2) generic interface (multimodal/hybrid; embed under language_model)
        if w is None:
            try:
                w = getattr(lm.get_input_embeddings(), "weight", None)
            except Exception:
                w = None
        # # 3) fallback: scan VocabParallelEmbedding
        # if w is None:
        #     from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
        #     for _, m in lm.named_modules():
        #         if isinstance(m, VocabParallelEmbedding) and getattr(m, "weight", None) is not None:
        #             w = m.weight
        #             break

        # 3) fallback: scan VocabParallelEmbedding, but skip lm_head (ParallelLMHead is a subclass),
        #    and prefer input embeddings whose name contains "embed" to avoid ChatGLM picking output_layer.
        if w is None:
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                VocabParallelEmbedding, ParallelLMHead,
            )
            cand = None
            for name, m in lm.named_modules():
                if isinstance(m, ParallelLMHead):
                    continue
                if isinstance(m, VocabParallelEmbedding) and getattr(m, "weight", None) is not None:
                    if "embed" in name.lower():
                        w = m.weight
                        break
                    cand = cand or m
            if w is None and cand is not None:
                w = cand.weight
        if w is None:
            raise RuntimeError("InjectionRuntime: cannot locate token embedding weights")
        self._embed_w_cache = w
        return w

    # ---- Resolve layer indices to inject (called once at model construction) -----------------------
    def resolve_inject_layers(self, num_layers: int) -> Optional[set]:
        """
        Return the set of layer indices to inject; None means all layers.
          inject_layer == -1 -> None (all layers)
          inject_layer == k  -> {k}  (layer k only, 0-indexed)
          out of range/invalid -> set() (inject no layers)
        """
        spec = self.config.inject_layer
        if spec is None or spec == -1:
            return None
        if spec < -1 or spec >= num_layers:
            return set()
        return {spec}

    # ---- committed-anchor: finalize Δ using the previous step's actually sampled token -------------------
    def _finalize_committed_deltas(self, input_ids: torch.Tensor,
                                   layout: BatchLayout) -> None:
        """
        committed-anchor mode: e_hard = embedding of the token committed on the previous step
        (that token is this step's input_ids), minus e_soft stored in pending_soft gives Δ.

        Note: in committed mode, self.deltas is written only by this method (observe phase);
        update_deltas_from_logits only writes pending_soft. Responsibilities do not overlap.
        """
        pending = self.pending_soft
        if not pending:
            self.deltas = {}
            self.pending_soft = {}
            return

        offsets, rids = [], []
        offset = 0
        for req_id, n in zip(layout.req_ids, layout.num_scheduled):
            # Finalize only pure decode steps (n==1) with pending e_soft from the previous step;
            # requests that finished last step (absent this step) have pending cleared at the end.
            if n == 1 and req_id in pending:
                offsets.append(offset)
                rids.append(req_id)
            offset += n

        if not offsets:
            self.deltas = {}
            self.pending_soft = {}
            return

        E = self.embed_weight
        idx = torch.as_tensor(offsets, device=input_ids.device, dtype=torch.long)
        committed = input_ids[idx]                       # [m] tokens sampled per request on previous step
        e_hard = E[committed].float()                    # [m, D]

        self.deltas = {
            rid: (pending[rid] - e_hard[j]).to(E.dtype).detach()
            for j, rid in enumerate(rids)
        }
        self.pending_soft = {}

    # ---- Before forward: update think state from this step's input tokens -------------------
    def observe_input_tokens(self, input_ids: Optional[torch.Tensor]) -> None:
        """
        Two tasks:
          1) (committed-anchor mode) Finalize Δ using this step's input_ids as the previous commit token;
             independent of inject_phase; needed in both phases.
          2) (only when inject_phase == "think") Track whether each request is inside
             <think>...</think>, for build_inject_tensor to decide injection.
        """
        if not self.config.enabled:
            return
        if input_ids is None:
            return
        layout = BATCH_CTX.get()
        # print("[obs] layout is None?", layout is None)   # add this line
        if layout is None:
            return
        if input_ids.dim() != 1 or input_ids.shape[0] != sum(layout.num_scheduled):
            return

        # ---- (1) committed-anchor finalize (before think state update; does not depend on in_think) ----
        if self.config.hard_anchor == "committed":
            self._finalize_committed_deltas(input_ids, layout)

        # ---- (2) think phase only: maintain <think>...</think> state ----
        if self.config.inject_phase != "think":
            return

        start_id = self.config.think_start_id
        end_id = self.config.think_end_id

            # add here
        if max(layout.num_scheduled) > 1:   # print only on prefill steps to avoid decode spam
            print("[think-check] start_id=", start_id,
                  "in prompt?", bool((input_ids == start_id).any()) if start_id is not None else None)
        # end add here

        # prefill (n>1) marks a new sequence start -> reset state; unseen req defaults outside think.
        # So even if vLLM reuses req_id across generate() calls, prior request state is not carried over.
        for req_id, n in zip(layout.req_ids, layout.num_scheduled):
            if n > 1 or req_id not in self.in_think:
                self.in_think[req_id] = False

        if start_id is None and end_id is None:
            return

        # Fast path: no boundary tokens this step, state unchanged (one GPU reduction, no per-token copy)
        mask = None
        if start_id is not None:
            mask = input_ids == start_id
        if end_id is not None:
            m2 = input_ids == end_id
            mask = m2 if mask is None else (mask | m2)
        if mask is None or not bool(mask.any()):
            return

        ids_cpu = input_ids.tolist()
        offset = 0
        for req_id, n in zip(layout.req_ids, layout.num_scheduled):
            state = self.in_think.get(req_id, False)
            for tok in ids_cpu[offset:offset + n]:
                if start_id is not None and tok == start_id:
                    state = True
                elif end_id is not None and tok == end_id:
                    state = False
            self.in_think[req_id] = state
            offset += n
        # temp debug: inspect final state
        if max(layout.num_scheduled) > 1:
            print("[think-check] in_think after prefill =", dict(self.in_think))

    # ---- forward: build [num_tokens, D] injection tensor (already scaled by α) ---------------------
    def build_inject_tensor(self, num_tokens, hidden_size, device, dtype):
        out = torch.zeros((num_tokens, hidden_size), device=device, dtype=dtype)
        if not self.config.enabled:
            return out

        layout = BATCH_CTX.get()
        if layout is None:
            return out

        if sum(layout.num_scheduled) != num_tokens:
            return out

        # think_only = self.config.inject_phase == "think"

        # offset = 0
        # for req_id, n in zip(layout.req_ids, layout.num_scheduled):
        #     if n == 1 and req_id in self.deltas:
        #         if (not think_only) or self.in_think.get(req_id, False):
        #             out[offset] = self.deltas[req_id].to(device=device, dtype=dtype)
        #     offset += n'
        think_only = self.config.inject_phase == "think"
        offset = 0
        n_inj = 0                                              # debug
        for req_id, n in zip(layout.req_ids, layout.num_scheduled):
            if n == 1 and req_id in self.deltas:
                if (not think_only) or self.in_think.get(req_id, False):
                    out[offset] = self.deltas[req_id].to(device=device, dtype=dtype)
                    n_inj += 1                                 # debug
            offset += n
        # # debug: print only when in_think turns True or injection actually happens
        # if think_only and (n_inj or any(self.in_think.values())):
        #     print(f"[inject] injected={n_inj} in_think={dict(self.in_think)} "
        #           f"have_deltas={list(self.deltas)}")

        if self.config.alpha != 1.0:
            out.mul_(self.config.alpha)
        return out

    # ---- Per valid row: sampling params (prefer layout, else CONFIG broadcast) ----------
    def _row_sampling_params(self, layout: BatchLayout, valid_rows: List[int],
                             device: torch.device):
        """
        Return (temperature[R], top_k[R], top_p[R]) aligned with valid_rows.
        If layout published per-request params, take per row; else broadcast CONFIG soft_*.
        """
        R = len(valid_rows)
        max_r = max(valid_rows)

        def pick(seq, default, dtype):
            if seq is not None and len(seq) > max_r:
                vals = [seq[r] for r in valid_rows]
            else:
                vals = [default] * R
            return torch.tensor(vals, device=device, dtype=dtype)

        temp  = pick(layout.temperature, self.config.soft_temperature, torch.float32)
        top_k = pick(layout.top_k,       self.config.soft_top_k,       torch.long)
        top_p = pick(layout.top_p,       self.config.soft_top_p,       torch.float32)
        return temp, top_k, top_p

    # ---- Soft/hard embeddings after mirroring sampler (temperature -> top_k -> top_p) ---------
    def _aligned_soft_hard(self, logits_valid: torch.Tensor,
                           temp: torch.Tensor, top_k: torch.Tensor,
                           top_p: torch.Tensor, E: torch.Tensor):
        """
        e_soft = probability-weighted embedding over the distribution after
                 {temperature scaling -> top_k truncation -> top_p truncation},
                 matching vLLM V1 sampler application order.
        e_hard = argmax embedding (invariant to temperature/top_k/top_p, i.e. α=0 greedy endpoint).

        Implementation uses a candidate pool (soft_pool_k) for sparse gather: top_k/top_p survivor
        sets are among the highest-logit tokens and must lie in the pool; exact when pool >= survivor size.
        top_p cumulative probability is computed on in-pool softmax (tail mass truncated); error negligible
        when the pool is large enough.
        """
        R, V = logits_valid.shape
        pool = min(self.config.soft_pool_k, V)
        pool_vals, pool_idx = torch.topk(logits_valid, pool, dim=-1)   # descending
        f = pool_vals.float()

        # 1) temperature scaling (greedy rows temp->0: softmax -> one-hot, e_soft->e_hard, Δ->0)
        f = f / temp.clamp_min(1e-5).unsqueeze(-1)

        # 2) top_k: pool is descending; truncate by column index (<=0 treated as disabled)
        col = torch.arange(pool, device=f.device).unsqueeze(0)        # [1, pool]
        kk = top_k.clone()
        kk[kk <= 0] = pool
        kk = kk.clamp(max=pool).unsqueeze(-1)                          # [R, 1]
        f = f.masked_fill(col >= kk, float("-inf"))

        # 3) top_p: keep minimal prefix where cumulative prob first reaches p (1.0 = disabled)
        #    descending: token i kept <=> prior cumulative (csum_i - p_i) < top_p
        probs = f.softmax(dim=-1)
        csum = probs.cumsum(dim=-1)
        keep = (csum - probs) < top_p.unsqueeze(-1)
        keep[:, 0] = True                                             # always keep argmax
        f = f.masked_fill(~keep, float("-inf"))

        # 4) probability-weighted embedding on filtered distribution
        p = f.softmax(dim=-1)                                         # [R, pool]
        emb = E[pool_idx].float()                                    # [R, pool, D]
        e_soft = (p.unsqueeze(-1) * emb).sum(dim=1)
        e_hard = emb[:, 0, :]                                         # argmax (invariant)
        return e_soft, e_hard

    # ---- compute_logits: update Δ from this step's logits (for next step) --------------------
    @torch.no_grad()
    def update_deltas_from_logits(self, logits: torch.Tensor):
        if not self.config.enabled:
            return

        layout = BATCH_CTX.get()
        if layout is None or logits is None:
            return

        req_ids = layout.req_ids
        n_map = min(logits.shape[0], len(req_ids), len(layout.num_scheduled))

        valid_rows = [
            r for r in range(n_map)
            if layout.num_scheduled[r] == 1
        ]

        if not valid_rows:
            # committed mode: self.deltas maintained by observe; here only clear pending.
            if self.config.hard_anchor == "committed":
                self.pending_soft = {}
            else:
                self.deltas = {}
            return

        logits_valid = logits[valid_rows]
        E = self.embed_weight

        # ---- compute e_soft (and e_hard for argmax mode) ----
        if not self.config.align_sampler:
            # Legacy path (preserves historical reproducibility): temperature=1.0, top_k=config.top_k, no top_p.
            # Note: logits here are raw lm_head output; sampler temperature/top_p/top_k come after,
            # so this path is decoupled from the actual sampling distribution.
            k = min(self.config.top_k, logits_valid.shape[-1])
            top_vals, top_idx = torch.topk(logits_valid, k, dim=-1)
            p = torch.softmax(top_vals.float(), dim=-1)
            emb = E[top_idx].float()
            e_soft = (p.unsqueeze(-1) * emb).sum(dim=1)
            e_hard = emb[:, 0, :]
        else:
            # New path: e_soft mirrors sampler temperature -> top_k -> top_p.
            temp, top_k_t, top_p_t = self._row_sampling_params(
                layout, valid_rows, logits_valid.device)
            e_soft, e_hard = self._aligned_soft_hard(
                logits_valid, temp, top_k_t, top_p_t, E)

        # ---- hard_anchor selects e_hard semantics ----
        if self.config.hard_anchor == "committed":
            # Store e_soft only; e_hard computed next step from actual commit token (see _finalize_*).
            # Do not write self.deltas this step: observe already finalized and build_inject consumed it.
            self.pending_soft = {
                req_ids[r]: e_soft[i].detach()
                for i, r in enumerate(valid_rows)
            }
        else:
            # argmax anchor (default): e_hard = greedy token, invariant to temp/top_k/top_p.
            delta = (e_soft - e_hard).to(E.dtype)
            self.deltas = {
                req_ids[r]: delta[i].detach()
                for i, r in enumerate(valid_rows)
            }

#         print(
#     "[delta]",
#     "valid_rows=", valid_rows,
#     "req_ids=", [req_ids[r] for r in valid_rows],
#     "delta_norm=", delta.float().norm(dim=-1).tolist(),
#     "stored=", list(self.deltas.keys()),
# )
