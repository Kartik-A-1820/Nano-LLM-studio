"""
Attention implementations.

GQA  — Grouped Query Attention (LLaMA-3 / DeepSeek-V1 style).
MLA  — Multi-head Latent Attention (DeepSeek-V2/V3 native).

Key MLA design points (faithful to the DeepSeek-V2 paper):

  1. Decoupled RoPE
     Q and K are split into a content part (no RoPE) and a tiny rope part.
     RoPE is applied *only* to the rope part.
     The rope key vector is shared across all heads (1 head, broadcast).

  2. KV compression
     Instead of projecting x → full K,V, we project to a low-rank latent:
       c_kv  = RMSNorm( W_kv_a(x) )          shape: (B, T, kv_lora_rank)
     At inference, only c_kv and the rope key k_pe are cached — not full K/V.
     Cache size per position per layer:
       MHA  → 2 * n_heads * head_dim
       GQA  → 2 * n_kv_heads * head_dim
       MLA  → kv_lora_rank + qk_rope_head_dim   (≈ 10–15× smaller than MHA)

  3. Q compression (optional)
     Q also goes through a low-rank bottleneck to reduce activation memory.

  4. Absorbed inference (decode path, T_q == 1)
     To avoid ever materialising full K and V during decode:
       k_nope[h] = W_UK[h] @ c_kv       W_UK extracted from kv_b_proj weights
       q_nope @ k_nope.T  ≡  (q_nope @ W_UK.T) @ c_kv.T
     So we pre-multiply q_nope into latent space (O(n_heads * qk_nope * kv_rank))
     and attend directly over cached c_kv (O(kv_rank * T_kv)).
     Same trick for V: attn_weights @ c_kv → absorb W_UV → output.
     This is exact, not approximate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .layers import RMSNorm, apply_rope


# ------------------------------------------------------------------ #
# Shared helper
# ------------------------------------------------------------------ #

def _sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None,
          is_causal: bool = False) -> Tensor:
    """
    Thin wrapper over F.scaled_dot_product_attention.
    Uses memory-efficient backend (works on Turing/GTX 1650 Ti).
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal
            )
    except (ImportError, AttributeError):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal
        )


# ------------------------------------------------------------------ #
# Grouped Query Attention
# ------------------------------------------------------------------ #

def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads to match Q heads. x: (B, T, n_kv_heads, head_dim)."""
    if n_rep == 1:
        return x
    B, T, H, D = x.shape
    return x.unsqueeze(3).expand(B, T, H, n_rep, D).reshape(B, T, H * n_rep, D)


class GroupedQueryAttention(nn.Module):
    """GQA: n_kv_heads shared KV heads, n_heads Q heads. Full RoPE on all dims."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.head_dim()

        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
        past_kv: tuple | None = None,
        is_causal: bool = False,
    ) -> tuple[Tensor, tuple]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=1)
            v = torch.cat([pv, v], dim=1)

        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = _sdpa(q, k, v, mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), (k.transpose(1, 2), v.transpose(1, 2))


# ------------------------------------------------------------------ #
# Multi-head Latent Attention  (DeepSeek-V2 / V3)
# ------------------------------------------------------------------ #

class MultiHeadLatentAttention(nn.Module):
    """
    Faithful DeepSeek-V2 MLA with:
      • Decoupled RoPE (q_pe / k_pe — small, positional)
      • Content key/value via low-rank KV latent c_kv
      • Optional Q low-rank compression
      • Compressed KV cache: (c_kv, k_pe) only — not full K, V
      • Absorbed-weight inference for decode: never materialise full K/V

    Projections (no bias anywhere):
      Q path:
        q_a_proj   : dim  → q_lora_rank          (optional, skipped if q_lora_rank=None)
        q_a_norm   : RMSNorm on q_lora_rank
        q_b_proj   : q_lora_rank → n_heads*(qk_nope+qk_rope)

      KV path:
        kv_a_proj  : dim → kv_lora_rank + qk_rope  (latent AND rope key packed together)
        kv_a_norm  : RMSNorm on kv_lora_rank        (only the latent part)
        kv_b_proj  : kv_lora_rank → n_heads*(qk_nope + v_head_dim)
                     first n_heads*qk_nope rows → K_nope (content keys)
                     next  n_heads*v_head_dim rows → V

      O proj: n_heads*v_head_dim → dim
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads        = config.n_heads
        self.qk_nope        = config.resolved_qk_nope_head_dim()
        self.qk_rope        = config.resolved_qk_rope_head_dim()
        self.v_dim          = config.resolved_v_head_dim()
        self.kv_rank        = config.resolved_kv_lora_rank()
        self.q_lora_rank    = config.q_lora_rank
        self.qk_head_dim    = self.qk_nope + self.qk_rope   # total attention head dim
        self.scale          = self.qk_head_dim ** -0.5

        d = config.dim

        # ---- Q path ----
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(d, self.q_lora_rank, bias=False)
            self.q_a_norm = RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(d, self.n_heads * self.qk_head_dim, bias=False)

        # ---- KV path ----
        # kv_a_proj packs the latent (kv_rank dims) and the rope key (qk_rope dims)
        # into one projection to save a matmul.
        self.kv_a_proj = nn.Linear(d, self.kv_rank + self.qk_rope, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_rank)

        # kv_b_proj up-projects the latent to content keys + values
        # Layout of output: [K_nope ... | V ...]
        self.kv_b_proj = nn.Linear(
            self.kv_rank,
            self.n_heads * (self.qk_nope + self.v_dim),
            bias=False,
        )

        # ---- Output ----
        self.o_proj = nn.Linear(self.n_heads * self.v_dim, d, bias=False)

    # ---------------------------------------------------------------- #
    # Internal: extract W_K, W_V slices from kv_b_proj for absorption #
    # ---------------------------------------------------------------- #

    def _weight_slices(self):
        """
        Split kv_b_proj.weight into W_K and W_V.
        kv_b_proj.weight shape: (n_heads*(qk_nope+v_dim), kv_rank)
        W_K: (n_heads, qk_nope, kv_rank)
        W_V: (n_heads, v_dim,   kv_rank)
        """
        w = self.kv_b_proj.weight                            # (out, kv_rank)
        k_rows = self.n_heads * self.qk_nope
        W_K = w[:k_rows].view(self.n_heads, self.qk_nope, self.kv_rank)
        W_V = w[k_rows:].view(self.n_heads, self.v_dim,  self.kv_rank)
        return W_K, W_V

    # ---------------------------------------------------------------- #
    # Shared: compute Q                                                 #
    # ---------------------------------------------------------------- #

    def _compute_q(self, x: Tensor, freqs: Tensor):
        """Returns q_nope (B,T,H,qk_nope) and q_pe (B,T,H,qk_rope) with RoPE on pe."""
        B, T, _ = x.shape
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.view(B, T, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope, self.qk_rope], dim=-1)
        q_pe = apply_rope(q_pe, freqs)            # RoPE on small positional part only
        return q_nope, q_pe

    # ---------------------------------------------------------------- #
    # Shared: compress current token(s) KV and concat with cache       #
    # ---------------------------------------------------------------- #

    def _compress_kv(
        self, x: Tensor, freqs: Tensor, past_kv: tuple | None
    ) -> tuple[Tensor, Tensor, tuple]:
        """
        Returns:
            c_kv  (B, T_kv, kv_rank)           — full latent (past + new)
            k_pe  (B, T_kv, 1, qk_rope)        — rope key, shared across heads
            new_kv = (c_kv, k_pe)              — to be stored in cache
        """
        B, T, _ = x.shape
        kv_ab = self.kv_a_proj(x)                            # (B,T, kv_rank+qk_rope)
        c_kv_new, k_pe_new = kv_ab.split([self.kv_rank, self.qk_rope], dim=-1)
        c_kv_new = self.kv_a_norm(c_kv_new)                  # normalise latent
        k_pe_new = apply_rope(k_pe_new.unsqueeze(2), freqs)  # (B,T,1,qk_rope)

        if past_kv is not None:
            pc, pk = past_kv
            c_kv = torch.cat([pc, c_kv_new], dim=1)
            k_pe = torch.cat([pk, k_pe_new], dim=1)
        else:
            c_kv, k_pe = c_kv_new, k_pe_new

        return c_kv, k_pe, (c_kv, k_pe)

    # ---------------------------------------------------------------- #
    # Training / prefill path: expand c_kv to full K, V; use SDPA     #
    # ---------------------------------------------------------------- #

    def _forward_train(
        self, x: Tensor, freqs: Tensor, mask: Tensor | None, is_causal: bool
    ) -> tuple[Tensor, tuple]:
        B, T, _ = x.shape

        q_nope, q_pe = self._compute_q(x, freqs)

        # Compress KV (no past cache during training)
        c_kv, k_pe, new_kv = self._compress_kv(x, freqs, past_kv=None)

        # Expand latent → content K and V
        kv_full = self.kv_b_proj(c_kv)                       # (B,T, n_heads*(qk_nope+v_dim))
        kv_full = kv_full.view(B, T, self.n_heads, self.qk_nope + self.v_dim)
        k_nope, v = kv_full.split([self.qk_nope, self.v_dim], dim=-1)

        # Assemble full Q: [nope | rope]
        q = torch.cat([q_nope, q_pe], dim=-1)                # (B,T,H,qk_head_dim)

        # Assemble full K: [k_nope | k_pe broadcast across heads]
        k_pe_heads = k_pe.expand(-1, -1, self.n_heads, -1)   # (B,T,H,qk_rope)
        k = torch.cat([k_nope, k_pe_heads], dim=-1)          # (B,T,H,qk_head_dim)

        # (B,H,T,D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = _sdpa(q, k, v, mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_dim)
        return self.o_proj(out), new_kv

    # ---------------------------------------------------------------- #
    # Inference / decode path: absorbed weights, never expand K/V      #
    # ---------------------------------------------------------------- #

    def _forward_decode(
        self, x: Tensor, freqs: Tensor, past_kv: tuple, mask: Tensor | None
    ) -> tuple[Tensor, tuple]:
        """
        Absorbed attention for autoregressive decode.

        Instead of:
            k_nope = W_K @ c_kv
            logits += q_nope @ k_nope.T

        We do:
            q_abs = q_nope @ W_K.T       (absorb W_K into Q — done once per step)
            logits += q_abs  @ c_kv.T    (attend over cached latent directly)

        KV cache stores only (c_kv, k_pe), not expanded K or V.
        """
        B, T, _ = x.shape

        q_nope, q_pe = self._compute_q(x, freqs)             # (B,T,H,qk_nope/rope)
        c_kv, k_pe, new_kv = self._compress_kv(x, freqs, past_kv)
        T_kv = c_kv.shape[1]

        W_K, W_V = self._weight_slices()

        # ---- Absorbed Q: project into latent key space ----
        # q_nope: (B, T, H, qk_nope)   W_K: (H, qk_nope, kv_rank)
        q_abs = torch.einsum("bthd,hdk->bthk", q_nope, W_K)  # (B,T,H,kv_rank)

        # ---- Attention logits ----
        # content part: q_abs  @ c_kv.T  → (B,H,T,T_kv)
        #   c_kv: (B, T_kv, kv_rank)
        logits_c = torch.einsum(
            "bthk,bsk->bhts",
            q_abs,
            c_kv,
        )  # (B, H, T, T_kv)

        # rope part: q_pe @ k_pe.T  → (B,H,T,T_kv)
        #   q_pe: (B,T,H,qk_rope)   k_pe: (B,T_kv,1,qk_rope) shared across H
        logits_r = torch.einsum(
            "bthd,bsd->bhts",
            q_pe,
            k_pe.squeeze(2),          # (B, T_kv, qk_rope)
        )  # (B, H, T, T_kv)   — head dim broadcasts automatically

        attn_logits = (logits_c + logits_r) * self.scale

        # Causal mask (passed in from model)
        if mask is not None:
            attn_logits = attn_logits + mask

        attn_weights = torch.softmax(attn_logits.float(), dim=-1).to(q_nope.dtype)

        # ---- Absorbed V: attend over latent, then project ----
        # attn_weights: (B,H,T,T_kv)   c_kv: (B,T_kv,kv_rank)
        ctx_latent = torch.einsum(
            "bhts,bsk->bhtk",
            attn_weights,
            c_kv,
        )  # (B, H, T, kv_rank)

        # ctx_latent @ W_V.T  →  (B, H, T, v_dim)
        out = torch.einsum("bhtk,hvk->bhtv", ctx_latent, W_V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_dim)
        return self.o_proj(out), new_kv

    # ---------------------------------------------------------------- #
    # Public forward — dispatches to train or decode path              #
    # ---------------------------------------------------------------- #

    def forward(
        self,
        x: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
        past_kv: tuple | None = None,
        is_causal: bool = False,
    ) -> tuple[Tensor, tuple]:
        if past_kv is not None:
            return self._forward_decode(x, freqs, past_kv, mask)
        return self._forward_train(x, freqs, mask, is_causal)


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

def build_attention(config: ModelConfig) -> nn.Module:
    if config.use_mla:
        return MultiHeadLatentAttention(config)
    return GroupedQueryAttention(config)
