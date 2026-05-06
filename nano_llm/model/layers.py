"""Foundational layer primitives: RMSNorm, SwiGLU, RoPE."""
import torch
import torch.nn as nn
from torch import Tensor


# ------------------------------------------------------------------ #
# RMSNorm
# ------------------------------------------------------------------ #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ------------------------------------------------------------------ #
# SwiGLU FFN
# ------------------------------------------------------------------ #

class SwiGLU(nn.Module):
    """
    FFN with SwiGLU activation (Noam Shazeer 2020).
    gate = SiLU(W1(x)), up = W3(x) — element-wise product → W2 down.
    No bias anywhere.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x)))


# ------------------------------------------------------------------ #
# Rotary Positional Embeddings (RoPE)
# ------------------------------------------------------------------ #

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10_000.0) -> Tensor:
    """
    Returns complex frequency tensor of shape (max_seq_len, dim // 2).
    Usage: freqs = precompute_rope_freqs(head_dim, max_seq_len)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """
    Apply RoPE to x.
    x shape: (B, T, n_heads, head_dim)
    freqs shape: (T, head_dim // 2)  — complex
    """
    B, T, H, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D//2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)
