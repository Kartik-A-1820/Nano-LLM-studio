"""NanoLLM: Full transformer model with TransformerBlock."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .layers import RMSNorm, SwiGLU, precompute_rope_freqs
from .attention import build_attention


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.dim)
        self.attn = build_attention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = SwiGLU(config.dim, config.ffn_hidden_dim(), config.dropout)

    def _body(
        self,
        x: Tensor,
        freqs: Tensor,
        mask: Tensor | None,
        past_kv: tuple | None,
        is_causal: bool,
    ) -> tuple[Tensor, tuple]:
        h, new_kv = self.attn(
            self.attn_norm(x), freqs, mask, past_kv, is_causal=is_causal
        )
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv

    def forward(
        self,
        x: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
        past_kv: tuple | None = None,
        use_checkpoint: bool = False,
        is_causal: bool = False,
    ) -> tuple[Tensor, tuple]:
        if use_checkpoint and self.training:
            # Gradient checkpointing: recompute activations on backward pass.
            # past_kv is always None during training, so no KV cache needed here.
            def _fwd(x_in, freqs_in):
                h, _ = self.attn(
                    self.attn_norm(x_in), freqs_in, mask, None, is_causal=is_causal
                )
                x_out = x_in + h
                x_out = x_out + self.ffn(self.ffn_norm(x_out))
                return x_out

            x = checkpoint(_fwd, x, freqs, use_reentrant=False)
            return x, None

        return self._body(x, freqs, mask, past_kv, is_causal)


class NanoLLM(nn.Module):
    """
    Nano Language Model.
      • GQA or MLA attention (configured via ModelConfig.use_mla)
      • SwiGLU FFN, RMSNorm, RoPE
      • Tied input/output embeddings (optional)
      • Gradient checkpointing support
      • KV-cache inference (MLA caches compressed latents, not full K/V)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.head.weight = self.embed.weight

        # RoPE frequencies.
        # For MLA, RoPE only operates on qk_rope_head_dim (the small positional slice).
        # For GQA, RoPE covers the full head_dim.
        rope_dim = config.rope_head_dim()
        freqs = precompute_rope_freqs(rope_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_freqs", freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        past_kvs: list | None = None,
        use_checkpoint: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            input_ids:       (B, T) long tensor
            targets:         (B, T) long tensor — if given, compute cross-entropy loss
            past_kvs:        list of per-layer KV caches (for autoregressive decode).
                             For MLA each entry is (c_kv, k_pe); for GQA it is (K, V).
                             Pass None during training.
            use_checkpoint:  enable gradient checkpointing (training only)

        Returns:
            (logits, loss)   loss is None if targets are not provided
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed(input_ids)  # (B, T, dim)

        # During training (no past_kvs), let SDPA handle the causal mask internally.
        # During inference (past_kvs provided), build an explicit additive mask so
        # positions in the new token(s) cannot attend to future cached positions.
        is_causal = past_kvs is None
        mask = None
        if past_kvs is not None:
            # Get the length of the existing cache from the first non-None entry
            past_len = 0
            for pv in past_kvs:
                if pv is not None:
                    past_len = pv[0].shape[1]
                    break
            total_len = past_len + T
            # Additive causal mask: -inf where attention is forbidden
            mask = torch.full((T, total_len), float("-inf"), device=device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=past_len + 1)

        new_kvs: list = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(
                x, self.rope_freqs, mask, past_kv, use_checkpoint, is_causal
            )
            new_kvs.append(new_kv)

        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        # Return KV caches during inference so the caller can pass them back
        if past_kvs is not None:
            return logits, loss, new_kvs

        return logits, loss

    def resize_vocab(self, new_vocab_size: int) -> None:
        """
        Resize the embedding table and output head to a new vocabulary size.

        Existing token weights are preserved exactly.
        New token rows are initialised with the same N(0, 0.02) used at
        model creation — identical to how GPT-2 / LLaMA handle vocab extension.

        This must be called on CPU or with the model already on the target
        device; the caller is responsible for moving the model back to the
        training device afterward if needed.

        Args:
            new_vocab_size: Target vocabulary size. Must be >= current vocab_size.
        """
        old_vocab = self.config.vocab_size
        if new_vocab_size == old_vocab:
            return
        if new_vocab_size < old_vocab:
            raise ValueError(
                f"resize_vocab: new_vocab_size ({new_vocab_size}) < "
                f"current ({old_vocab}). Shrinking is not supported."
            )

        device = self.embed.weight.device
        dim    = self.config.dim
        n_new  = new_vocab_size - old_vocab

        # ── Embedding table ───────────────────────────────────────────
        old_embed_w = self.embed.weight.data          # (old_vocab, dim)
        new_embed_w = torch.empty(n_new, dim, device=device)
        nn.init.normal_(new_embed_w, mean=0.0, std=0.02)
        new_embed_weight = torch.cat([old_embed_w, new_embed_w], dim=0)

        self.embed = nn.Embedding(new_vocab_size, dim)
        self.embed.weight = nn.Parameter(new_embed_weight)

        # ── Output head ───────────────────────────────────────────────
        if self.config.tie_embeddings:
            # Tied: head shares the embedding weight — just point it
            self.head = nn.Linear(dim, new_vocab_size, bias=False)
            self.head.weight = self.embed.weight
        else:
            old_head_w = self.head.weight.data        # (old_vocab, dim)
            new_head_w = torch.empty(n_new, dim, device=device)
            nn.init.normal_(new_head_w, mean=0.0, std=0.02)
            new_head_weight = torch.cat([old_head_w, new_head_w], dim=0)
            self.head = nn.Linear(new_vocab_size, dim, bias=False)
            self.head.weight = nn.Parameter(new_head_weight)

        self.config.vocab_size = new_vocab_size
        import logging as _log
        _log.getLogger(__name__).info(
            f"resize_vocab: {old_vocab:,} → {new_vocab_size:,} "
            f"(+{n_new:,} new tokens)"
        )


    @classmethod
    def from_config(cls, config: ModelConfig) -> "NanoLLM":
        model = cls(config)
        config.param_count = model.param_count()
        return model
