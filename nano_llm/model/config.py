"""Model configuration and architecture tier definitions."""
from dataclasses import dataclass, field
from typing import Optional


# ── Vocabulary constants ───────────────────────────────────────────────
# Byte-level BPE: 4 special tokens + 256 byte tokens + learned merges.
# GPT-2 uses 50,257 tokens. We match that as the default target.
# The embedding table dominates memory at large vocab, so tiers are
# sized accordingly.
GPT2_VOCAB_SIZE = 50_257
DEFAULT_VOCAB_SIZE = GPT2_VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_multiplier: float = 8 / 3   # SwiGLU canonical ratio ≈ 2.667
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True
    # LLaMA-style long-context RoPE theta.
    # 500_000 is the LLaMA-3 value; gives good extrapolation beyond training length.
    rope_theta: float = 500_000.0

    # ------------------------------------------------------------------ #
    # Attention mode
    # ------------------------------------------------------------------ #
    use_mla: bool = True             # True: MLA (DeepSeek-V3 style), False: GQA

    # GQA parameters (used when use_mla=False)
    n_kv_heads: int = 2             # must divide n_heads evenly

    # MLA parameters (used when use_mla=True)
    qk_nope_head_dim: int = 0       # 0 = auto: head_dim * 3 // 4
    qk_rope_head_dim: int = 0       # 0 = auto: head_dim * 1 // 4  (must be even)
    v_head_dim: int = 0             # 0 = auto: same as head_dim
    kv_lora_rank: int = 0           # 0 = auto: dim // n_heads (≈ head_dim)
    q_lora_rank: Optional[int] = None  # None = no Q compression

    # Set by auto_arch after selection
    tier_name: str = "nano"
    param_count: int = 0

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #

    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def resolved_qk_nope_head_dim(self) -> int:
        if self.qk_nope_head_dim > 0:
            return self.qk_nope_head_dim
        hd = self.head_dim()
        return (hd * 3 // 4 + 1) & ~1   # round to even

    def resolved_qk_rope_head_dim(self) -> int:
        if self.qk_rope_head_dim > 0:
            return self.qk_rope_head_dim
        hd = self.head_dim()
        rope = hd - self.resolved_qk_nope_head_dim()
        return max(rope, 2)  # at least 2 for complex RoPE

    def resolved_v_head_dim(self) -> int:
        return self.v_head_dim if self.v_head_dim > 0 else self.head_dim()

    def resolved_kv_lora_rank(self) -> int:
        return self.kv_lora_rank if self.kv_lora_rank > 0 else self.head_dim()

    def rope_head_dim(self) -> int:
        return self.resolved_qk_rope_head_dim() if self.use_mla else self.head_dim()

    def ffn_hidden_dim(self) -> int:
        """Round to nearest 64 for cuBLAS efficiency."""
        raw = int(self.dim * self.ffn_multiplier)
        return (raw + 63) // 64 * 64

    # ------------------------------------------------------------------ #
    # Parameter count estimation
    # ------------------------------------------------------------------ #

    def estimate_param_count(self) -> int:
        d = self.dim
        h = self.ffn_hidden_dim()

        if self.use_mla:
            nope = self.resolved_qk_nope_head_dim()
            rope = self.resolved_qk_rope_head_dim()
            vdim = self.resolved_v_head_dim()
            r_kv = self.resolved_kv_lora_rank()
            r_q  = self.q_lora_rank
            qk_total = nope + rope

            if r_q is not None:
                attn_q = d * r_q + r_q + r_q * self.n_heads * qk_total
            else:
                attn_q = d * self.n_heads * qk_total

            attn_kv = d * (r_kv + rope) + r_kv + r_kv * self.n_heads * (nope + vdim)
            attn_o  = self.n_heads * vdim * d
            attn    = attn_q + attn_kv + attn_o
        else:
            hd   = self.head_dim()
            attn = (
                d * self.n_heads    * hd   # Q
                + d * self.n_kv_heads * hd # K
                + d * self.n_kv_heads * hd # V
                + self.n_heads * hd * d    # O
            )

        ffn       = d * h + d * h + h * d   # SwiGLU: W1 + W3 + W2
        norms     = 2 * d                   # two RMSNorm per block
        per_layer = attn + ffn + norms

        # Embedding dominates at large vocab — this is intentional (tied weights)
        embed      = self.vocab_size * d
        head       = 0 if self.tie_embeddings else self.vocab_size * d
        final_norm = d

        return self.n_layers * per_layer + embed + head + final_norm

    def estimate_training_vram_mb(
        self,
        batch_size: int = 1,
        seq_len: int = 2048,
        use_gradient_checkpointing: bool = True,
    ) -> float:
        """
        Fast analytical VRAM estimate (no GPU required).

        Accounts for:
          - fp16 model weights + gradients  (4 bytes/param)
          - fp32 AdamW optimizer states     (12 bytes/param: master + m + v)
          - fp16 activations with GC        (2 bytes × n_layers × seq_len × dim)
          - fp16 logits buffer              (2 bytes × batch × seq_len × vocab_size)
          - CUDA context overhead           (300 MB fixed)

        Returns MB.
        """
        params = self.estimate_param_count()

        # Weights (fp16) + grads (fp16) + optimizer (fp32 master+m+v)
        param_mb = params * (2 + 2 + 12) / (1024 ** 2)

        # Activations: with GC only one checkpoint per layer is stored
        # Each checkpoint: batch × seq_len × dim × fp16
        if use_gradient_checkpointing:
            act_mb = (batch_size * seq_len * self.dim * 2 * self.n_layers) / (1024 ** 2)
        else:
            # Without GC all intermediate activations are kept
            act_mb = (batch_size * seq_len * self.dim * 2 * self.n_layers * 4) / (1024 ** 2)

        # Logits buffer: batch × seq_len × vocab_size × fp16
        logit_mb = (batch_size * seq_len * self.vocab_size * 2) / (1024 ** 2)

        # Fixed CUDA overhead
        cuda_overhead_mb = 300.0

        return param_mb + act_mb + logit_mb + cuda_overhead_mb


# ------------------------------------------------------------------ #
# Architecture tiers
#
# Sized for byte-level BPE with vocab ~50k and context up to 4096.
# The embedding table (vocab_size × dim) is the dominant cost at large
# vocab, so dims are chosen to keep total params reasonable on consumer
# GPUs.
#
# Memory budget breakdown for training (fp16 + AdamW, GC on, batch=1):
#   16 bytes/param (weights fp16 + grads fp16 + optimizer fp32 states)
#   + activations (small with GC)
#   + logits buffer (seq_len × vocab × 2 bytes)
#
# Approximate peak VRAM at vocab=50257, seq=2048, batch=1, GC=on:
#   pico   :  ~900 MB   → fits 4GB with headroom for batch > 1
#   micro  : ~1400 MB   → fits 4GB comfortably
#   nano   : ~2200 MB   → fits 4GB tightly (batch=1 only)
#   small  : ~3800 MB   → needs 6GB+
#   medium : ~6500 MB   → needs 8GB+
#   large  : ~12000 MB  → needs 16GB+
#
# 4GB GPU target: pico (primary) + micro (with GC, batch=1)
# ------------------------------------------------------------------ #
TIERS: dict[str, ModelConfig] = {
    "large": ModelConfig(
        dim=1024, n_layers=16, n_heads=16,
        max_seq_len=4096, tier_name="large",
        use_mla=True, kv_lora_rank=128, q_lora_rank=256,
        rope_theta=500_000.0,
    ),
    "medium": ModelConfig(
        dim=768, n_layers=12, n_heads=12,
        max_seq_len=4096, tier_name="medium",
        use_mla=True, kv_lora_rank=96, q_lora_rank=192,
        rope_theta=500_000.0,
    ),
    "small": ModelConfig(
        dim=512, n_layers=10, n_heads=8,
        max_seq_len=2048, tier_name="small",
        use_mla=True, kv_lora_rank=64, q_lora_rank=128,
        rope_theta=500_000.0,
    ),
    "nano": ModelConfig(
        dim=384, n_layers=8, n_heads=8,
        max_seq_len=2048, tier_name="nano",
        use_mla=True, kv_lora_rank=48, q_lora_rank=96,
        rope_theta=500_000.0,
    ),
    "micro": ModelConfig(
        dim=256, n_layers=6, n_heads=8,
        max_seq_len=2048, tier_name="micro",
        use_mla=True, kv_lora_rank=32, q_lora_rank=64,
        rope_theta=500_000.0,
    ),
    # ── 4GB VRAM optimized ──────────────────────────────────────────
    # Design constraints for GTX 1650 Ti / 1660 / RX 580 (4GB):
    #
    #   Total VRAM:          4096 MB
    #   CUDA context:        ~300 MB
    #   Safety margin:        400 MB
    #   Usable budget:       ~3400 MB
    #
    #   At vocab=50257, dim=192:
    #     Embedding (fp16):   50257 × 192 × 2 =  ~19 MB
    #     Optimizer states:   50257 × 192 × 12 = ~110 MB
    #     Transformer layers: 4 layers × small  = ~180 MB
    #     Activations (GC):   2048 × 192 × 4 × 2 = ~3 MB
    #     Logits buffer:      2048 × 50257 × 2  = ~196 MB
    #     Total estimate:     ~900 MB → leaves ~2500 MB for batch > 1
    #
    #   Architectural choices:
    #     - dim=192: divisible by 8 (head_dim=24), keeps embedding small
    #     - 4 layers: enough depth for coherent text, minimal activation memory
    #     - 4 heads: head_dim=48 after MLA split, sufficient for attention quality
    #     - MLA with tiny kv_lora_rank=24: KV cache is 24+6=30 floats/position/layer
    #       vs 2×48=96 for standard MHA — 3× KV cache reduction during inference
    #     - q_lora_rank=48: halves Q projection memory during training
    #     - seq_len=2048: full GPT-2 context, feasible at this dim
    #     - tie_embeddings=True: saves 50257×192×2 = 19MB (the full output head)
    #     - ffn_multiplier=8/3: SwiGLU canonical, hidden=512 (rounded to 512)
    #
    #   Expected params at vocab=50257: ~20M
    #   Expected peak VRAM (batch=1, seq=2048, GC=on): ~900 MB
    #   Expected peak VRAM (batch=4, seq=2048, GC=on): ~1400 MB
    "pico": ModelConfig(
        dim=192,
        n_layers=4,
        n_heads=4,
        ffn_multiplier=8 / 3,
        max_seq_len=2048,
        dropout=0.0,
        tie_embeddings=True,
        rope_theta=500_000.0,
        use_mla=True,
        kv_lora_rank=24,   # KV latent: 24 floats/pos/layer (vs 96 for MHA)
        q_lora_rank=48,    # Q compression: halves Q activation memory
        tier_name="pico",
    ),
}

TIER_ORDER = ["large", "medium", "small", "nano", "micro", "pico"]

# ── 4GB preset: recommended settings for 4GB VRAM GPUs ───────────────
# Use these when calling auto_configure with force_tier="pico" or when
# the auto-selector lands on pico.
VRAM_4GB_PRESET = {
    "tier": "pico",
    "seq_len": 2048,
    "vocab_size": 50_257,
    "use_gradient_checkpointing": True,
    "lr": 3e-4,
    "grad_accum_target": 32,   # effective batch = physical_batch × 32
    "note": (
        "Optimized for 4GB VRAM (GTX 1650 Ti / 1660 / RX 580). "
        "~20M params, 2048 context, byte-level BPE vocab=50257. "
        "Gradient checkpointing mandatory. "
        "Expect batch_size=2-4 with grad_accum=8-16 for effective batch=32."
    ),
}
