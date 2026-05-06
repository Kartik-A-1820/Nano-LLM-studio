"""
Auto architecture and batch size selection.
Probes GPU memory with real model forward+backward passes to find the
largest architecture that fits, then binary-searches for max batch size.

For 4GB VRAM GPUs the analytical pre-filter in ModelConfig.estimate_training_vram_mb
is used to skip obviously-too-large tiers before running the expensive GPU probe.
"""
import copy
import logging

from .probe import get_free_vram_mb, probe_training_vram_mb
from nano_llm.model.config import ModelConfig, TIERS, TIER_ORDER

logger = logging.getLogger(__name__)

# Headroom reserved for CUDA context, driver, and OS overhead.
# 4GB cards have less slack so we use a tighter margin there.
VRAM_SAFETY_MARGIN_MB = 400
VRAM_4GB_THRESHOLD_MB = 4500   # cards with total VRAM <= this get tighter margin
VRAM_4GB_SAFETY_MB    = 350    # slightly tighter — driver overhead is more predictable

MAX_BATCH_SEARCH = 32
DEFAULT_MAX_AUTO_TIER = "small"


def _safety_margin(total_vram_mb: float) -> float:
    """Return the appropriate safety margin for the detected GPU."""
    if total_vram_mb <= VRAM_4GB_THRESHOLD_MB:
        return VRAM_4GB_SAFETY_MB
    return VRAM_SAFETY_MARGIN_MB


def select_tier(
    available_vram_mb: float,
    seq_len: int = 2048,
    use_gradient_checkpointing: bool = True,
    vocab_size: int = 50_257,
    total_vram_mb: float = 0.0,
    max_tier: str | None = DEFAULT_MAX_AUTO_TIER,
) -> tuple[ModelConfig, str]:
    """
    Iterate tiers from smaller to larger within the allowed range.
    Uses a fast analytical pre-filter to skip obviously-too-large tiers,
    then confirms with a real GPU probe.
    Returns (config, tier_name) for the first tier that fits.
    Raises RuntimeError if even pico doesn't fit.
    """
    margin = _safety_margin(total_vram_mb or available_vram_mb)
    budget = available_vram_mb - margin
    logger.info(
        f"VRAM budget: {budget:.0f} MB "
        f"(available={available_vram_mb:.0f}, margin={margin:.0f})"
    )

    tier_candidates = TIER_ORDER
    if max_tier is not None and max_tier in TIER_ORDER:
        tier_candidates = TIER_ORDER[TIER_ORDER.index(max_tier):]
    tier_candidates = list(reversed(tier_candidates))

    for tier_name in tier_candidates:
        config = copy.deepcopy(TIERS[tier_name])
        config.vocab_size = vocab_size
        config.max_seq_len = max(seq_len, config.max_seq_len)

        # Fast analytical pre-filter — skip GPU probe if clearly too large
        analytical_mb = config.estimate_training_vram_mb(
            batch_size=1,
            seq_len=seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        if analytical_mb > budget * 1.5:   # 50% slack for estimation error
            logger.info(
                f"  Skip '{tier_name}': analytical estimate {analytical_mb:.0f} MB "
                f">> budget {budget:.0f} MB"
            )
            continue

        estimated_params = config.estimate_param_count()
        logger.info(
            f"Probing tier '{tier_name}': ~{estimated_params/1e6:.1f}M params, "
            f"analytical={analytical_mb:.0f} MB, seq_len={seq_len}"
        )

        peak_mb = probe_training_vram_mb(
            config, batch_size=1, seq_len=seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        logger.info(
            f"  Peak VRAM (batch=1): {peak_mb:.0f} MB  "
            f"(budget: {budget:.0f} MB)"
        )

        if peak_mb <= budget:
            logger.info(f"  ✓ Selected tier: '{tier_name}'")
            return config, tier_name

    raise RuntimeError(
        f"No architecture tier fits in {available_vram_mb:.0f} MB VRAM. "
        "Try reducing seq_len, or run on CPU (slow but no VRAM limit)."
    )


def find_max_batch_size(
    config: ModelConfig,
    available_vram_mb: float,
    seq_len: int,
    use_gradient_checkpointing: bool = True,
    total_vram_mb: float = 0.0,
) -> int:
    """
    Binary search for the largest batch_size that fits in VRAM budget.
    Returns at least 1.
    """
    margin = _safety_margin(total_vram_mb or available_vram_mb)
    budget = available_vram_mb - margin
    lo, hi = 1, MAX_BATCH_SEARCH

    # Fast analytical upper bound — no point probing beyond what fits analytically
    for hi_candidate in range(MAX_BATCH_SEARCH, 0, -1):
        est = config.estimate_training_vram_mb(hi_candidate, seq_len, use_gradient_checkpointing)
        if est <= budget:
            hi = hi_candidate
            break
    else:
        hi = 1

    if lo == hi:
        return 1

    # Confirm hi with real probe
    peak_hi = probe_training_vram_mb(config, hi, seq_len, use_gradient_checkpointing)
    if peak_hi <= budget:
        logger.info(f"Max batch size: {hi}")
        return hi

    # Binary search between lo and hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        peak = probe_training_vram_mb(config, mid, seq_len, use_gradient_checkpointing)
        logger.info(f"  Batch {mid}: peak={peak:.0f} MB, budget={budget:.0f} MB")
        if peak <= budget:
            lo = mid
        else:
            hi = mid - 1

    logger.info(f"Max batch size: {lo}")
    return lo


def auto_configure(
    seq_len: int = 2048,
    use_gradient_checkpointing: bool = True,
    vocab_size: int = 50_257,
    force_tier: str | None = None,
    target_effective_batch: int = 32,
    max_tier: str | None = DEFAULT_MAX_AUTO_TIER,
) -> tuple[ModelConfig, int, int]:
    """
    Full auto-configuration:
    1. Query free + total VRAM
    2. Select best-fitting architecture tier (or use force_tier)
    3. Find max physical batch size
    4. Compute grad_accum_steps to reach target_effective_batch

    target_effective_batch raised to 32 (from 16) — better gradient
    estimates, especially important for small physical batches on 4GB cards.

    Returns:
        (config, batch_size, grad_accum_steps)
    """
    import torch
    free_mb  = get_free_vram_mb()
    total_mb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        if torch.cuda.is_available() else free_mb
    )

    if force_tier is not None:
        config = copy.deepcopy(TIERS[force_tier])
        config.vocab_size = vocab_size
        config.max_seq_len = max(seq_len, config.max_seq_len)
        tier_name = force_tier
    else:
        config, tier_name = select_tier(
            free_mb, seq_len, use_gradient_checkpointing, vocab_size,
            total_vram_mb=total_mb,
            max_tier=max_tier,
        )

    batch_size = find_max_batch_size(
        config, free_mb, seq_len, use_gradient_checkpointing,
        total_vram_mb=total_mb,
    )

    grad_accum = max(1, target_effective_batch // batch_size)

    analytical_mb = config.estimate_training_vram_mb(batch_size, seq_len, use_gradient_checkpointing)

    logger.info(
        f"\n{'='*54}\n"
        f"Auto-Config Result\n"
        f"  GPU VRAM:          {total_mb:.0f} MB total / {free_mb:.0f} MB free\n"
        f"  Tier:              {tier_name}\n"
        f"  Params:            ~{config.estimate_param_count()/1e6:.1f}M\n"
        f"  dim / layers:      {config.dim} / {config.n_layers}\n"
        f"  n_heads:           {config.n_heads}\n"
        f"  vocab_size:        {config.vocab_size:,}\n"
        f"  max_seq_len:       {config.max_seq_len:,}\n"
        f"  Batch size:        {batch_size}\n"
        f"  Grad accum steps:  {grad_accum}\n"
        f"  Effective batch:   {batch_size * grad_accum}\n"
        f"  seq_len:           {seq_len}\n"
        f"  Est. VRAM usage:   ~{analytical_mb:.0f} MB\n"
        f"{'='*54}"
    )

    return config, batch_size, grad_accum
