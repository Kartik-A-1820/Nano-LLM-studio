"""Optimizer factory with optional 8-bit Adam (bitsandbytes)."""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with separate param groups for weight decay.
    Embeddings and norm weights are excluded from weight decay.
    Attempts 8-bit Adam if use_8bit=True and bitsandbytes is available.
    """
    # Separate into decay and no-decay groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "norm" in name.lower() or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    logger.info(
        f"Optimizer: {len(decay_params)} decay params, "
        f"{len(no_decay_params)} no-decay params"
    )

    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
            logger.info("Using 8-bit AdamW (bitsandbytes)")
            return opt
        except ImportError:
            logger.warning("bitsandbytes not available. Falling back to standard AdamW.")

    # Use fused AdamW on CUDA for speed
    fused_available = torch.cuda.is_available()
    opt = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused_available,
    )
    logger.info(f"Using AdamW (fused={fused_available})")
    return opt
