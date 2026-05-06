"""
GPU VRAM probing: measure actual memory usage by running dummy forward+backward passes.
This is more reliable than theoretical formulas because it captures cuDNN workspace
and AMP overhead that cannot be calculated from parameter counts alone.
"""
import gc
import logging

import torch

logger = logging.getLogger(__name__)


def get_free_vram_mb() -> float:
    """
    Returns available VRAM in MB after current allocations.
    Returns available RAM (partial) if no CUDA device is present.
    """
    if not torch.cuda.is_available():
        import psutil
        available = psutil.virtual_memory().available / (1024 ** 2)
        logger.info(f"No CUDA device. Available RAM: {available:.0f} MB")
        return available

    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    free = total - reserved  # PyTorch caches memory; reserved - allocated = cache
    logger.info(
        f"GPU {device}: total={total:.0f}MB  allocated={allocated:.0f}MB  "
        f"reserved={reserved:.0f}MB  free_estimate={free:.0f}MB"
    )
    return free


def probe_training_vram_mb(
    config,
    batch_size: int,
    seq_len: int,
    use_gradient_checkpointing: bool = True,
) -> float:
    """
    Build a real NanoLLM, run one forward+backward pass on the target device,
    return peak VRAM consumed in MB. Cleans up completely afterward.

    Returns float('inf') if OOM occurs.
    """
    from nano_llm.model.transformer import NanoLLM

    if not torch.cuda.is_available():
        # On CPU, estimate naively: 16 bytes/param (fp32 model + grad + adam states)
        params = config.estimate_param_count()
        return params * 16 / (1024 ** 2)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    model = None
    try:
        model = NanoLLM(config).to(device)
        model.train()

        # Dummy inputs
        dummy_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        dummy_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(
                dummy_ids,
                targets=dummy_targets,
                use_checkpoint=use_gradient_checkpointing,
            )

        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(loss).backward()

        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return peak - baseline / (1024 ** 2)

    except torch.cuda.OutOfMemoryError:
        return float("inf")
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
