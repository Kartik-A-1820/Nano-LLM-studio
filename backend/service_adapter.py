"""Adapters that expose existing app services through a clean API surface."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import UploadFile

from ui.services import (
    data_tab_readiness,
    describe_data_status,
    get_training_readiness,
    inference_ready_message,
    load_corpus_chunks,
    poll_crawl_status,
    poll_training,
    reset_library,
    set_data_chunks,
    start_corpus_training,
    start_crawl_collection,
    start_live_training,
    stop_crawl_collection,
    stop_training,
)
from ui.state import STATE

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("outputs/api_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_dataframe(df) -> list[dict]:
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")
    return []


def _data_store_metrics() -> dict:
    store = STATE.corpus_store
    if store is None:
        corpus_dir = Path("train")
        if (corpus_dir / "corpus.jsonl").exists():
            from nano_llm.data.corpus_store import CorpusStore

            store = CorpusStore(corpus_dir)
            STATE.corpus_store = store

    if store is None:
        return {
            "stored_docs": 0,
            "stored_chars": 0,
            "stored_chars_m": 0.0,
            "jsonl_mb": 0.0,
            "bin_mb": 0.0,
        }

    jsonl_mb = (
        store.jsonl_path.stat().st_size / (1024 ** 2)
        if store.jsonl_path.exists()
        else 0.0
    )
    bin_path = getattr(store, "bin_path", None)
    bin_mb = (
        bin_path.stat().st_size / (1024 ** 2)
        if bin_path is not None and bin_path.exists()
        else 0.0
    )
    return {
        "stored_docs": int(getattr(store, "total_docs", 0)),
        "stored_chars": int(getattr(store, "total_chars", 0)),
        "stored_chars_m": round(float(getattr(store, "total_chars", 0)) / 1_000_000, 3),
        "jsonl_mb": round(jsonl_mb, 2),
        "bin_mb": round(bin_mb, 2),
    }


def _latest_loss(points: list[tuple[int, float]]) -> float | None:
    if not points:
        return None
    return round(float(points[-1][1]), 4)


def _stage_label(state) -> str:
    stage = state.training_stage
    if stage == "tokenizer":
        return "Training vocabulary on your corpus..."
    if stage == "config":
        return "Analyzing hardware and configuring model..."
    if stage == "training":
        ep = state.training_current_epoch
        total = state.training_total_epochs
        return f"Training model — Epoch {ep} of {total}"
    if stage == "done":
        return "Training complete"
    return ""


def _compute_pct(state, logged_steps: int) -> float:
    if state.training_stage == "done":
        return 100.0
    if state.training_stage == "training" and state.training_total_steps > 0:
        return round(min(100.0, logged_steps / state.training_total_steps * 100), 1)
    return 0.0


def get_data_status() -> dict:
    metrics = _data_store_metrics()
    return {
        "readiness": data_tab_readiness(STATE),
        "library_status": describe_data_status(STATE),
        "collector_status": poll_crawl_status(STATE),
        "preview": STATE.data_preview_text,
        "is_collecting": STATE.is_crawl_collecting,
        "active_source": None if STATE.active_data_source == "none" else STATE.active_data_source,
        "prepared_chunks": len(STATE.extracted_texts),
        "has_data": bool(STATE.extracted_texts) or metrics["stored_docs"] > 0,
        **metrics,
    }


async def process_uploads(files: list[UploadFile]) -> dict:
    if not files:
        return {"message": "No files uploaded.", **get_data_status()}

    from nano_llm.data.cleaner import clean_text, split_into_chunks
    from nano_llm.data.extractor import extract_file

    chunks: list[str] = []
    status_lines: list[str] = []

    for upload in files:
        filename = upload.filename or "upload.bin"
        dest = UPLOAD_DIR / filename
        data = await upload.read()
        dest.write_bytes(data)
        try:
            raw = extract_file(dest)
            cleaned = clean_text(raw)
            split = split_into_chunks(cleaned, chunk_size=1024, overlap=64)
            chunks.extend(split)
            status_lines.append(f"{filename}: {len(split)} chunks")
        except Exception as exc:
            logger.exception("API upload extraction failed")
            status_lines.append(f"{filename}: failed ({exc})")

    status = "Uploads processed.\n" + "\n".join(status_lines)
    set_data_chunks(STATE, chunks, source="uploads", status=status)
    return {"message": status, **get_data_status()}


def import_local_library() -> dict:
    from nano_llm.data.cleaner import clean_text, split_into_chunks
    from nano_llm.data.extractor import ingest_folder

    train_dir = Path("train")
    if not train_dir.exists():
        return {"message": "train/ folder not found.", **get_data_status()}

    chunks: list[str] = []
    status_lines: list[str] = []
    for path, raw in ingest_folder(train_dir):
        cleaned = clean_text(raw)
        split = split_into_chunks(cleaned, chunk_size=1024, overlap=64)
        chunks.extend(split)
        status_lines.append(f"{path.name}: {len(split)} chunks")

    status = "Local library imported.\n" + "\n".join(status_lines)
    set_data_chunks(STATE, chunks, source="local library", status=status)
    return {"message": status, **get_data_status()}


def finalize_crawl_dataset() -> dict:
    preview, message = load_corpus_chunks(STATE)
    return {
        "message": message,
        "preview": preview,
        **get_data_status(),
    }


def clear_library() -> dict:
    message = reset_library(STATE)
    return {"message": message, **get_data_status()}


def start_crawl(payload) -> dict:
    message = start_crawl_collection(
        payload.wikipedia,
        payload.gutenberg,
        payload.commoncrawl,
        payload.archive_org,
        payload.rss,
        payload.char_threshold_m,
        STATE,
    )
    return {"message": message, **get_data_status()}


def stop_crawl() -> dict:
    message = stop_crawl_collection(STATE)
    return {"message": message, **get_data_status()}


def get_train_status() -> dict:
    plot, sample, progress = poll_training(STATE)
    loss_points = _serialize_dataframe(plot)
    history = (
        STATE.continuous_loss_history
        if STATE.training_mode == "live"
        else STATE.loss_history
    )
    logged_steps = len(history)
    latest_loss = _latest_loss(history)
    config = STATE.model_config
    return {
        "readiness": get_training_readiness(STATE),
        "progress": progress,
        "sample": sample,
        "loss_points": loss_points,
        "is_training": STATE.is_training,
        "mode": STATE.training_mode,
        "auto_config_summary": STATE.auto_config_summary,
        "latest_checkpoint": (
            str(STATE.latest_checkpoint_path)
            if STATE.latest_checkpoint_path is not None
            else None
        ),
        "logged_steps": logged_steps,
        "latest_loss": latest_loss,
        "current_round": (
            int(STATE.continuous_current_round)
            if STATE.training_mode == "live"
            else None
        ),
        "sample_count": (
            len(STATE.continuous_samples)
            if STATE.training_mode == "live"
            else (1 if sample and sample != "No samples yet." else 0)
        ),
        "has_model": STATE.generator is not None,
        # Stage-level progress fields
        "stage": STATE.training_stage,
        "stage_label": _stage_label(STATE),
        "epoch": STATE.training_current_epoch,
        "total_epochs": STATE.training_total_epochs,
        "total_steps": STATE.training_total_steps,
        "pct": _compute_pct(STATE, logged_steps),
        # Auto-configured model details
        "arch_name": config.tier_name if config is not None else None,
        "param_count_m": (
            round(config.estimate_param_count() / 1e6, 1) if config is not None else None
        ),
        "batch_size": STATE.batch_size,
        "grad_accum_steps": STATE.grad_accum_steps,
        "device": STATE.device,
    }


def start_corpus_train(payload) -> dict:
    try:
        message = start_corpus_training(
            STATE,
            seq_len=payload.seq_len,
            epochs=payload.epochs,
            epoch_mode=payload.epoch_mode,
        )
        return {"message": message, **get_train_status()}
    except Exception as exc:
        logger.exception("Corpus training start failed")
        return {"message": f"Training failed to start: {exc}", **get_train_status()}


def start_live_train(payload) -> dict:
    try:
        message = start_live_training(
            STATE,
            payload.wikipedia,
            payload.gutenberg,
            payload.commoncrawl,
            payload.archive_org,
            payload.rss,
            seq_len=payload.seq_len,
        )
        return {"message": message, **get_train_status()}
    except Exception as exc:
        logger.exception("Live training start failed")
        return {"message": f"Live training failed to start: {exc}", **get_train_status()}


def stop_current_training() -> dict:
    message = stop_training(STATE)
    return {"message": message, **get_train_status()}


def get_inference_status() -> dict:
    checkpoints = list_checkpoints()
    return {
        "readiness": inference_ready_message(STATE),
        "has_model": STATE.generator is not None,
        "latest_checkpoint": (
            str(STATE.latest_checkpoint_path)
            if STATE.latest_checkpoint_path is not None
            else None
        ),
        "available_checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
    }


def chat_once(message: str, temperature: float, max_output_tokens: int) -> dict:
    if STATE.generator is None:
        return {
            "message": "No model available yet. Train one in the Train tab first.",
            "response": "",
            **get_inference_status(),
        }

    response = "".join(
        STATE.generator.stream_generate(
            prompt=message,
            max_new_tokens=int(max_output_tokens),
            temperature=float(temperature),
            top_k=50,
            top_p=0.9,
        )
    )
    return {"message": "ok", "response": response, **get_inference_status()}


def list_checkpoints() -> list[str]:
    ckpt_dir = Path("outputs/checkpoints")
    if not ckpt_dir.exists():
        return []
    ordered = [ckpt_dir / "last.pt", ckpt_dir / "best.pt"]
    return [str(path) for path in ordered if path.exists()]


def load_checkpoint(path: str) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {"message": f"Checkpoint not found: {checkpoint_path}", **get_inference_status()}
    if STATE.model is None or STATE.model_config is None or STATE.tokenizer is None:
        return {
            "message": "Train a model first so the architecture is available for loading.",
            **get_inference_status(),
        }

    try:
        import torch
        from nano_llm.inference.generator import Generator

        ckpt = torch.load(checkpoint_path, map_location=STATE.device)
        STATE.model.load_state_dict(ckpt["model_state"])
        STATE.model.to(STATE.device)
        STATE.generator = Generator(STATE.model, STATE.tokenizer, STATE.device)
        STATE.latest_checkpoint_path = checkpoint_path
        return {
            "message": f"Checkpoint loaded (epoch {ckpt.get('epoch', '?')}). Model is ready.",
            **get_inference_status(),
        }
    except Exception as exc:
        logger.exception("API checkpoint load failed")
        return {"message": f"Error loading checkpoint: {exc}", **get_inference_status()}
