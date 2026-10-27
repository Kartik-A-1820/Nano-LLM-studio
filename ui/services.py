"""App-facing orchestration helpers for the simplified 3-tab UI."""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import pandas as pd

from ui.state import AppState

logger = logging.getLogger(__name__)

TRAIN_DIR = Path("train")
OUTPUTS_DIR = Path("outputs")
DEFAULT_VOCAB_SIZE = 50_257
DEFAULT_SEQ_LEN = 2048
DEFAULT_EPOCHS = 1
DEFAULT_MAX_AUTO_EPOCHS = 200
DEFAULT_LR = 3e-4
DEFAULT_SAMPLE_PROMPT = "Write a short answer:"
DEFAULT_STANDARD_STEPS_PER_EPOCH = 80
DEFAULT_LIVE_STEPS_PER_EPOCH = 40
DEFAULT_STANDARD_EARLY_STOP_PATIENCE = 4
DEFAULT_STANDARD_EARLY_STOP_DELTA = 0.002
FREE_NO_KEY_SOURCES = {
    "wikipedia",
    "gutenberg",
    "commoncrawl",
    "archive_org",
    "rss",
}

VOCAB_SIZE_LIMIT_RE = re.compile(r"value\s*<=\s*(\d+)", re.IGNORECASE)


def make_loss_plot(history: list[tuple[int, float]]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame({"step": [], "loss": []})
    steps, losses = zip(*history)
    return pd.DataFrame({"step": list(steps), "loss": list(losses)})


def ensure_corpus_store(state: AppState):
    from nano_llm.data.corpus_store import CorpusStore

    if state.corpus_store is None:
        state.corpus_store = CorpusStore(TRAIN_DIR)
    return state.corpus_store


def _sanitize_sources(selected: list[str]) -> list[str]:
    sources = [name for name in selected if name in FREE_NO_KEY_SOURCES]
    return sources or ["wikipedia", "gutenberg"]


def describe_data_status(state: AppState) -> str:
    lines: list[str] = []
    if state.active_data_source != "none":
        lines.append(f"Source: {state.active_data_source}")
    if state.extracted_texts:
        lines.append(f"Prepared chunks: {len(state.extracted_texts):,}")
    store = state.corpus_store
    if store is not None:
        lines.append(store.summary())
    elif (TRAIN_DIR / "corpus.jsonl").exists():
        store = ensure_corpus_store(state)
        lines.append(store.summary())
    return "\n".join(lines) if lines else "Your library is empty."


def data_tab_readiness(state: AppState) -> str:
    """Plain-language readiness string for the Data tab."""
    n = len(state.extracted_texts)
    if n > 0:
        return f"Ready for training — {n:,} text chunks in the library."
    store = state.corpus_store
    if store is None and (TRAIN_DIR / "corpus.jsonl").exists():
        store = ensure_corpus_store(state)
    if store is not None and store.total_docs > 0:
        return (
            f"Web Collector has gathered {store.total_docs:,} docs "
            f"({store.total_chars/1e6:.1f}M chars). "
            "Click Finalize Dataset to prepare for training."
        )
    return "Your library is empty. Upload documents or start the Web Collector."


def data_ready(state: AppState) -> bool:
    if state.extracted_texts:
        return True
    store = state.corpus_store
    if store is not None and store.total_docs > 0:
        return True
    return (TRAIN_DIR / "corpus.jsonl").exists()


def get_training_readiness(state: AppState) -> str:
    if state.is_training:
        return state.training_status_text
    if state.model is not None and state.generator is not None:
        return "Training complete. Open the Inference tab to chat with your model."
    if state.model is not None:
        return "Model trained. Finalizing inference setup..."
    if state.tokenizer is not None:
        return "Library is ready. Click Start Training to begin."
    if data_ready(state):
        return "Library is ready. Click Start Training — vocabulary will be prepared automatically."
    return "No data loaded yet. Add data in the Data tab first."


def set_data_chunks(state: AppState, texts: list[str], source: str, status: str) -> None:
    state.mark_data_updated(texts, source=source, status=status)
    state.data_status_text = f"{status}\n\n{describe_data_status(state)}"


def load_corpus_chunks(state: AppState) -> tuple[str, str]:
    store = ensure_corpus_store(state)
    if not store.jsonl_path.exists():
        return "", "No train/corpus.jsonl found yet."
    chunks = store.load_chunks(chunk_size=1024, overlap=64)
    status = (
        "Loaded corpus from crawler store.\n"
        f"Documents: {store.total_docs:,}\n"
        f"Characters: {store.total_chars:,}\n"
        f"Prepared chunks: {len(chunks):,}"
    )
    set_data_chunks(state, chunks, source="crawler corpus", status=status)
    return state.data_preview_text, state.data_status_text


def reset_library(state: AppState) -> str:
    store = ensure_corpus_store(state)
    store.clear()
    state.extracted_texts = []
    state.data_preview_text = ""
    state.data_status_text = "Your library is empty."
    state.active_data_source = "none"
    state.data_revision += 1
    state.tokenizer = None
    state.tokenizer_path = None
    state.tokenizer_revision = -1
    state.model = None
    state.model_config = None
    state.generator = None
    state.latest_checkpoint_path = None
    state.training_mode = "idle"
    state.training_status_text = "Idle."
    return "Library cleared. Add files or start the Web Collector."


def start_crawl_collection(
    wiki: bool,
    gutenberg: bool,
    commoncrawl: bool,
    archive_org: bool,
    rss: bool,
    char_threshold_m: float,
    state: AppState,
) -> str:
    if state.is_crawl_collecting:
        return "Crawl collection is already running."

    from nano_llm.data.cleaner import clean_text
    from nano_llm.data.corpus_store import CorpusStore
    from nano_llm.data.crawler import CrawlerConfig, MultiCrawler

    sources = _sanitize_sources([
        name
        for enabled, name in [
            (wiki, "wikipedia"),
            (gutenberg, "gutenberg"),
            (commoncrawl, "commoncrawl"),
            (archive_org, "archive_org"),
            (rss, "rss"),
        ]
        if enabled
    ])

    threshold = int(char_threshold_m * 1_000_000)
    store = CorpusStore(TRAIN_DIR)
    state.corpus_store = store
    multi = MultiCrawler(CrawlerConfig(sources=sources, buffer_size=100))
    multi.start()
    state.crawl_stop_event = threading.Event()

    def _run():
        state.is_crawl_collecting = True
        try:
            while not state.crawl_stop_event.is_set():
                docs = multi.drain(10, timeout=15)
                if not docs:
                    continue
                for title, text in docs:
                    if state.crawl_stop_event.is_set():
                        break
                    cleaned = clean_text(text)
                    if len(cleaned) > 200:
                        store.append(source="crawler", title=title, text=cleaned)
                if store.total_chars >= threshold:
                    break
        finally:
            multi.stop()
            state.is_crawl_collecting = False
            state.data_status_text = describe_data_status(state)

    state.crawl_thread = threading.Thread(target=_run, daemon=True)
    state.crawl_thread.start()
    return (
        f"Crawl started using: {', '.join(sources)}\n"
        f"Target size: {char_threshold_m:.1f}M characters\n"
        "Store: train/corpus.jsonl"
    )


def stop_crawl_collection(state: AppState) -> str:
    if state.crawl_stop_event is None:
        return "No crawl is running."
    state.crawl_stop_event.set()
    return "Stop signal sent to crawlers."


def poll_crawl_status(state: AppState) -> str:
    store = state.corpus_store
    if store is None and (TRAIN_DIR / "corpus.jsonl").exists():
        store = ensure_corpus_store(state)
    if store is None:
        return "No corpus store yet."
    prefix = "Collecting" if state.is_crawl_collecting else "Stopped"
    return f"{prefix} | {store.summary()}"


def _load_training_chunks(state: AppState) -> list[str]:
    if state.extracted_texts:
        return state.extracted_texts

    store = ensure_corpus_store(state)
    if store.jsonl_path.exists():
        chunks = store.load_chunks(chunk_size=1024, overlap=64)
        if chunks:
            set_data_chunks(
                state,
                chunks,
                source="crawler corpus",
                status=(
                    "Prepared training data from crawler corpus.\n"
                    f"Documents: {store.total_docs:,}\n"
                    f"Characters: {store.total_chars:,}"
                ),
            )
            return chunks
    return []


def _retry_vocab_size_from_error(exc: Exception) -> int | None:
    match = VOCAB_SIZE_LIMIT_RE.search(str(exc))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def ensure_tokenizer_ready(state: AppState, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
    if state.tokenizer is not None and state.tokenizer_revision == state.data_revision:
        return

    texts = _load_training_chunks(state)
    if not texts:
        raise ValueError("No training data available. Add or load data first.")

    from nano_llm.tokenizer.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    try:
        tok.train(texts=texts, vocab_size=vocab_size, model_prefix="outputs/tokenizer/bpe")
    except RuntimeError as exc:
        fallback_vocab = _retry_vocab_size_from_error(exc)
        if fallback_vocab is None:
            raise
        logger.warning(
            "Tokenizer vocab target %s is too high for current corpus. Retrying with %s.",
            vocab_size,
            fallback_vocab,
        )
        tok.train(texts=texts, vocab_size=fallback_vocab, model_prefix="outputs/tokenizer/bpe")
    tok.save_pretrained(Path("outputs/tokenizer"))
    state.tokenizer = tok
    state.tokenizer_path = Path("outputs/tokenizer")
    state.tokenizer_revision = state.data_revision


def auto_prepare_model(state: AppState, seq_len: int = DEFAULT_SEQ_LEN) -> str:
    state.training_status_text = "Step 1/3: Preparing vocabulary..."
    ensure_tokenizer_ready(state)

    state.training_status_text = "Step 2/3: Analyzing hardware (RAM/VRAM)..."
    import torch
    from nano_llm.memory.auto_arch import auto_configure

    config, batch_size, grad_accum = auto_configure(
        seq_len=seq_len,
        use_gradient_checkpointing=True,
        vocab_size=state.tokenizer.vocab_size(),
        max_tier="small",
    )
    state.model_config = config
    state.batch_size = batch_size
    state.grad_accum_steps = grad_accum
    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    summary = (
        f"Success: Selected {config.tier_name} architecture.\n"
        f"Details: ~{config.estimate_param_count()/1e6:.1f}M parameters.\n"
        f"Context: {seq_len} tokens.\n"
        f"Optimal batch: {batch_size} x {grad_accum} accumulation steps.\n"
        f"Device: {state.device.upper()}"
    )
    state.auto_config_summary = summary
    state.training_status_text = f"Step 3/3: Model architecture ready.\n\n{summary}"
    return summary


def _update_latest_checkpoint(state: AppState) -> None:
    ckpt_dir = OUTPUTS_DIR / "checkpoints"
    if not ckpt_dir.exists():
        return
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    if last_path.exists():
        state.latest_checkpoint_path = last_path
    elif best_path.exists():
        state.latest_checkpoint_path = best_path


def start_corpus_training(
    state: AppState,
    seq_len: int = DEFAULT_SEQ_LEN,
    epochs: int = DEFAULT_EPOCHS,
    epoch_mode: str = "custom",
    sample_prompt: str = DEFAULT_SAMPLE_PROMPT,
) -> str:
    if state.is_training:
        return "A training run is already in progress."

    texts = _load_training_chunks(state)
    if not texts:
        return "No training data available."

    effective_epochs = DEFAULT_MAX_AUTO_EPOCHS if epoch_mode == "auto" else int(epochs)

    # Reset state and configure stage metadata before spawning so UI sees it immediately
    state.reset_manual_training()
    state.training_mode = "corpus"
    state.training_stage = "tokenizer"
    state.training_total_epochs = effective_epochs
    state.training_total_steps = effective_epochs * DEFAULT_STANDARD_STEPS_PER_EPOCH
    state.training_status_text = "Starting training pipeline..."

    def _run():
        try:
            # Stage 1: tokenizer
            state.training_stage = "tokenizer"
            state.training_status_text = "Preparing vocabulary..."
            ensure_tokenizer_ready(state)
            if state.training_abort:
                state.training_stage = "idle"
                state.training_status_text = "Training stopped before model configuration."
                return

            # Stage 2: hardware probe + model config
            state.training_stage = "config"
            state.training_status_text = "Analyzing hardware and configuring model..."
            import torch
            from nano_llm.memory.auto_arch import auto_configure

            config, batch_size, grad_accum = auto_configure(
                seq_len=seq_len,
                use_gradient_checkpointing=True,
                vocab_size=state.tokenizer.vocab_size(),
                max_tier="small",
            )
            state.model_config = config
            state.batch_size = batch_size
            state.grad_accum_steps = grad_accum
            state.device = "cuda" if torch.cuda.is_available() else "cpu"
            state.auto_config_summary = (
                f"Success: Selected {config.tier_name} architecture.\n"
                f"Details: ~{config.estimate_param_count()/1e6:.1f}M parameters.\n"
                f"Context: {seq_len} tokens.\n"
                f"Optimal batch: {batch_size} x {grad_accum} accumulation steps.\n"
                f"Device: {state.device.upper()}"
            )
            state.training_status_text = "Model configured. Building dataset..."
            if state.training_abort:
                state.training_stage = "idle"
                state.training_status_text = "Training stopped before dataset build."
                return

            # Build dataset and model (still "config" stage visually)
            from nano_llm.data.dataset import build_train_validation_loaders, build_dataset
            from nano_llm.model.transformer import NanoLLM
            from nano_llm.training.callbacks import (
                BaseCallback,
                GradioUpdateCallback,
                LossLoggerCallback,
                SampleGenerationCallback,
            )
            from nano_llm.training.trainer import Trainer, TrainerConfig

            dataset_bin = Path("outputs") / f"train_data_{state.data_revision}_{int(time.time())}.bin"
            dataset = build_dataset(texts, state.tokenizer, int(seq_len), dataset_bin)
            if len(dataset) == 0:
                state.training_stage = "idle"
                state.training_status_text = "Prepared dataset is empty. Add more data."
                return

            dataloader, validation_loader = build_train_validation_loaders(dataset, state.batch_size)
            state.model_config.vocab_size = state.tokenizer.vocab_size()
            state.model = NanoLLM.from_config(state.model_config)

            gradio_cb = GradioUpdateCallback()
            sample_cb = SampleGenerationCallback(every_k_epochs=max(1, min(effective_epochs, 4)))
            log_cb = LossLoggerCallback()

            class StatusUpdateCallback(BaseCallback):
                def on_train_start(self, trainer) -> None:
                    state.training_stage = "training"
                    state.training_status_text = "Training model..."

                def on_epoch_end(self, epoch: int, mean_loss: float, trainer) -> None:
                    state.training_current_epoch = epoch
                    state.training_status_text = (
                        f"Running: Epoch {epoch}/{effective_epochs} | Train: {mean_loss:.4f} | Validation: {trainer.validation_loss:.4f}"
                    )

                def on_train_end(self, trainer) -> None:
                    state.training_stage = "done"
                    if epoch_mode == "auto":
                        state.training_status_text = "Training complete. Auto mode stopped on plateau or limit."
                    else:
                        state.training_status_text = f"Training complete. Finished {effective_epochs} epochs."

            status_cb = StatusUpdateCallback()

            trainer = Trainer(
                model=state.model,
                dataloader=dataloader,
                config=TrainerConfig(
                    epochs=effective_epochs,
                    steps_per_epoch=DEFAULT_STANDARD_STEPS_PER_EPOCH,
                    early_stopping_patience=DEFAULT_STANDARD_EARLY_STOP_PATIENCE if epoch_mode == "auto" else None,
                    min_loss_delta=DEFAULT_STANDARD_EARLY_STOP_DELTA if epoch_mode == "auto" else 0.0,
                    lr=DEFAULT_LR,
                    use_gradient_checkpointing=True,
                    grad_accum_steps=state.grad_accum_steps,
                    sample_prompt=sample_prompt,
                    sample_every_n_epochs=max(1, min(int(epochs), 4)) if epoch_mode != "auto" else 4,
                ),
                tokenizer=state.tokenizer,
                validation_dataloader=validation_loader,
                callbacks=[gradio_cb, sample_cb, log_cb, status_cb],
                device=state.device,
            )

            state.gradio_callback = gradio_cb
            state.sample_callback = sample_cb
            state.trainer = trainer

            # Stage 3: model training
            state.training_stage = "training"
            state.training_status_text = f"Training model. Goal: {effective_epochs} epochs."
            trainer.train()
        except Exception as e:
            logger.exception("Training thread failed")
            state.training_stage = "idle"
            state.training_status_text = f"Training failed: {e}"
        finally:
            state.is_training = False
            if state.model is not None:
                from nano_llm.inference.generator import Generator

                state.generator = Generator(state.model, state.tokenizer, state.device)
            _update_latest_checkpoint(state)

    # Set is_training before the thread so get_train_status immediately reflects it
    state.is_training = True
    state.training_thread = threading.Thread(target=_run, daemon=True)
    state.training_thread.start()

    epoch_text = (
        f"Auto mode: up to {DEFAULT_MAX_AUTO_EPOCHS} epochs with early stopping."
        if epoch_mode == "auto"
        else f"Epochs: {epochs}"
    )
    return (
        f"Training started. {epoch_text}\n"
        f"Training budget: {DEFAULT_STANDARD_STEPS_PER_EPOCH} steps per epoch"
    )


def start_live_training(
    state: AppState,
    wiki: bool,
    gutenberg: bool,
    commoncrawl: bool,
    archive_org: bool,
    rss: bool,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> str:
    if state.is_training:
        return "A training run is already in progress."

    from nano_llm.training.continuous import ContinuousConfig, ContinuousTrainer

    sources = _sanitize_sources([
        name
        for enabled, name in [
            (wiki, "wikipedia"),
            (gutenberg, "gutenberg"),
            (commoncrawl, "commoncrawl"),
            (archive_org, "archive_org"),
            (rss, "rss"),
        ]
        if enabled
    ])

    cfg = ContinuousConfig(
        sources=sources,
        docs_per_round=40,
        crawler_buffer=200,
        epochs_per_round=4,
        steps_per_epoch=DEFAULT_LIVE_STEPS_PER_EPOCH,
        lr=DEFAULT_LR,
        seq_len=int(seq_len),
        use_gradient_checkpointing=True,
        patience=3,
        max_rounds=0,
        corpus_window_rounds=3,
        sample_prompt=DEFAULT_SAMPLE_PROMPT,
        retrain_tokenizer_every_n_rounds=5,
        min_chars_for_retrain=5_000_000,
        tokenizer_vocab_size=DEFAULT_VOCAB_SIZE,
    )

    def _on_step(step: int, loss: float):
        state.continuous_loss_history.append((step, loss))

    def _on_sample(round_num: int, epoch: int, loss: float, text: str):
        state.continuous_samples.append(
            {"round": round_num, "epoch": epoch, "loss": round(loss, 4), "sample": text}
        )

    def _on_round_start(round_num: int):
        state.continuous_current_round = round_num

    def _on_tok_retrain(round_num: int, old_vocab: int, new_vocab: int):
        state.continuous_tok_retrains.append(
            {"round": round_num, "old_vocab": old_vocab, "new_vocab": new_vocab}
        )

    def _on_model_ready(ct):
        state.model = ct.model
        state.tokenizer = ct.tokenizer
        state.device = ct.device
        state.tokenizer_revision = state.data_revision

    state.reset_live_training()
    ct = ContinuousTrainer(
        model=state.model,
        tokenizer=state.tokenizer,
        config=cfg,
        device=state.device,
        on_step=_on_step,
        on_sample=_on_sample,
        on_round_start=_on_round_start,
        on_tokenizer_retrain=_on_tok_retrain,
        on_model_ready=_on_model_ready,
    )
    state.continuous_trainer = ct
    state.training_mode = "live"
    state.training_status_text = "Starting live training..."

    def _run():
        state.is_training = True
        try:
            ct.run()
        finally:
            state.is_training = False
            if state.model is not None and state.tokenizer is not None:
                from nano_llm.inference.generator import Generator

                state.generator = Generator(state.model, state.tokenizer, state.device)
            _update_latest_checkpoint(state)
            state.training_status_text = "Live training stopped."

    state.continuous_thread = threading.Thread(target=_run, daemon=True)
    state.continuous_thread.start()
    return (
        "Live training started.\n"
        f"Sources: {', '.join(sources)}\n"
        f"Training budget: {DEFAULT_LIVE_STEPS_PER_EPOCH} steps per epoch\n"
        "Live data is stored as rolling text shards on disk (3-round window).\n"
        "Architecture and tokenizer will be prepared automatically."
    )


def stop_training(state: AppState) -> str:
    if state.training_mode == "corpus" and state.is_training:
        state.training_abort = True
        if state.trainer is not None:
            state.trainer.stop()
        state.training_status_text = "Stopping training..."
        return state.training_status_text
    if state.training_mode == "live" and state.continuous_trainer is not None and state.is_training:
        state.continuous_trainer.stop()
        state.training_status_text = "Stopping live training..."
        return state.training_status_text
    return "No training is running."


def poll_training(state: AppState) -> tuple[pd.DataFrame, str, str]:
    if state.training_mode == "live":
        plot = make_loss_plot(state.continuous_loss_history[-1000:])
        samples = state.continuous_samples[-3:][::-1]
        if samples:
            sample_text = "\n\n".join(
                f"Round {s['round']} | Epoch {s['epoch']} | Loss {s['loss']}\n{s['sample']}"
                for s in samples
            )
        else:
            sample_text = "No live samples yet."
        trainer_status = (
            state.continuous_trainer.get_status()
            if state.continuous_trainer is not None
            else "idle"
        )
        status = (
            f"{'Running' if state.is_training else 'Stopped'} | "
            f"Mode: live | Round: {state.continuous_current_round} | {trainer_status}"
        )
        return plot, sample_text, status

    if state.gradio_callback is not None:
        state.loss_history.extend(state.gradio_callback.get_updates())
    plot = make_loss_plot(state.loss_history[-500:])
    sample_text = state.sample_callback.get_sample() if state.sample_callback is not None else "No samples yet."
    status = (
        f"{'Running' if state.is_training else 'Stopped'} | "
        f"Mode: corpus | Logged steps: {len(state.loss_history)}"
    )
    return plot, sample_text, status


def inference_ready_message(state: AppState) -> str:
    if state.generator is not None:
        return "Model is ready! Type a message below to start chatting."
    if state.is_training:
        return "Training in progress — the model will appear here automatically when done."
    if state.latest_checkpoint_path is not None:
        return f"Checkpoint available: {state.latest_checkpoint_path.name}. Use 'Open Saved Model' above to load it."
    return "No model available yet. Train one in the Train tab, then come back here."
