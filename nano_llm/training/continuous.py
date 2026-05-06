"""
Continuous training loop with parallel pre-fetching.

Architecture
────────────
  MultiCrawler (N daemon threads, one per source)
       │  fills a shared queue while training runs
       ▼
  ContinuousTrainer (one daemon thread)
       │  drains queue → builds dataset → trains N epochs
       │  generates sample → checks auto-stop heuristic
       └─ repeat

Key insight: each training round takes several minutes, so the crawlers
have plenty of time to pre-fill the buffer before the next round starts.
Rate limits per source are respected independently; the trainer never
waits for the network.

Auto-stop heuristic
───────────────────
  After every round a sample is generated and the mean epoch loss is
  recorded.  If loss has not improved by more than `min_loss_delta` for
  `patience` consecutive rounds the loop stops automatically.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContinuousConfig:
    # ── Crawler ──────────────────────────────────────────────────────
    sources: list[str] = field(
        default_factory=lambda: ["wikipedia", "gutenberg", "archive_org", "rss"]
    )
    rate_limits: dict[str, float] = field(default_factory=dict)
    docs_per_round: int = 40
    crawler_buffer: int = 200
    rss_feeds: list[str] = field(default_factory=list)

    # ── Training ─────────────────────────────────────────────────────
    epochs_per_round: int = 5
    steps_per_epoch: int = 40
    lr: float = 3e-4
    seq_len: int = 2048
    use_gradient_checkpointing: bool = True
    sample_prompt: str = "Once upon a time"
    corpus_window_rounds: int = 3

    # ── Tokenizer retraining ──────────────────────────────────────────
    # Retrain BPE once the corpus has accumulated enough characters.
    # After retraining, the model's embedding / head are resized to match
    # the new vocab and training continues — existing token weights are
    # preserved, new tokens are randomly initialised.
    #
    # Set retrain_tokenizer_every_n_rounds=0 to disable auto-retraining.
    retrain_tokenizer_every_n_rounds: int = 5   # 0 = disabled
    min_chars_for_retrain: int = 5_000_000      # ~5M chars ≈ decent BPE corpus
    tokenizer_vocab_size: int = 50_257          # target vocab after retrain

    # ── Auto-stop ────────────────────────────────────────────────────
    patience: int = 3
    min_loss_delta: float = 0.005
    max_rounds: int = 0

    # ── Paths ────────────────────────────────────────────────────────
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    data_cache_dir: Path = field(default_factory=lambda: Path("outputs/crawl_cache"))
    live_chunk_dir: Path = field(default_factory=lambda: Path("outputs/live_chunks"))
    tokenizer_dir:  Path = field(default_factory=lambda: Path("outputs/tokenizer"))


class ContinuousTrainer:
    """
    Orchestrates: start crawlers → (crawl ‖ train) → evaluate → repeat.
    """

    def __init__(
        self,
        model,           # NanoLLM instance, or None to bootstrap from crawled data
        tokenizer,       # ByteLevelBPETokenizer, or None to bootstrap from crawled data
        config: ContinuousConfig,
        device: str = "cpu",
        on_sample: Optional[Callable[[int, int, float, str], None]] = None,
        on_round_start: Optional[Callable[[int], None]] = None,
        on_step: Optional[Callable[[int, float], None]] = None,
        on_tokenizer_retrain: Optional[Callable[[int, int, int], None]] = None,
        on_model_ready: Optional[Callable[["ContinuousTrainer"], None]] = None,
    ):
        """
        Args:
            model:            Pre-built NanoLLM, or None — will be auto-built after
                              the first crawl batch trains the tokenizer.
            tokenizer:        Pre-trained tokenizer, or None — will be trained on
                              the first crawl batch before model construction.
            on_tokenizer_retrain: callback(round, old_vocab, new_vocab)
            on_model_ready:   callback(trainer) — fired after bootstrap model is built,
                              so the UI can store the model/tokenizer references.
        """
        self.model     = model
        self.tokenizer = tokenizer
        self.config    = config
        self.device    = device

        self.on_sample            = on_sample
        self.on_round_start       = on_round_start
        self.on_step              = on_step
        self.on_tokenizer_retrain = on_tokenizer_retrain
        self.on_model_ready       = on_model_ready

        self._stop_flag = threading.Event()
        self.global_step  = 0
        self.global_epoch = 0
        self.current_round = 0

        self.loss_history: list[float] = []
        self.last_sample: str = ""
        self.status: str = "idle"
        self._lock = threading.Lock()

        # Total characters seen — used to decide when to retrain tokenizer
        self._total_chars: int = 0
        # Round of last tokenizer retrain
        self._last_tok_retrain_round: int = 0

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.data_cache_dir.mkdir(parents=True, exist_ok=True)
        config.live_chunk_dir.mkdir(parents=True, exist_ok=True)
        config.tokenizer_dir.mkdir(parents=True, exist_ok=True)

        from nano_llm.data.corpus_store import ChunkedTextWindowStore

        self._window_store = ChunkedTextWindowStore(
            config.live_chunk_dir,
            max_shards=config.corpus_window_rounds,
        )
        self._window_store.clear()

    # ── Public API ────────────────────────────────────────────────────

    def stop(self):
        self._stop_flag.set()
        self._set_status("stop requested")

    def is_running(self) -> bool:
        return not self._stop_flag.is_set()

    def get_last_sample(self) -> str:
        with self._lock:
            return self.last_sample

    def get_status(self) -> str:
        with self._lock:
            return self.status

    # ── Entry point ───────────────────────────────────────────────────

    def run(self):
        """Call this in a daemon thread."""
        self._set_status("starting")
        logger.info("ContinuousTrainer starting.")
        try:
            self._loop()
        except Exception as e:
            logger.exception("ContinuousTrainer crashed")
            self._set_status(f"error: {e}")
        finally:
            logger.info("ContinuousTrainer finished.")
            self._set_status("stopped")

    # ── Main loop ─────────────────────────────────────────────────────

    def _loop(self):
        from nano_llm.data.crawler import MultiCrawler, CrawlerConfig
        from nano_llm.data.cleaner import clean_text, split_into_chunks
        from nano_llm.data.dataset import build_dataset, build_dataloader
        from nano_llm.training.trainer import Trainer, TrainerConfig
        from nano_llm.training.callbacks import (
            LossLoggerCallback, SampleGenerationCallback,
        )
        from nano_llm.memory.auto_arch import auto_configure

        # ── Start all crawlers ────────────────────────────────────────
        crawler_cfg = CrawlerConfig(
            sources=self.config.sources,
            rate_limits=self.config.rate_limits,
            buffer_size=self.config.crawler_buffer,
            rss_feeds=self.config.rss_feeds,
        )
        multi = MultiCrawler(crawler_cfg)
        multi.start()
        self._set_status("crawlers running, waiting for initial buffer fill")

        logger.info("Waiting for first crawler response...")
        self._wait_for_first_documents(multi)

        # ── Bootstrap: tokenizer + model if not provided ──────────────
        if self.tokenizer is None:
            self._bootstrap_tokenizer(multi)

        if self.model is None:
            self._bootstrap_model()
            if self.on_model_ready:
                self.on_model_ready(self)

        # Auto-configure batch size / grad accum once
        _, batch_size, grad_accum = auto_configure(
            seq_len=self.config.seq_len,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            vocab_size=self.tokenizer.vocab_size(),
            max_tier="small",
        )

        best_loss        = float("inf")
        no_improve_count = 0
        round_num        = 0

        while not self._stop_flag.is_set():
            round_num += 1
            self.current_round = round_num

            if self.config.max_rounds > 0 and round_num > self.config.max_rounds:
                logger.info(f"Reached max_rounds={self.config.max_rounds}. Stopping.")
                self._set_status("stopped (max rounds)")
                break

            if self.on_round_start:
                self.on_round_start(round_num)

            logger.info(
                f"\n{'='*60}\n"
                f"Round {round_num}  |  buffer={multi.buffer_size()}  |  "
                f"stats={multi.stats()}\n"
                f"{'='*60}"
            )

            # ── 1. Drain buffer ───────────────────────────────────────
            self._set_status(f"round {round_num}: draining crawler buffer")
            raw_docs = multi.drain(self.config.docs_per_round, timeout=60)

            if not raw_docs:
                logger.warning("No documents available. Waiting 15s and retrying.")
                self._set_status(f"round {round_num}: waiting for data")
                time.sleep(15)
                continue

            logger.info(f"  Got {len(raw_docs)} docs from buffer")

            # ── 2. Clean + chunk ──────────────────────────────────────
            new_chunks = []
            for title, text in raw_docs:
                cleaned = clean_text(text)
                chunks  = split_into_chunks(cleaned, chunk_size=1024, overlap=64)
                new_chunks.extend(chunks)
                self._total_chars += len(cleaned)

            self._window_store.append_round(round_num, new_chunks)
            window_chunks = self._window_store.load_window_chunks()

            logger.info(
                f"  New chunks: {len(new_chunks):,}  |  "
                f"Window total: {len(window_chunks):,}  |  "
                f"Total chars seen: {self._total_chars:,}"
            )
            logger.info(f"  {self._window_store.summary()}")

            # ── 3. Tokenizer retrain check ────────────────────────────
            if self._should_retrain_tokenizer(round_num):
                self._retrain_tokenizer_and_resize(round_num)
                # Recompute batch size after vocab resize (embedding is larger)
                _, batch_size, grad_accum = auto_configure(
                    seq_len=self.config.seq_len,
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                    vocab_size=self.tokenizer.vocab_size(),
                    max_tier="small",
                )

            # ── 4. Build dataset ──────────────────────────────────────
            self._set_status(f"round {round_num}: building dataset")
            bin_path = self.config.data_cache_dir / f"round_{round_num:04d}.bin"
            try:
                dataset = build_dataset(
                    window_chunks, self.tokenizer,
                    self.config.seq_len, bin_path,
                )
            except Exception as e:
                logger.error(f"Dataset build failed: {e}")
                continue

            if len(dataset) == 0:
                logger.warning("Dataset empty. Skipping round.")
                continue

            dataloader = build_dataloader(dataset, batch_size, num_workers=0)

            # ── 5. Train N epochs ─────────────────────────────────────
            self._set_status(
                f"round {round_num}: training {self.config.epochs_per_round} epochs "
                f"(vocab={self.tokenizer.vocab_size():,})"
            )

            trainer_cfg = TrainerConfig(
                epochs=self.config.epochs_per_round,
                steps_per_epoch=self.config.steps_per_epoch,
                lr=self.config.lr,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                grad_accum_steps=grad_accum,
                sample_prompt=self.config.sample_prompt,
                sample_every_n_epochs=self.config.epochs_per_round,
                checkpoint_dir=self.config.checkpoint_dir,
                save_every_n_epochs=self.config.epochs_per_round,
            )

            step_cb   = _StepCallback(self.on_step)
            sample_cb = SampleGenerationCallback(
                every_k_epochs=self.config.epochs_per_round,
                max_new_tokens=200,
            )
            log_cb = LossLoggerCallback()

            trainer = Trainer(
                model=self.model,
                dataloader=dataloader,
                config=trainer_cfg,
                tokenizer=self.tokenizer,
                callbacks=[step_cb, sample_cb, log_cb],
                device=self.device,
            )
            trainer.global_step = self.global_step

            bridge = _StopBridge(self._stop_flag, trainer)
            bridge.start()

            trainer.train()

            self.global_step   = trainer.global_step
            self.global_epoch += self.config.epochs_per_round

            # ── 6. Evaluate ───────────────────────────────────────────
            mean_loss   = _last_epoch_loss(log_cb)
            sample_text = sample_cb.get_sample()

            with self._lock:
                self.last_sample = sample_text
            self.loss_history.append(mean_loss)

            if self.on_sample:
                self.on_sample(round_num, self.global_epoch, mean_loss, sample_text)

            logger.info(
                f"Round {round_num} complete | "
                f"global_epoch={self.global_epoch} | loss={mean_loss:.4f} | "
                f"vocab={self.tokenizer.vocab_size():,} | "
                f"buffer={multi.buffer_size()} (crawlers still running)"
            )

            # ── 7. Auto-stop heuristic ────────────────────────────────
            if mean_loss < best_loss - self.config.min_loss_delta:
                best_loss        = mean_loss
                no_improve_count = 0
                logger.info(f"  ✓ Loss improved → best={best_loss:.4f}")
            else:
                no_improve_count += 1
                logger.info(
                    f"  ✗ No improvement "
                    f"({no_improve_count}/{self.config.patience})"
                )

            if no_improve_count >= self.config.patience:
                logger.info(
                    f"Auto-stopping: no improvement for "
                    f"{self.config.patience} consecutive rounds."
                )
                self._set_status("auto-stopped (loss plateau)")
                break

        multi.stop()
        self._set_status("stopped")

    # ── Helpers ───────────────────────────────────────────────────────

    def _set_status(self, s: str):
        with self._lock:
            self.status = s

    def _wait_for_first_documents(self, multi) -> None:
        """
        Wait until at least one crawler has produced at least one document.
        This prevents live training from crashing just because the first poll
        was empty or some sources are temporarily unavailable.
        """
        while not self._stop_flag.is_set():
            if multi.buffer_size() > 0:
                logger.info(f"Initial crawler response received. Buffer={multi.buffer_size()}")
                return

            stats = multi.stats()
            if any(count > 0 for count in stats.values()):
                logger.info(f"First crawler documents observed via stats={stats}")
                return

            self._set_status("waiting for first crawler response")
            time.sleep(2.0)

    def _bootstrap_tokenizer(self, multi) -> None:
        """
        Crawl a seed batch and train the initial BPE tokenizer from scratch.
        Called only when no tokenizer was provided at construction time.
        """
        from nano_llm.data.cleaner import clean_text, split_into_chunks
        from nano_llm.tokenizer.bpe_tokenizer import ByteLevelBPETokenizer

        self._set_status("bootstrap: crawling seed data for tokenizer training")
        logger.info(
            f"\n{'─'*60}\n"
            f"Bootstrap: crawling seed docs for initial tokenizer\n"
            f"  Target docs : {self.config.docs_per_round}\n"
            f"{'─'*60}"
        )

        # Drain enough docs for a reasonable tokenizer corpus.
        # Do not crash just because the first fetch window is empty.
        seed_docs = []
        while not seed_docs and not self._stop_flag.is_set():
            self._set_status("bootstrap: waiting for first crawler documents")
            seed_docs = multi.drain(self.config.docs_per_round, timeout=30)

        if self._stop_flag.is_set():
            raise RuntimeError("Bootstrap stopped before any crawler returned data.")

        seed_chunks = []
        for title, text in seed_docs:
            cleaned = clean_text(text)
            chunks  = split_into_chunks(cleaned, chunk_size=1024, overlap=64)
            seed_chunks.extend(chunks)
            self._total_chars += len(cleaned)

        logger.info(f"  Seed corpus: {len(seed_chunks):,} chunks, {self._total_chars:,} chars")
        self._window_store.append_round(0, seed_chunks)
        logger.info(f"  {self._window_store.summary()}")

        self._set_status(
            f"bootstrap: training tokenizer "
            f"(vocab={self.config.tokenizer_vocab_size:,})"
        )
        tok = ByteLevelBPETokenizer()
        tok.train(texts=seed_chunks, vocab_size=self.config.tokenizer_vocab_size)
        tok.save_pretrained(self.config.tokenizer_dir)
        self.tokenizer = tok
        logger.info(f"  Bootstrap tokenizer ready: vocab={tok.vocab_size():,}")

    def _bootstrap_model(self) -> None:
        """
        Auto-configure and build the model from scratch.
        Called only when no model was provided at construction time.
        """
        import torch
        from nano_llm.memory.auto_arch import auto_configure
        from nano_llm.model.transformer import NanoLLM

        self._set_status("bootstrap: auto-configuring model architecture")
        logger.info("Bootstrap: auto-configuring model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        config, _, _ = auto_configure(
            seq_len=self.config.seq_len,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            vocab_size=self.tokenizer.vocab_size(),
            max_tier="small",
        )
        config.vocab_size = self.tokenizer.vocab_size()
        self.model = NanoLLM.from_config(config)

        logger.info(
            f"  Bootstrap model ready: tier={config.tier_name}, "
            f"params={self.model.param_count():,}, device={self.device}"
        )

        # Notify UI via on_tokenizer_retrain callback reused as a model-ready signal
        # (the UI will pick up model/tokenizer from the trainer reference)

    def _should_retrain_tokenizer(self, round_num: int) -> bool:
        """Return True if conditions are met to retrain the BPE tokenizer."""
        cfg = self.config
        if cfg.retrain_tokenizer_every_n_rounds <= 0:
            return False
        # Must have accumulated enough characters
        if self._total_chars < cfg.min_chars_for_retrain:
            logger.info(
                f"  Tokenizer retrain: {self._total_chars:,} / "
                f"{cfg.min_chars_for_retrain:,} chars accumulated"
            )
            return False
        # Must be on a retrain interval
        rounds_since = round_num - self._last_tok_retrain_round
        if rounds_since < cfg.retrain_tokenizer_every_n_rounds:
            return False
        return True

    def _retrain_tokenizer_and_resize(self, round_num: int) -> None:
        """
        Retrain BPE on the current corpus, then resize the model's
        embedding table and output head to match the new vocab.

        Steps:
          1. Train a fresh ByteLevelBPETokenizer on the current live window
          2. Save it to config.tokenizer_dir
          3. Call model.resize_vocab(new_vocab_size) — preserves existing weights
          4. Rebuild the AdamW optimizer (state tensors are now the wrong shape)
          5. Update self.tokenizer reference
        """
        from nano_llm.tokenizer.bpe_tokenizer import ByteLevelBPETokenizer

        old_vocab = self.tokenizer.vocab_size()
        target_vocab = self.config.tokenizer_vocab_size

        self._set_status(
            f"round {round_num}: retraining tokenizer "
            f"(window={len(self._window_store.load_window_chunks()):,} chunks, "
            f"target_vocab={target_vocab:,})"
        )
        logger.info(
            f"\n{'─'*60}\n"
            f"Tokenizer retrain @ round {round_num}\n"
            f"  Window shards : {self._window_store.shard_count:,}\n"
            f"  Total chars   : {self._total_chars:,}\n"
            f"  Old vocab     : {old_vocab:,}\n"
            f"  Target vocab  : {target_vocab:,}\n"
            f"{'─'*60}"
        )

        try:
            window_chunks = self._window_store.load_window_chunks()
            if not window_chunks:
                logger.warning("Tokenizer retrain skipped because the live window is empty.")
                return
            new_tok = ByteLevelBPETokenizer()
            new_tok.train(
                texts=window_chunks,
                vocab_size=target_vocab,
            )
            new_tok.save_pretrained(self.config.tokenizer_dir)
            new_vocab = new_tok.vocab_size()

            # Resize model if vocab changed
            if new_vocab != old_vocab:
                # Move model to CPU for the resize (avoids fragmentation)
                self.model.cpu()
                self.model.resize_vocab(new_vocab)
                self.model.to(self.device)
                logger.info(
                    f"  Model embedding resized: {old_vocab:,} → {new_vocab:,}"
                )
            else:
                logger.info(
                    f"  Vocab unchanged at {new_vocab:,} — no resize needed"
                )

            self.tokenizer = new_tok
            self._last_tok_retrain_round = round_num

            if self.on_tokenizer_retrain:
                self.on_tokenizer_retrain(round_num, old_vocab, new_vocab)

            logger.info(f"  Tokenizer retrain complete. New vocab: {new_vocab:,}")

        except Exception as e:
            logger.error(f"Tokenizer retrain failed: {e}", exc_info=True)
            logger.warning("Continuing with existing tokenizer.")


# ── Internal helpers ──────────────────────────────────────────────────

class _StepCallback:
    """Forwards step events to an optional callable."""

    def __init__(self, fn):
        self._fn = fn

    def on_step_end(self, step: int, loss: float, trainer) -> None:
        if self._fn:
            self._fn(step, loss)

    def on_epoch_end(self, *a): pass
    def on_train_start(self, *a): pass
    def on_train_end(self, *a): pass


class _StopBridge(threading.Thread):
    """Watches the outer stop flag and forwards it to the inner Trainer."""

    def __init__(self, flag: threading.Event, trainer):
        super().__init__(daemon=True)
        self._flag    = flag
        self._trainer = trainer

    def run(self):
        while not self._flag.is_set():
            time.sleep(0.5)
        self._trainer.stop()


def _wait_for_buffer(multi, target: int, timeout: float):
    """Block until the crawler buffer has at least `target` items."""
    deadline = time.time() + timeout
    while multi.buffer_size() < target and time.time() < deadline:
        time.sleep(1.0)
    logger.info(f"Buffer ready: {multi.buffer_size()} docs")


def _last_epoch_loss(log_cb) -> float:
    """Read the last epoch-end mean loss from the LossLoggerCallback file."""
    try:
        import json
        path = log_cb.log_path
        if not path.exists():
            return float("inf")
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                rec = json.loads(line)
                if rec.get("step") == "epoch_end":
                    return rec["mean_loss"]
            except Exception:
                continue
    except Exception:
        pass
    return float("inf")
