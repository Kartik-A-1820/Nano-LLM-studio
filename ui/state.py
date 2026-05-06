"""Shared mutable application state across the 3-tab UI."""
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppState:
    # Data pipeline
    extracted_texts: list[str] = field(default_factory=list)
    data_preview_text: str = ""
    data_status_text: str = "No data loaded yet."
    data_revision: int = 0
    active_data_source: str = "none"
    corpus_path: Optional[Path] = None
    corpus_store: object = None

    # Crawl collection
    is_crawl_collecting: bool = False
    crawl_stop_event: Optional[threading.Event] = None
    crawl_thread: Optional[threading.Thread] = None

    # Tokenizer / model readiness
    tokenizer: object = None
    tokenizer_path: Optional[Path] = None
    tokenizer_revision: int = -1
    model: object = None
    model_config: object = None
    generator: object = None
    latest_checkpoint_path: Optional[Path] = None
    device: str = "cpu"

    # Unified training state
    training_mode: str = "idle"
    training_status_text: str = "Idle."
    is_training: bool = False
    training_thread: Optional[threading.Thread] = None
    trainer: object = None
    continuous_trainer: object = None
    continuous_thread: Optional[threading.Thread] = None
    gradio_callback: object = None
    sample_callback: object = None
    loss_history: list[tuple[int, float]] = field(default_factory=list)
    continuous_loss_history: list[tuple[int, float]] = field(default_factory=list)
    continuous_samples: list[dict] = field(default_factory=list)
    continuous_current_round: int = 0
    continuous_tok_retrains: list[dict] = field(default_factory=list)
    auto_config_summary: str = ""
    batch_size: int = 1
    grad_accum_steps: int = 4

    # Training stage tracking (for progress UI)
    training_stage: str = "idle"  # idle | tokenizer | config | training | done
    training_current_epoch: int = 0
    training_total_epochs: int = 0
    training_total_steps: int = 0
    training_abort: bool = False

    lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_data_updated(self, texts: list[str], source: str, status: str) -> None:
        self.extracted_texts = texts
        self.data_preview_text = "\n\n".join(texts[:3])[:2000] if texts else ""
        self.data_status_text = status
        self.active_data_source = source
        self.data_revision += 1
        self.generator = None

    def reset_manual_training(self) -> None:
        self.training_thread = None
        self.trainer = None
        self.gradio_callback = None
        self.sample_callback = None
        self.loss_history.clear()
        self.training_stage = "idle"
        self.training_current_epoch = 0
        self.training_abort = False

    def reset_live_training(self) -> None:
        self.continuous_trainer = None
        self.continuous_thread = None
        self.continuous_loss_history.clear()
        self.continuous_samples.clear()
        self.continuous_current_round = 0
        self.continuous_tok_retrains.clear()


STATE = AppState()
