"""Training callbacks for logging, sample generation, and Gradio UI updates."""
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer

logger = logging.getLogger(__name__)


class BaseCallback:
    """Base class for training callbacks."""
    def on_step_end(self, step: int, loss: float, trainer: "Trainer") -> None:
        pass

    def on_epoch_end(self, epoch: int, mean_loss: float, trainer: "Trainer") -> None:
        pass

    def on_train_start(self, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class LossLoggerCallback(BaseCallback):
    """Appends training metrics to a JSONL file."""

    def __init__(self, log_path: Path = Path("outputs/logs/loss.jsonl")):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, step: int, loss: float, trainer: "Trainer") -> None:
        record = {
            "epoch": trainer.current_epoch,
            "step": step,
            "loss": round(loss, 6),
            "timestamp": time.time(),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def on_epoch_end(self, epoch: int, mean_loss: float, trainer: "Trainer") -> None:
        record = {
            "epoch": epoch,
            "step": "epoch_end",
            "mean_loss": round(mean_loss, 6),
            "timestamp": time.time(),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class SampleGenerationCallback(BaseCallback):
    """
    Generates a text sample every K epochs using the model's current weights.
    Stores the result in self.last_sample for the UI to read.
    """

    def __init__(self, every_k_epochs: int = 1, max_new_tokens: int = 150):
        self.every_k = every_k_epochs
        self.max_new_tokens = max_new_tokens
        self.last_sample: str = ""
        self._lock = threading.Lock()

    def on_epoch_end(self, epoch: int, mean_loss: float, trainer: "Trainer") -> None:
        if epoch % self.every_k != 0:
            return

        prompt = trainer.config.sample_prompt
        if not prompt:
            return

        try:
            from nano_llm.inference.generator import Generator
            gen = Generator(trainer.model, trainer.tokenizer, trainer.device)
            sample = gen.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.8,
                top_k=50,
            )
            result = f"[Epoch {epoch}] {prompt}{sample}"
            with self._lock:
                self.last_sample = result
            logger.info(f"\n{'─'*60}\nSample @ epoch {epoch} (loss={mean_loss:.4f}):\n{result}\n{'─'*60}")
        except Exception as e:
            with self._lock:
                self.last_sample = f"[Epoch {epoch}] Generation failed: {e}"
            logger.warning(f"Sample generation failed at epoch {epoch}: {e}")

    def get_sample(self) -> str:
        with self._lock:
            return self.last_sample


class GradioUpdateCallback(BaseCallback):
    """
    Pushes (global_step, loss) tuples into a thread-safe queue.
    The Gradio UI polls this queue to update its live loss chart.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self.global_step = 0
        self._stop_event = threading.Event()

    def on_step_end(self, step: int, loss: float, trainer: "Trainer") -> None:
        self.global_step += 1
        self._queue.put_nowait((self.global_step, loss))

    def get_updates(self) -> list[tuple[int, float]]:
        """Drain the queue and return all pending (step, loss) pairs."""
        updates = []
        while not self._queue.empty():
            try:
                updates.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return updates

    def signal_stop(self):
        self._stop_event.set()
