"""Training loop with mixed precision, gradient checkpointing, and callbacks."""
import math
from itertools import islice
from dataclasses import asdict, is_dataclass
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .callbacks import BaseCallback
from .optimizer import build_optimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    use_fp16: bool = True
    use_gradient_checkpointing: bool = True
    grad_accum_steps: int = 4
    steps_per_epoch: int | None = None
    early_stopping_patience: int | None = None
    min_loss_delta: float = 0.0
    save_every_n_epochs: int = 1
    sample_every_n_epochs: int = 1
    sample_prompt: str = "Once upon a time"
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    use_8bit_adam: bool = False
    log_every_n_steps: int = 10


class Trainer:
    def __init__(
        self,
        model,
        dataloader: DataLoader,
        config: TrainerConfig,
        tokenizer,
        callbacks: list[BaseCallback] | None = None,
        device: str | None = None,
        validation_dataloader: DataLoader | None = None,
    ):
        if config.grad_accum_steps < 1 or config.epochs < 1 or len(dataloader) == 0:
            raise ValueError("Training requires positive epochs/accumulation and a nonempty loader")
        if config.steps_per_epoch is not None and config.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        if config.save_every_n_epochs < 1 or config.log_every_n_steps < 1:
            raise ValueError("Save/log intervals must be positive")
        if validation_dataloader is not None and len(validation_dataloader) == 0:
            raise ValueError("Validation loader is empty")
        self.validation_dataloader = validation_dataloader
        self.validation_loss = None
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.tokenizer = tokenizer
        self.callbacks = callbacks or []
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.optimizer = build_optimizer(
            model, lr=config.lr,
            weight_decay=config.weight_decay,
            use_8bit=config.use_8bit_adam,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_fp16 and self.device == "cuda")

        # One schedule entry per optimizer update, including a short final group.
        updates_per_epoch = config.steps_per_epoch or math.ceil(len(dataloader) / config.grad_accum_steps)
        self.total_steps = updates_per_epoch * config.epochs
        warmup = min(100, self.total_steps // 10)
        def lr_factor(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, max(0.0, (step - warmup) / max(1, self.total_steps - warmup)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)

        self.schedule_signature = {
            "epochs": config.epochs, "grad_accum_steps": config.grad_accum_steps,
            "steps_per_epoch": config.steps_per_epoch, "loader_batches": len(dataloader),
            "batch_size": dataloader.batch_size, "validation": validation_dataloader is not None,
        }
        self.current_epoch = 0
        self._stop_flag = threading.Event()
        self.global_step = 0
        self.best_mean_loss = float("inf")
        self.no_improve_count = 0

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def stop(self):
        """Signal training to stop after the current step."""
        self._stop_flag.set()

    def train(self) -> None:
        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(self.current_epoch + 1, self.config.epochs + 1):
            if self._stop_flag.is_set():
                logger.info("Training stopped by user.")
                break
            self.current_epoch = epoch
            mean_loss = self._train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{self.config.epochs}  loss={mean_loss:.4f}")

            if self._stop_flag.is_set():
                # Resume is at an epoch boundary; repeat this partial epoch.
                self.current_epoch = epoch - 1
                self.save_checkpoint(epoch - 1, filename=self.config.last_checkpoint_name)
                break
            self.validation_loss = self.evaluate() if self.validation_dataloader is not None else None
            if self.validation_loss is not None:
                if self.validation_loss < self.best_mean_loss - self.config.min_loss_delta:
                    self.best_mean_loss = self.validation_loss
                    self.no_improve_count = 0
                    self.save_checkpoint(epoch, filename=self.config.best_checkpoint_name)
                else:
                    self.no_improve_count += 1

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, mean_loss, self)

            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, filename=self.config.last_checkpoint_name)

            if (
                self.validation_dataloader is not None
                and self.config.early_stopping_patience is not None
                and self.no_improve_count >= self.config.early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered after %s epochs without sufficient improvement.",
                    self.no_improve_count,
                )
                break

        if self.current_epoch > 0:
            self.save_checkpoint(self.current_epoch, filename=self.config.last_checkpoint_name)
        for cb in self.callbacks:
            cb.on_train_end(self)

    def _batch_stream(self):
        if self.config.steps_per_epoch is None:
            yield from self.dataloader
            return
        iterator = iter(self.dataloader)
        for _ in range(self.config.steps_per_epoch * self.config.grad_accum_steps):
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                yield next(iterator)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, total_tokens = 0.0, 0
        stream = iter(self._batch_stream())
        self.optimizer.zero_grad(set_to_none=True)
        while not self._stop_flag.is_set():
            group = list(islice(stream, self.config.grad_accum_steps))
            if not group:
                break
            # Mean cross-entropy is weighted by supervised tokens, not batches.
            counts = [int((y != -100).sum()) for _, y in group]
            count = sum(counts)
            if count == 0:
                raise ValueError("Accumulation group has no supervised tokens")
            for (x, y), n_tokens in zip(group, counts):
                if n_tokens == 0:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                with torch.amp.autocast("cuda" if self.device == "cuda" else "cpu",
                        dtype=torch.float16, enabled=self.config.use_fp16 and self.device == "cuda"):
                    _, loss = self.model(x, targets=y, use_checkpoint=self.config.use_gradient_checkpointing)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                self.scaler.scale(loss * (n_tokens / count)).backward()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            # A decreasing scaler means the optimizer update was skipped.
            if self.scaler.get_scale() >= previous_scale:
                self.scheduler.step()
                self.global_step += 1
                for cb in self.callbacks:
                    cb.on_step_end(self.global_step, total_loss / total_tokens, self)
        return total_loss / max(total_tokens, 1)

    @torch.inference_mode()
    def evaluate(self) -> float:
        was_training = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        try:
            for x, y in self.validation_dataloader:
                n_tokens = int((y != -100).sum())
                if n_tokens == 0:
                    continue
                _, loss = self.model(x.to(self.device), targets=y.to(self.device), use_checkpoint=False)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite validation loss")
                total += float(loss) * n_tokens
                count += n_tokens
        finally:
            self.model.train(was_training)
        if not count:
            raise ValueError("Validation has no supervised tokens")
        return total / count

    def save_checkpoint(
        self,
        epoch: int,
        filename: str | None = None,
        extra_state: dict | None = None,
    ) -> Path:
        path = self.config.checkpoint_dir / (filename or f"epoch_{epoch:04d}.pt")
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": asdict(self.model.config) if is_dataclass(self.model.config) else vars(self.model.config),
            "scheduler_state": self.scheduler.state_dict(),
            "total_steps": self.total_steps,
            "schedule_signature": self.schedule_signature,
            "best_mean_loss": self.best_mean_loss,
            "no_improve_count": self.no_improve_count,
            "validation_loss": self.validation_loss,
            "selection_metric": "validation_loss" if self.validation_dataloader is not None else None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "global_step": self.global_step,
        }
        if extra_state:
            payload.update(extra_state)
        torch.save(
            payload,
            str(path),
        )
        self._prune_checkpoints()
        logger.info(f"Checkpoint saved: {path}")
        return path

    def _prune_checkpoints(self) -> None:
        keep = {
            self.config.best_checkpoint_name,
            self.config.last_checkpoint_name,
        }
        for path in self.config.checkpoint_dir.glob("*.pt"):
            if path.name not in keep:
                path.unlink(missing_ok=True)

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
        if ckpt.get("total_steps") != self.total_steps or ckpt.get("schedule_signature") != self.schedule_signature:
            raise ValueError("Resume requires the same planned epochs and loader/accumulation schedule")
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.best_mean_loss = ckpt.get("best_mean_loss", float("inf"))
        self.no_improve_count = ckpt.get("no_improve_count", 0)
        self.validation_loss = ckpt.get("validation_loss")
        torch.set_rng_state(ckpt["rng_state"].cpu())
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng_state"]])
        self.global_step = ckpt.get("global_step", 0)
        epoch = ckpt["epoch"]
        self.current_epoch = epoch
        logger.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch
