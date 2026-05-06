"""Training loop with mixed precision, gradient checkpointing, and callbacks."""
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
    ):
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

        # Cosine LR scheduler with warmup
        steps_per_epoch = config.steps_per_epoch or max(1, len(dataloader) // config.grad_accum_steps)
        total_steps = max(1, steps_per_epoch * config.epochs)
        warmup_steps = min(100, total_steps // 10)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=warmup_steps / max(total_steps, 1),
            anneal_strategy="cos",
        )

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

        for epoch in range(1, self.config.epochs + 1):
            if self._stop_flag.is_set():
                logger.info("Training stopped by user.")
                break
            self.current_epoch = epoch
            mean_loss = self._train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{self.config.epochs}  loss={mean_loss:.4f}")

            if mean_loss < self.best_mean_loss - self.config.min_loss_delta:
                self.best_mean_loss = mean_loss
                self.no_improve_count = 0
                self.save_checkpoint(
                    epoch,
                    filename=self.config.best_checkpoint_name,
                    extra_state={"best_mean_loss": self.best_mean_loss},
                )
                logger.info(
                    f"New best checkpoint saved: {self.config.best_checkpoint_name} "
                    f"(loss={mean_loss:.4f})"
                )
            else:
                self.no_improve_count += 1

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, mean_loss, self)

            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, filename=self.config.last_checkpoint_name)

            if (
                self.config.early_stopping_patience is not None
                and self.no_improve_count >= self.config.early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered after %s epochs without sufficient improvement.",
                    self.no_improve_count,
                )
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer_steps = 0
        self.optimizer.zero_grad()

        if self.config.steps_per_epoch is None:
            batch_iter = iter(self.dataloader)
            target_optimizer_steps = None
        else:
            batch_iter = iter(self.dataloader)
            target_optimizer_steps = int(self.config.steps_per_epoch)

        step_in_epoch = 0
        while True:
            if self._stop_flag.is_set():
                break
            if target_optimizer_steps is not None and optimizer_steps >= target_optimizer_steps:
                break
            try:
                x, y = next(batch_iter)
            except StopIteration:
                if target_optimizer_steps is None:
                    break
                batch_iter = iter(self.dataloader)
                x, y = next(batch_iter)

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            use_gc = self.config.use_gradient_checkpointing
            amp_enabled = self.config.use_fp16 and self.device == "cuda"

            with torch.amp.autocast("cuda" if self.device == "cuda" else "cpu",
                                    dtype=torch.float16, enabled=amp_enabled):
                _, loss = self.model(x, targets=y, use_checkpoint=use_gc)
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.config.grad_accum_steps
            n_batches += 1

            # Gradient accumulation step
            if (step_in_epoch + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
                optimizer_steps += 1

                step_loss = total_loss / n_batches
                if self.global_step % self.config.log_every_n_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"  Epoch {epoch} | step {self.global_step} | "
                        f"loss={step_loss:.4f} | lr={lr:.2e}"
                    )

                for cb in self.callbacks:
                    cb.on_step_end(self.global_step, step_loss, self)

            step_in_epoch += 1

        return total_loss / max(n_batches, 1)

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
            "config": self.model.config,
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
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = ckpt.get("global_step", 0)
        epoch = ckpt["epoch"]
        logger.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch
