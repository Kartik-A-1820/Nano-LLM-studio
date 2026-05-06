"""
Headless CLI training — no UI required.

Usage:
    python scripts/train_cli.py --help
    python scripts/train_cli.py --epochs 20 --seq_len 512 --vocab_size 8000
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_cli")


def parse_args():
    p = argparse.ArgumentParser(description="Nano-LLM CLI trainer")
    p.add_argument("--train_dir", default="train", help="Directory with training files")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq_len", type=int, default=2048,
                   help="Context length (default 2048, matches GPT-2)")
    p.add_argument("--vocab_size", type=int, default=50_257,
                   help="BPE vocabulary size (default 50257, matches GPT-2)")
    p.add_argument("--tier", default=None, help="Force architecture tier (micro/nano/small/medium/large)")
    p.add_argument("--no_gc", action="store_true", help="Disable gradient checkpointing")
    p.add_argument("--sample_prompt", default="Once upon a time", help="Sample prompt")
    p.add_argument("--sample_k", type=int, default=5, help="Generate sample every K epochs")
    p.add_argument("--checkpoint_dir", default="outputs/checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Ingest data
    from nano_llm.data.extractor import ingest_folder
    from nano_llm.data.cleaner import clean_text, split_into_chunks

    train_dir = Path(args.train_dir)
    if not train_dir.exists():
        logger.error(f"train/ directory not found: {train_dir.absolute()}")
        sys.exit(1)

    logger.info(f"Ingesting files from {train_dir.absolute()} ...")
    pairs = ingest_folder(train_dir)
    if not pairs:
        logger.error("No supported files found in train/. Add .txt/.pdf/.docx files.")
        sys.exit(1)

    all_chunks = []
    for path, raw in pairs:
        cleaned = clean_text(raw)
        chunks = split_into_chunks(cleaned, chunk_size=1024, overlap=64)
        all_chunks.extend(chunks)
    logger.info(f"Total chunks: {len(all_chunks):,}")

    # 2. Train tokenizer
    from nano_llm.tokenizer.bpe_tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train(all_chunks, vocab_size=args.vocab_size, model_prefix="outputs/tokenizer/bpe")
    tok.save_pretrained(Path("outputs/tokenizer"))
    logger.info(f"Tokenizer ready: vocab_size={tok.vocab_size():,}")

    # 3. Auto-configure architecture
    use_gc = not args.no_gc
    from nano_llm.memory.auto_arch import auto_configure
    config, batch_size, grad_accum = auto_configure(
        seq_len=args.seq_len,
        use_gradient_checkpointing=use_gc,
        vocab_size=tok.vocab_size(),
        force_tier=args.tier,
    )

    # 4. Build dataset
    from nano_llm.data.dataset import build_dataset, build_dataloader
    dataset = build_dataset(all_chunks, tok, args.seq_len, Path("outputs/train_data.bin"))
    if len(dataset) == 0:
        logger.error("Dataset is empty — not enough text for even one sample.")
        sys.exit(1)
    dataloader = build_dataloader(dataset, batch_size, num_workers=0)

    # 5. Build model
    from nano_llm.model.transformer import NanoLLM
    config.vocab_size = tok.vocab_size()
    model = NanoLLM.from_config(config)
    logger.info(f"Model parameters: {model.param_count():,}")

    # 6. Train
    from nano_llm.training.trainer import Trainer, TrainerConfig
    from nano_llm.training.callbacks import LossLoggerCallback, SampleGenerationCallback

    trainer_cfg = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        use_gradient_checkpointing=use_gc,
        grad_accum_steps=grad_accum,
        sample_prompt=args.sample_prompt,
        sample_every_n_epochs=args.sample_k,
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    sample_cb = SampleGenerationCallback(every_k_epochs=args.sample_k)
    log_cb = LossLoggerCallback()

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        config=trainer_cfg,
        tokenizer=tok,
        callbacks=[sample_cb, log_cb],
    )
    trainer.train()

    logger.info("Training complete. Checkpoints saved to outputs/checkpoints/")
    last_sample = sample_cb.get_sample()
    if last_sample:
        logger.info(f"Last sample:\n{last_sample}")


if __name__ == "__main__":
    main()
