"""PyTorch Dataset backed by a memory-mapped numpy binary file."""
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TokenizedDataset(Dataset):
    """
    Reads a flat uint16 numpy binary file via mmap.
    Each sample: (input_ids[i:i+seq_len], target_ids[i+1:i+seq_len+1]).
    """

    def __init__(self, bin_path: Path, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        n_tokens = len(self.data)
        self.n_samples = max(0, (n_tokens - 1) // seq_len)
        logger.info(
            f"Dataset: {n_tokens:,} tokens, {self.n_samples:,} samples "
            f"(seq_len={seq_len})"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def build_dataset(
    texts: list[str],
    tokenizer,
    seq_len: int,
    output_bin: Path,
) -> TokenizedDataset:
    """
    Tokenizes all texts, concatenates with BOS/EOS separators, writes a flat
    uint16 .bin file, returns a TokenizedDataset ready for training.
    """
    output_bin = Path(output_bin)
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    all_ids: list[int] = []
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        all_ids.extend(ids)
        if (i + 1) % 100 == 0:
            logger.info(f"Tokenized {i+1}/{len(texts)} chunks")

    logger.info(f"Total tokens: {len(all_ids):,}")
    arr = np.array(all_ids, dtype=np.uint16)
    with open(output_bin, "wb") as f:
        arr.tofile(f)
    logger.info(f"Written to {output_bin}")

    return TokenizedDataset(output_bin, seq_len)


def build_dataset_from_store(
    store,
    tokenizer,
    seq_len: int,
    output_bin: Path,
    force_rebuild: bool = False,
) -> TokenizedDataset:
    """
    Build a TokenizedDataset from a CorpusStore.

    Fast path: if the store already has a pre-built corpus.bin and
    force_rebuild is False, reuses it directly — no re-tokenization.
    Otherwise tokenizes all JSONL records and writes a new .bin.

    Args:
        store:         CorpusStore instance
        tokenizer:     trained tokenizer
        seq_len:       sequence length for training
        output_bin:    path for the output .bin (ignored if store.bin_built)
        force_rebuild: always re-tokenize even if .bin exists
    """
    if store.bin_built and not force_rebuild:
        logger.info(f"Reusing pre-built corpus.bin: {store.bin_path}")
        return TokenizedDataset(store.bin_path, seq_len)

    logger.info(f"Building dataset from CorpusStore ({store.total_docs:,} docs)...")
    texts = store.load_texts()
    return build_dataset(texts, tokenizer, seq_len, output_bin)


def build_dataloader(
    dataset: TokenizedDataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
