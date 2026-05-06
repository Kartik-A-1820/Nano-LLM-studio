"""
CorpusStore — efficient storage for crawled training data.

Storage formats
───────────────
  train/corpus.jsonl   — raw cleaned text, one JSON object per line:
                          {"source": "wikipedia", "title": "...", "text": "...", "chars": N}
                         Appendable, seekable, human-readable, single file.
                         Much more efficient than thousands of individual .txt files.

  train/corpus.bin     — pre-tokenized uint16 token IDs (memory-mapped).
                         Built on demand once a char threshold is crossed.
                         The dataloader reads this directly with zero parsing overhead.

Why JSONL over individual .txt files
──────────────────────────────────────
  • One file descriptor instead of thousands
  • Metadata (source, title, char count) stored alongside text
  • Appendable without reading the whole file
  • Trivially readable by any tool (jq, pandas, etc.)
  • Can be streamed line-by-line — no need to load everything into RAM

Usage
─────
  store = CorpusStore(Path("train"))
  store.append(source="wikipedia", title="Python", text="Python is...")
  store.append(source="gutenberg", title="Moby Dick", text="Call me Ishmael...")

  # Check progress
  print(store.total_chars)   # total chars stored so far
  print(store.total_docs)    # number of documents

  # Load all texts for tokenizer training or dataset building
  texts = store.load_texts()

  # Pre-tokenize to .bin once threshold is reached
  store.build_bin(tokenizer, seq_len=2048)
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

JSONL_FILENAME = "corpus.jsonl"
BIN_FILENAME   = "corpus.bin"
META_FILENAME  = "corpus_meta.json"
WINDOW_META_FILENAME = "window_meta.json"
SHARD_SEPARATOR = "\n\n<<<NANO_LLM_CHUNK_BREAK>>>\n\n"


class CorpusStore:
    """
    Thread-safe append-only store for crawled documents.
    Writes to JSONL; optionally pre-tokenizes to a binary file.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.directory / JSONL_FILENAME
        self._bin_path   = self.directory / BIN_FILENAME
        self._meta_path  = self.directory / META_FILENAME

        self._lock = threading.Lock()

        # Load existing metadata if present
        self._total_chars = 0
        self._total_docs  = 0
        self._bin_built   = False
        self._load_meta()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def total_chars(self) -> int:
        return self._total_chars

    @property
    def total_docs(self) -> int:
        return self._total_docs

    @property
    def bin_path(self) -> Path:
        return self._bin_path

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    @property
    def bin_built(self) -> bool:
        return self._bin_built and self._bin_path.exists()

    # ── Write ──────────────────────────────────────────────────────────

    def append(self, source: str, title: str, text: str) -> None:
        """
        Append one document to the JSONL store.
        Thread-safe — safe to call from multiple crawler threads.
        """
        if not text.strip():
            return

        record = {
            "source": source,
            "title":  title,
            "text":   text,
            "chars":  len(text),
            "ts":     time.time(),
        }
        line = json.dumps(record, ensure_ascii=False)

        with self._lock:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._total_chars += len(text)
            self._total_docs  += 1
            self._save_meta()

    # ── Read ───────────────────────────────────────────────────────────

    def iter_records(self) -> Iterator[dict]:
        """Iterate over all stored records without loading everything into RAM."""
        if not self._jsonl_path.exists():
            return
        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def load_texts(self) -> list[str]:
        """Load all stored texts into a list (for tokenizer training)."""
        return [rec["text"] for rec in self.iter_records()]

    def load_chunks(self, chunk_size: int = 1024, overlap: int = 64) -> list[str]:
        """Load and chunk all texts (for dataset building without pre-tokenization)."""
        from nano_llm.data.cleaner import split_into_chunks
        chunks = []
        for rec in self.iter_records():
            chunks.extend(split_into_chunks(rec["text"], chunk_size, overlap))
        return chunks

    # ── Pre-tokenize to .bin ───────────────────────────────────────────

    def build_bin(self, tokenizer, seq_len: int = 2048) -> Path:
        """
        Tokenize all stored texts and write a flat uint16 .bin file.
        This is the most efficient format for the dataloader — memory-mapped,
        zero parsing overhead, random access.

        Returns the path to the .bin file.
        """
        logger.info(
            f"Building corpus.bin from {self._total_docs:,} docs "
            f"({self._total_chars:,} chars)..."
        )

        all_ids: list[int] = []
        for i, rec in enumerate(self.iter_records()):
            ids = tokenizer.encode(rec["text"], add_bos=True, add_eos=True)
            all_ids.extend(ids)
            if (i + 1) % 500 == 0:
                logger.info(f"  Tokenized {i+1:,} / {self._total_docs:,} docs")

        if not all_ids:
            raise ValueError("No tokens produced — corpus may be empty.")

        arr = np.array(all_ids, dtype=np.uint16)
        arr.tofile(str(self._bin_path))

        with self._lock:
            self._bin_built = True
            self._save_meta()

        logger.info(
            f"corpus.bin written: {len(all_ids):,} tokens → {self._bin_path}"
        )
        return self._bin_path

    # ── Utilities ──────────────────────────────────────────────────────

    def clear(self) -> None:
        """Delete all stored data and reset counters."""
        with self._lock:
            for p in [self._jsonl_path, self._bin_path, self._meta_path]:
                if p.exists():
                    p.unlink()
            self._total_chars = 0
            self._total_docs  = 0
            self._bin_built   = False

    def summary(self) -> str:
        size_mb = (
            self._jsonl_path.stat().st_size / (1024 ** 2)
            if self._jsonl_path.exists() else 0
        )
        bin_mb = (
            self._bin_path.stat().st_size / (1024 ** 2)
            if self._bin_path.exists() else 0
        )
        return (
            f"Docs: {self._total_docs:,} | "
            f"Chars: {self._total_chars:,} ({self._total_chars/1e6:.1f}M) | "
            f"JSONL: {size_mb:.1f} MB"
            + (f" | BIN: {bin_mb:.1f} MB" if bin_mb else "")
        )

    # ── Persistence ────────────────────────────────────────────────────

    def _save_meta(self) -> None:
        meta = {
            "total_chars": self._total_chars,
            "total_docs":  self._total_docs,
            "bin_built":   self._bin_built,
        }
        self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_meta(self) -> None:
        if not self._meta_path.exists():
            # Recount from JSONL if meta is missing
            if self._jsonl_path.exists():
                self._recount()
            return
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._total_chars = meta.get("total_chars", 0)
            self._total_docs  = meta.get("total_docs",  0)
            self._bin_built   = meta.get("bin_built",   False)
        except Exception:
            self._recount()

    def _recount(self) -> None:
        """Recount chars/docs by scanning the JSONL file."""
        chars = 0
        docs  = 0
        for rec in self.iter_records():
            chars += rec.get("chars", len(rec.get("text", "")))
            docs  += 1
        self._total_chars = chars
        self._total_docs  = docs
        self._save_meta()
        logger.info(f"CorpusStore recounted: {docs:,} docs, {chars:,} chars")


class ChunkedTextWindowStore:
    """
    Disk-backed rolling window of live-training text chunks.

    Each round is written to a bounded `.txt` shard file so live training does
    not keep appending to one ever-growing JSONL. Older shard files are pruned
    automatically as the window advances.
    """

    def __init__(self, directory: Path, max_shards: int = 3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_shards = max(1, int(max_shards))
        self._meta_path = self.directory / WINDOW_META_FILENAME
        self._lock = threading.Lock()

        self._total_chars_seen = 0
        self._total_chunks_seen = 0
        self._shards: list[dict] = []
        self._load_meta()
        self._prune_excess_shards()

    @property
    def total_chars_seen(self) -> int:
        return self._total_chars_seen

    @property
    def total_chunks_seen(self) -> int:
        return self._total_chunks_seen

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def append_round(self, round_num: int, chunks: list[str]) -> Optional[Path]:
        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        if not valid_chunks:
            return None

        shard_path = self.directory / f"round_{round_num:04d}.txt"
        payload = SHARD_SEPARATOR.join(valid_chunks)
        chars = sum(len(chunk) for chunk in valid_chunks)

        with self._lock:
            shard_path.write_text(payload, encoding="utf-8")
            self._shards.append(
                {
                    "round": round_num,
                    "path": shard_path.name,
                    "chunks": len(valid_chunks),
                    "chars": chars,
                    "ts": time.time(),
                }
            )
            self._total_chars_seen += chars
            self._total_chunks_seen += len(valid_chunks)
            self._prune_excess_shards()
            self._save_meta()

        return shard_path

    def load_window_chunks(self) -> list[str]:
        chunks: list[str] = []
        for shard in self._shards:
            path = self.directory / shard["path"]
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            chunks.extend(
                piece for piece in text.split(SHARD_SEPARATOR) if piece.strip()
            )
        return chunks

    def summary(self) -> str:
        window_chars = sum(int(shard.get("chars", 0)) for shard in self._shards)
        window_chunks = sum(int(shard.get("chunks", 0)) for shard in self._shards)
        size_mb = sum(
            (self.directory / shard["path"]).stat().st_size
            for shard in self._shards
            if (self.directory / shard["path"]).exists()
        ) / (1024 ** 2)
        return (
            f"Live window: {self.shard_count} shard(s) | "
            f"Window chunks: {window_chunks:,} | "
            f"Window chars: {window_chars:,} ({window_chars/1e6:.1f}M) | "
            f"Disk: {size_mb:.1f} MB"
        )

    def clear(self) -> None:
        with self._lock:
            for path in self.directory.glob("round_*.txt"):
                path.unlink(missing_ok=True)
            if self._meta_path.exists():
                self._meta_path.unlink()
            self._shards = []
            self._total_chars_seen = 0
            self._total_chunks_seen = 0

    def _prune_excess_shards(self) -> None:
        while len(self._shards) > self.max_shards:
            stale = self._shards.pop(0)
            stale_path = self.directory / stale["path"]
            if stale_path.exists():
                stale_path.unlink()

    def _save_meta(self) -> None:
        meta = {
            "max_shards": self.max_shards,
            "total_chars_seen": self._total_chars_seen,
            "total_chunks_seen": self._total_chunks_seen,
            "shards": self._shards,
        }
        self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_meta(self) -> None:
        if not self._meta_path.exists():
            return
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self.max_shards = max(1, int(meta.get("max_shards", self.max_shards)))
            self._total_chars_seen = int(meta.get("total_chars_seen", 0))
            self._total_chunks_seen = int(meta.get("total_chunks_seen", 0))
            self._shards = [
                shard for shard in meta.get("shards", [])
                if (self.directory / shard.get("path", "")).exists()
            ]
        except Exception:
            self._shards = []
            self._total_chars_seen = 0
            self._total_chunks_seen = 0
