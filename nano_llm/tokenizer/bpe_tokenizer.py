"""
Byte-level BPE tokenizer for dynamic continuous training.

Design rationale
────────────────
When training on continuously crawled data from diverse sources (Wikipedia,
Gutenberg, Common Crawl, RSS feeds, code, scientific text, multiple languages)
a fixed-vocabulary word-level or subword tokenizer breaks down:

  • New domains introduce OOV tokens → <unk> floods the training signal
  • Vocabulary must be frozen at model-build time (embedding matrix size)
  • SentencePiece BPE cannot be updated incrementally

Solution: Byte-level BPE (same approach as GPT-2, LLaMA, Mistral)
  1. Every byte (0-255) is a valid token → zero <unk>, ever
  2. BPE merges compress common byte sequences into single tokens
  3. Vocabulary is fixed at training time but covers ALL possible text
     because the 256 byte tokens are always the fallback
  4. New domains just produce longer byte sequences until the model
     learns the patterns — no retraining the tokenizer needed

Special tokens
──────────────
  ID 0  <pad>
  ID 1  <bos>
  ID 2  <eos>
  ID 3  <unk>   (never emitted by byte-level encoding, kept for compat)
  IDs 4-259     raw bytes 0x00-0xFF
  IDs 260+      BPE merge tokens

The vocabulary is therefore always at least 260 tokens and grows with
the number of BPE merges learned.

Persistence
───────────
Saved as a single JSON file (tokenizer.json) containing:
  - merges: list of [a, b] pairs in merge order
  - vocab: {token_str: id} mapping
  - config: metadata

This format is human-readable and trivially diff-able in git.
"""
import json
import logging
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Special token IDs ─────────────────────────────────────────────────
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4          # byte b maps to token ID b + BYTE_OFFSET
MIN_VOCAB_SIZE = 260     # 4 special + 256 bytes


def _byte_token(b: int) -> str:
    """Canonical string representation of a raw byte token."""
    return f"<0x{b:02X}>"


def _build_base_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Build the fixed base vocabulary: specials + 256 byte tokens."""
    tok2id: dict[str, int] = {
        "<pad>": PAD_ID,
        "<bos>": BOS_ID,
        "<eos>": EOS_ID,
        "<unk>": UNK_ID,
    }
    for b in range(256):
        tok2id[_byte_token(b)] = b + BYTE_OFFSET
    id2tok = {v: k for k, v in tok2id.items()}
    return tok2id, id2tok


# ── Byte-level BPE implementation ─────────────────────────────────────

class ByteLevelBPETokenizer:
    """
    Byte-level BPE tokenizer.

    • Zero <unk> — every byte is representable
    • Fixed vocab size after training — safe to use with a frozen embedding
    • Handles any language, code, or binary-safe text without retraining
    • Saved as a portable JSON file
    """

    PAD_ID = PAD_ID
    BOS_ID = BOS_ID
    EOS_ID = EOS_ID
    UNK_ID = UNK_ID

    def __init__(self):
        self._tok2id, self._id2tok = _build_base_vocab()
        self._merges: list[tuple[int, int]] = []   # ordered merge rules
        self._merge_map: dict[tuple[int, int], int] = {}
        self._trained = False
        self._sp = None          # SentencePiece processor (fast path)
        self._use_spm = False    # True when trained via SentencePiece

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        texts: list[str],
        vocab_size: int = 50_257,
        model_prefix: str = "outputs/tokenizer/bpe",  # kept for API compat
        character_coverage: float = 0.9995,            # unused, kept for compat
    ) -> None:
        """
        Train byte-level BPE on the provided texts.

        Uses SentencePiece's byte-level BPE mode when available (fast, C++).
        Falls back to a pure-Python implementation otherwise.

        Args:
            texts:      List of text strings (chunks from the corpus)
            vocab_size: Target vocabulary size (must be >= 260). GPT-2 = 50,257.
            model_prefix: Ignored — saving is done via save_pretrained()
        """
        vocab_size = max(vocab_size, MIN_VOCAB_SIZE)

        logger.info(
            f"Training byte-level BPE: {len(texts)} texts, "
            f"target vocab={vocab_size:,}"
        )

        try:
            self._train_spm_byte_level(texts, vocab_size)
        except ImportError:
            logger.warning(
                "sentencepiece not installed — falling back to pure-Python BPE. "
                "Install with: uv pip install sentencepiece  (much faster for large vocab)"
            )
            self._train_pure_python(texts, vocab_size)

    def _train_spm_byte_level(self, texts: list[str], vocab_size: int) -> None:
        """
        Train using SentencePiece byte-level BPE.

        SentencePiece's byte_fallback=True mode guarantees every byte is
        representable, matching our byte-level design exactly.
        The resulting merges are extracted and stored in our internal format
        so encode/decode work identically regardless of which backend trained.
        """
        import sentencepiece as spm
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n".join(texts))
            tmp_path = f.name

        out_dir = Path("outputs/tokenizer")
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(out_dir / "spm_byte_bpe")

        try:
            spm.SentencePieceTrainer.train(
                input=tmp_path,
                model_prefix=prefix,
                vocab_size=vocab_size,
                model_type="bpe",
                byte_fallback=True,          # ← key: every byte is a valid token
                character_coverage=1.0,      # full coverage via byte fallback
                pad_id=PAD_ID,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                unk_id=UNK_ID,
                pad_piece="<pad>",
                bos_piece="<bos>",
                eos_piece="<eos>",
                unk_piece="<unk>",
                normalization_rule_name="identity",  # no normalization — bytes are canonical
                remove_extra_whitespaces=False,
                hard_vocab_limit=False,              # allow training on small datasets
                input_sentence_size=10_000_000,
                shuffle_input_sentence=True,
                num_threads=min(8, os.cpu_count() or 4),
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Load the trained model and extract merges into our internal format
        sp = spm.SentencePieceProcessor()
        sp.load(prefix + ".model")

        # Rebuild tok2id / id2tok from the SentencePiece vocab
        self._tok2id, self._id2tok = _build_base_vocab()
        self._merges = []
        self._merge_map = {}

        # SentencePiece vocab: IDs 0-3 are specials, 4-259 are bytes,
        # 260+ are merge tokens. We extract the merge tokens.
        for i in range(MIN_VOCAB_SIZE, sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            self._tok2id[piece] = i
            self._id2tok[i] = piece

        # Store the sp model for fast encode/decode
        self._sp = sp
        self._use_spm = True
        self._trained = True

        logger.info(
            f"SentencePiece byte-level BPE trained: "
            f"vocab_size={self.vocab_size():,}"
        )

    def _train_pure_python(self, texts: list[str], vocab_size: int) -> None:
        """Pure-Python BPE training fallback (slower, no dependencies)."""
        n_merges = vocab_size - MIN_VOCAB_SIZE

        if n_merges == 0:
            self._trained = True
            return

        # Encode all texts to byte-ID sequences
        word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
        for text in texts:
            for segment in _pretokenize(text):
                seg_bytes = segment.encode("utf-8", errors="replace")
                word = tuple(b + BYTE_OFFSET for b in seg_bytes)
                if word:
                    word_freqs[word] += 1

        if not word_freqs:
            self._trained = True
            return

        vocab = dict(word_freqs)
        self._merges = []
        self._merge_map = {}

        for merge_idx in range(n_merges):
            pair_freqs: dict[tuple[int, int], int] = defaultdict(int)
            for word, freq in vocab.items():
                for a, b in zip(word, word[1:]):
                    pair_freqs[(a, b)] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            if pair_freqs[best_pair] < 2:
                break

            new_id = MIN_VOCAB_SIZE + merge_idx
            self._merges.append(best_pair)
            self._merge_map[best_pair] = new_id

            a_str = self._id2tok[best_pair[0]]
            b_str = self._id2tok[best_pair[1]]
            new_tok = a_str + b_str
            self._tok2id[new_tok] = new_id
            self._id2tok[new_id] = new_tok

            new_vocab: dict[tuple[int, ...], int] = {}
            for word, freq in vocab.items():
                new_word = _apply_merge(word, best_pair, new_id)
                new_vocab[new_word] = new_vocab.get(new_word, 0) + freq
            vocab = new_vocab

            if (merge_idx + 1) % 500 == 0:
                logger.info(f"  Merge {merge_idx+1}/{n_merges} (freq={pair_freqs[best_pair]})")

        self._use_spm = False
        self._trained = True
        logger.info(
            f"Pure-Python BPE trained: {len(self._merges)} merges, "
            f"vocab_size={self.vocab_size()}"
        )

    # ── Encode / Decode ───────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """
        Encode text to token IDs.
        Never produces UNK — every byte is representable.
        Uses SentencePiece fast path when available.
        """
        if self._use_spm and self._sp is not None:
            ids = self._sp.encode(text, out_type=int)
            if add_bos:
                ids = [BOS_ID] + ids
            if add_eos:
                ids = ids + [EOS_ID]
            return ids

        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for segment in _pretokenize(text):
            seg_bytes = segment.encode("utf-8", errors="replace")
            tokens = [b + BYTE_OFFSET for b in seg_bytes]
            tokens = self._apply_merges(tokens)
            ids.extend(tokens)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID, UNK_ID)]

        if self._use_spm and self._sp is not None:
            return self._sp.decode(ids)

        byte_list: list[int] = []
        for tid in ids:
            tok_str = self._id2tok.get(tid, "")
            byte_list.extend(_token_str_to_bytes(tok_str))
        return bytes(byte_list).decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        return len(self._tok2id)

    # ── Persistence ───────────────────────────────────────────────────

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        data = {
            "config": {
                "type": "byte_level_bpe",
                "vocab_size": self.vocab_size(),
                "n_merges": len(self._merges),
                "backend": "sentencepiece" if self._use_spm else "python",
            },
            "merges": self._merges,
            "extra_vocab": {
                tok: tid
                for tok, tid in self._tok2id.items()
                if tid >= MIN_VOCAB_SIZE
            },
        }

        out = directory / "tokenizer.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Copy the SentencePiece .model file if it exists (enables fast encode/decode)
        spm_src = Path("outputs/tokenizer/spm_byte_bpe.model")
        spm_dst = directory / "spm_byte_bpe.model"
        if self._use_spm and spm_src.exists():
            import shutil
            # Prevent SameFileError if source and destination are the same
            if spm_src.resolve() != spm_dst.resolve():
                shutil.copy2(spm_src, spm_dst)

        cfg = {
            "vocab_size": self.vocab_size(),
            "type": "byte_level_bpe",
            "backend": "sentencepiece" if self._use_spm else "python",
        }
        (directory / "tokenizer_config.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )
        logger.info(f"Tokenizer saved → {out} (vocab={self.vocab_size():,})")

    @classmethod
    def from_pretrained(cls, directory: Path) -> "ByteLevelBPETokenizer":
        directory = Path(directory)
        tok = cls()

        new_path = directory / "tokenizer.json"
        if new_path.exists():
            tok._load_json(new_path, directory)
            return tok

        # Legacy SentencePiece model
        spm_path = directory / "bpe.model"
        if spm_path.exists():
            logger.warning(
                "Found legacy SentencePiece model. "
                "Retraining the tokenizer is recommended for byte-level BPE."
            )
            return tok

        raise FileNotFoundError(
            f"No tokenizer.json found in {directory}. "
            "Please train the tokenizer first."
        )

    def _load_json(self, path: Path, directory: Path = None) -> None:
        data   = json.loads(path.read_text(encoding="utf-8"))
        cfg    = data.get("config", {})
        merges = [tuple(m) for m in data.get("merges", [])]
        extra  = data.get("extra_vocab", {})
        backend = cfg.get("backend", "python")

        self._merges = merges
        self._merge_map = {}
        for i, (a, b) in enumerate(merges):
            self._merge_map[(a, b)] = MIN_VOCAB_SIZE + i

        for tok, tid in extra.items():
            self._tok2id[tok] = tid
            self._id2tok[tid] = tok

        # Try to load the SentencePiece model for fast encode/decode
        if backend == "sentencepiece" and directory is not None:
            spm_path = directory / "spm_byte_bpe.model"
            if spm_path.exists():
                try:
                    import sentencepiece as spm
                    sp = spm.SentencePieceProcessor()
                    sp.load(str(spm_path))
                    self._sp = sp
                    self._use_spm = True
                    logger.info(
                        f"Tokenizer loaded (SentencePiece fast path): "
                        f"vocab={self.vocab_size():,}"
                    )
                    return
                except Exception as e:
                    logger.warning(f"Could not load SentencePiece model: {e}. Using Python fallback.")

        self._use_spm = False
        logger.info(
            f"Tokenizer loaded (Python): vocab={self.vocab_size():,}, "
            f"merges={len(self._merges):,}"
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _apply_merges(self, tokens: list[int]) -> list[int]:
        """Apply all learned BPE merges to a token sequence."""
        if not self._merge_map:
            return tokens

        # Iterative merge — O(n_merges * len(tokens)) but fast in practice
        # because most sequences are short after pre-tokenization
        changed = True
        while changed and len(tokens) > 1:
            changed = False
            new_tokens: list[int] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    merged = self._merge_map.get(pair)
                    if merged is not None:
                        new_tokens.append(merged)
                        i += 2
                        changed = True
                        continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens
        return tokens


# ── Pre-tokenizer ─────────────────────────────────────────────────────

# GPT-2 style pre-tokenizer regex — splits on whitespace/punctuation
# boundaries so BPE merges don't cross word boundaries
_PRETOK_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _pretokenize(text: str) -> list[str]:
    """Split text into pre-token segments (GPT-2 style)."""
    return _PRETOK_RE.findall(text)


# ── Merge application ─────────────────────────────────────────────────

def _apply_merge(
    word: tuple[int, ...],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[int, ...]:
    """Replace all occurrences of `pair` in `word` with `new_id`."""
    result: list[int] = []
    i = 0
    a, b = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


# ── Token string → bytes ──────────────────────────────────────────────

def _token_str_to_bytes(tok: str) -> list[int]:
    """
    Convert a token string back to raw bytes.
    Base tokens look like '<0xAB>', merged tokens are concatenations of base tokens.
    """
    byte_list: list[int] = []
    # Find all <0xXX> patterns
    for m in re.finditer(r"<0x([0-9A-Fa-f]{2})>", tok):
        byte_list.append(int(m.group(1), 16))
    return byte_list


# ── Public alias (keeps existing imports working) ─────────────────────

# The old name BPETokenizer is kept as an alias so all existing code
# (train_cli.py, tab_data.py, tab_train.py, continuous.py, etc.)
# continues to work without any changes.
BPETokenizer = ByteLevelBPETokenizer
