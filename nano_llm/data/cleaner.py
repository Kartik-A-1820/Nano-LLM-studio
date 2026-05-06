"""Text normalization and cleaning utilities."""
import re
import unicodedata


def remove_control_chars(text: str) -> str:
    """Strip unicode control characters except newline and tab."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )


def fix_encoding_artifacts(text: str) -> str:
    """Fix encoding artifacts using ftfy if available, else basic replacements."""
    try:
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        replacements = {
            "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--",
            "\u2026": "...", "\u00a0": " ",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs; preserve single newlines; collapse 3+ newlines to 2."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def deduplicate_lines(text: str, min_line_len: int = 20) -> str:
    """Remove exact duplicate lines (common in scraped PDFs with repeated headers/footers)."""
    seen = set()
    output = []
    for line in text.splitlines():
        key = line.strip()
        if len(key) >= min_line_len:
            if key in seen:
                continue
            seen.add(key)
        output.append(line)
    return "\n".join(output)


def clean_text(text: str) -> str:
    """Full cleaning pipeline: encoding -> control chars -> whitespace -> dedup."""
    text = fix_encoding_artifacts(text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)
    text = deduplicate_lines(text)
    return text


def split_into_chunks(text: str, chunk_size: int = 2048, overlap: int = 64) -> list[str]:
    """
    Character-level sliding window chunking for dataset construction.
    Each chunk is at most chunk_size characters, with overlap between chunks.
    Tries to split on word boundaries.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            # Try to break on a word boundary
            last_space = chunk.rfind(" ", chunk_size // 2)
            if last_space != -1:
                chunk = chunk[:last_space]
                end = start + last_space

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap if end < len(text) else len(text)

    return chunks
