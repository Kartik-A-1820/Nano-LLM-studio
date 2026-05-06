"""Text extraction from TXT, PDF, and DOCX files."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".jsonl"}


def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        pass

    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except ImportError:
        raise ImportError(
            "PDF extraction requires PyMuPDF (pip install PyMuPDF) "
            "or pdfplumber (pip install pdfplumber)."
        )


def extract_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise ImportError("DOCX extraction requires python-docx (pip install python-docx).")


def extract_jsonl(path: Path) -> str:
    """Extract text from a JSONL corpus file (CorpusStore format)."""
    import json
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return "\n\n".join(texts)


def extract_file(path: Path) -> str:
    """Dispatch extraction by file suffix. Raises ValueError for unsupported types."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return extract_txt(path)
    elif suffix == ".pdf":
        return extract_pdf(path)
    elif suffix == ".docx":
        return extract_docx(path)
    elif suffix == ".jsonl":
        return extract_jsonl(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")


def ingest_folder(folder: Path) -> list[tuple[Path, str]]:
    """
    Recursively finds and extracts text from all supported files in folder.
    Returns list of (path, text) pairs. Logs failures without crashing.
    """
    folder = Path(folder)
    results = []
    patterns = ["*.txt", "*.pdf", "*.docx", "*.md", "*.jsonl"]
    found = []
    for pattern in patterns:
        found.extend(folder.rglob(pattern))
    found = sorted(set(found))

    for path in found:
        try:
            text = extract_file(path)
            if text.strip():
                results.append((path, text))
                logger.info(f"Extracted {len(text):,} chars from {path.name}")
            else:
                logger.warning(f"Empty content: {path.name}")
        except Exception as e:
            logger.error(f"Failed to extract {path.name}: {e}")

    return results
