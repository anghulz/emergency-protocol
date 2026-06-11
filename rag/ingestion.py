"""Document ingestion: loading, cleaning, and chunking text.

Handles .pdf, .txt, and .md files, either from disk (the pre-loaded
corpus) or from Streamlit file uploads (file-like objects).
"""

import io
import re
from pathlib import Path

from pypdf import PdfReader


class DocumentIngestor:
    """Loads documents and breaks them into overlapping text chunks.

    Chunks overlap so that a sentence falling on a chunk boundary is
    still fully contained in at least one chunk, which improves recall
    during similarity search.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Args:
            chunk_size: target chunk length in characters.
            overlap: characters of overlap between consecutive chunks.
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Loading & cleaning
    # ------------------------------------------------------------------

    def load(self, file, filename: str | None = None) -> str:
        """Extract raw text from a file path or file-like object.

        Args:
            file: a str/Path to a file on disk, or a binary file-like
                object (e.g. a Streamlit UploadedFile).
            filename: required when `file` is a file-like object, used
                to determine the file type.

        Returns:
            The cleaned text content of the document.

        Raises:
            ValueError: if the file type is unsupported or the file
                contains no extractable text.
        """
        if isinstance(file, (str, Path)):
            path = Path(file)
            name = path.name
            data = path.read_bytes()
        else:
            if filename is None:
                raise ValueError("filename is required for file-like objects")
            name = filename
            data = file.read()

        ext = Path(name).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        if ext == ".pdf":
            text = self._extract_pdf_text(data)
        else:
            text = data.decode("utf-8", errors="replace")

        text = self._clean_text(text)
        if not text.strip():
            raise ValueError(f"No extractable text found in '{name}'.")
        return text

    @staticmethod
    def _extract_pdf_text(data: bytes) -> str:
        """Extract text from PDF bytes, page by page."""
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and strip non-printable characters."""
        # Remove control characters except newlines/tabs.
        text = re.sub(r"[^\S\n\t]+", " ", text)          # collapse runs of spaces
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)           # collapse blank lines
        return text.strip()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk(self, text: str, source: str) -> list[dict]:
        """Split text into overlapping chunks, preferring paragraph breaks.

        The splitter walks forward `chunk_size` characters at a time, then
        backtracks to the nearest paragraph or sentence boundary so chunks
        don't cut sentences in half mid-thought.

        Args:
            text: the full cleaned document text.
            source: document name, stored with each chunk for citations.

        Returns:
            A list of dicts: {"id", "text", "source"}.
        """
        chunks = []
        start = 0
        index = 0
        n = len(text)

        while start < n:
            end = min(start + self.chunk_size, n)

            if end < n:
                # Prefer to break at a paragraph, then a sentence end.
                window = text[start:end]
                break_at = window.rfind("\n\n")
                if break_at < self.chunk_size // 2:
                    break_at = window.rfind(". ")
                if break_at >= self.chunk_size // 2:
                    end = start + break_at + 1

            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    {"id": f"{source}::chunk{index}", "text": piece, "source": source}
                )
                index += 1

            if end >= n:
                break
            start = max(end - self.overlap, start + 1)

        return chunks

    def ingest(self, file, filename: str | None = None) -> list[dict]:
        """Convenience method: load a file and return its chunks."""
        if isinstance(file, (str, Path)):
            source = Path(file).name
        else:
            source = filename
        text = self.load(file, filename)
        return self.chunk(text, source)
