"""Document extractor: handles PDF, DOCX and plain-text files.

Supports native text extraction with a Tesseract OCR fallback for
scanned / image-only pages.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extract raw text and basic metadata from local document files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Extract text and metadata from *file_path*.

        Args:
            file_path: Path to the document to be extracted.

        Returns:
            A dict with keys ``text`` (str) and ``metadata`` (dict).

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If *file_path* does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if ext == ".pdf":
            return self._extract_pdf(path)
        if ext == ".docx":
            return self._extract_docx(path)
        return self._extract_text(path)

    # ── private helpers ──────────────────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> dict[str, Any]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required for PDF extraction.") from exc

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if not page_text.strip():
                page_text = self._ocr_page(page)
            pages.append(page_text)

        return {
            "text": "\n".join(pages),
            "metadata": {
                "source": str(path),
                "file_type": "pdf",
                "page_count": len(reader.pages),
            },
        }

    def _extract_docx(self, path: Path) -> dict[str, Any]:
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError(
                "python-docx is required for DOCX extraction."
            ) from exc

        doc = Document(str(path))
        text = "\n".join(para.text for para in doc.paragraphs)
        return {
            "text": text,
            "metadata": {"source": str(path), "file_type": "docx"},
        }

    def _extract_text(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "text": text,
            "metadata": {"source": str(path), "file_type": "txt"},
        }

    def _ocr_page(self, page: Any) -> str:
        """Run Tesseract OCR on a PDF page rendered as an image."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.warning("pytesseract/Pillow not installed; skipping OCR.")
            return ""

        try:
            for image_file in page.images:
                img = Image.open(io.BytesIO(image_file.data))
                return pytesseract.image_to_string(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed: %s", exc)
        return ""
