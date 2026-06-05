"""Text cleaner: normalises raw extracted text before chunking."""

from __future__ import annotations

import re
import unicodedata


class TextCleaner:
    """Apply a configurable sequence of cleaning steps to raw text."""

    def clean(self, text: str) -> str:
        """Return a cleaned version of *text*."""
        text = self._normalise_unicode(text)
        text = self._remove_control_chars(text)
        text = self._collapse_whitespace(text)
        return text.strip()

    # ── steps ────────────────────────────────────────────────────────────────

    def _normalise_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def _remove_control_chars(self, text: str) -> str:
        # Keep newlines (\n) and tabs (\t) for structural context
        return re.sub(r"[^\S\n\t](?<!\s)|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    def _collapse_whitespace(self, text: str) -> str:
        # Collapse multiple blank lines into at most two
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces on a single line
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text
