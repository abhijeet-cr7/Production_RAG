"""Web extractor: scrapes text content from URLs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WebExtractor:
    """Scrape text and metadata from a web page URL."""

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    def extract(self, url: str) -> dict[str, Any]:
        """Fetch *url* and return cleaned text plus metadata.

        Args:
            url: The URL to scrape.

        Returns:
            A dict with keys ``text`` (str) and ``metadata`` (dict).
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "requests and beautifulsoup4 are required for web extraction."
            ) from exc

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string.strip() if soup.title else ""

        return {
            "text": text,
            "metadata": {
                "source": url,
                "file_type": "web",
                "title": title,
                "content_type": response.headers.get("content-type", ""),
            },
        }
