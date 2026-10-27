"""Structured document blocks: prose, headings, tables and XBRL fact groups.

Extractors used to return one flat string, which destroys tables (cells end up on
separate lines, column alignment is collapsed). They now return an ordered list of
blocks so the chunker can keep a table whole and the verifier can check numbers
against cell values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

HEADING = "heading"
PARAGRAPH = "paragraph"
TABLE = "table"
FACTS = "facts"

MAX_CELL_CHARS = 500
MAX_TABLE_CELLS = 200_000


@dataclass
class Table:
    caption: str = ""
    header_rows: list[list[str]] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    units: str = ""
    scale: int = 0
    currency: str = ""
    footnotes: list[str] = field(default_factory=list)
    page: int | None = None
    confidence: str = "high"            # "low" for OCR / inferred structure

    @property
    def column_count(self) -> int:
        widths = [len(r) for r in self.header_rows + self.rows]
        return max(widths) if widths else 0

    def column_headers(self) -> list[str]:
        """One label per column; stacked header rows are joined with ' / '."""
        labels: list[str] = []
        for col in range(self.column_count):
            parts: list[str] = []
            for row in self.header_rows:
                cell = row[col].strip() if col < len(row) else ""
                if cell and (not parts or parts[-1] != cell):
                    parts.append(cell)
            labels.append(" / ".join(parts) or ("Item" if col == 0 else f"column {col + 1}"))
        return labels


@dataclass
class Fact:
    concept: str                         # e.g. "us-gaap:Revenues"
    label: str
    value: str | None                    # exact Decimal as text; None if unparseable
    display: str                         # what the document showed
    unit: str = ""
    period: str = ""                     # "2023-10-01..2024-09-28" or "2024-09-28"
    period_end: str = ""
    decimals: str = ""
    statement: str = "other"             # income | balance | cashflow | other
    dimensions: tuple[str, ...] = ()
    entity: str = ""


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0                       # heading level 1-6
    table: Table | None = None
    facts: list[Fact] = field(default_factory=list)
    page: int | None = None


def clean_cell(text: str) -> str:
    """Normalise one table cell without touching anything outside it."""
    import re
    import unicodedata

    from offline_pipeline.safety.injection import INVISIBLE_CHARS

    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE_CHARS.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:MAX_CELL_CHARS]


def blocks_to_text(blocks: list[Block]) -> str:
    """Flatten blocks for scanning and for callers that only want text."""
    parts: list[str] = []
    for block in blocks:
        if block.kind in (HEADING, PARAGRAPH):
            parts.append(block.text)
        elif block.kind == TABLE and block.table is not None:
            t = block.table
            if t.caption:
                parts.append(t.caption)
            for row in t.header_rows + t.rows:
                parts.append(" | ".join(row))
            parts.extend(t.footnotes)
        elif block.kind == FACTS:
            parts.extend(f"{f.label} {f.display}" for f in block.facts)
    return "\n".join(parts)


def blocks_to_json(blocks: list[Block]) -> list[dict[str, Any]]:
    return [asdict(b) for b in blocks]


def blocks_from_json(items: list[dict[str, Any]]) -> list[Block]:
    out: list[Block] = []
    for item in items:
        table = item.get("table")
        facts = [Fact(**{**f, "dimensions": tuple(f.get("dimensions", ()))}) for f in item.get("facts", [])]
        out.append(Block(
            kind=item["kind"], text=item.get("text", ""), level=item.get("level", 0),
            table=Table(**table) if table else None, facts=facts, page=item.get("page"),
        ))
    return out


def infer_header_rows(grid: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """Split a grid with no explicit header markup into (header rows, body rows).

    Leading rows whose non-label cells contain no digits are the header (at most
    three). If the very first row already has numbers, the first row is used as
    the header anyway so every column still gets a label.
    """
    import re

    from offline_pipeline.tables.numbers import parse_cell_number

    def has_data(row: list[str]) -> bool:
        # "Q1", "FY2024" and bare years are column labels, not values.
        return any(
            parse_cell_number(c) is not None and not re.fullmatch(r"(?:19|20)\d{2}", c.strip())
            for c in row[1:] if c
        )

    split = 0
    for row in grid[:3]:
        if has_data(row):
            break
        split += 1
    split = max(1, split)
    return grid[:split], grid[split:]


def trim_grid(grid: list[list[str]]) -> list[list[str]]:
    """Drop empty rows and empty trailing columns; pad rows to one width."""
    rows = [[clean_cell(c or "") for c in row] for row in grid]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    while width > 1 and not any(r[width - 1] for r in rows):
        width -= 1
        rows = [r[:width] for r in rows]
    return rows
