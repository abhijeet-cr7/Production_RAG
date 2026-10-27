"""Turn a *sanitised* HTML tree into ordered blocks, keeping tables intact."""

from __future__ import annotations

import re
from typing import Iterator

from bs4 import BeautifulSoup, NavigableString, Tag

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.tables.blocks import (
    HEADING,
    MAX_TABLE_CELLS,
    PARAGRAPH,
    TABLE,
    Block,
    Table,
    clean_cell,
    infer_header_rows,
)
from offline_pipeline.tables.numbers import detect_units

_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_BLOCK_TAGS = {"p", "li", "div", "section", "article", "main", "blockquote", "pre", "dd", "dt"}
_FOOTNOTE = re.compile(r"^\s*(?:\(\d{1,2}\)|\[\d{1,2}\]|\*+|[†‡]|note[s]?\b|source\b)", re.I)


def _text(tag: Tag) -> str:
    return clean_cell(tag.get_text(" ", strip=True))


def parse_table(tag: Tag) -> Table | None:
    """Expand colspan/rowspan into a rectangular grid of cleaned cell text."""
    grid: list[list[str]] = []
    header_flags: list[bool] = []
    pending: dict[tuple[int, int], str] = {}   # (row, col) -> text carried down by rowspan
    cells_seen = 0

    for r, tr in enumerate(tag.find_all("tr")):
        if tr.find_parent("table") is not tag:  # a nested table's rows belong to it
            continue
        row_index = len(grid)
        row: list[str] = []
        col = 0
        is_header = bool(tr.find_parent("thead")) or (
            bool(tr.find_all(["th"])) and not tr.find_all("td")
        )
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (row_index, col) in pending:
                row.append(pending.pop((row_index, col)))
                col += 1
            text = _text(cell)
            span = max(1, min(int(cell.get("colspan", 1) or 1), 50)) if str(cell.get("colspan", "1")).isdigit() else 1
            rspan = max(1, min(int(cell.get("rowspan", 1) or 1), 50)) if str(cell.get("rowspan", "1")).isdigit() else 1
            for k in range(span):
                row.append(text if k == 0 else "")
                if rspan > 1:
                    for extra in range(1, rspan):
                        pending[(row_index + extra, col)] = text if k == 0 else ""
                col += 1
            cells_seen += span
            if cells_seen > MAX_TABLE_CELLS:
                raise UnsafeContent("table_too_large", "Table has too many cells.")
        while (row_index, col) in pending:
            row.append(pending.pop((row_index, col)))
            col += 1
        if any(c for c in row):
            grid.append(row)
            header_flags.append(is_header)

    if len(grid) < 2:
        return None
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]

    header_rows = [r for r, h in zip(grid, header_flags) if h]
    body = [r for r, h in zip(grid, header_flags) if not h]
    if not header_rows:
        header_rows, body = infer_header_rows(grid)
    if not body:
        return None

    caption_tag = tag.find("caption")
    caption = _text(caption_tag) if caption_tag else ""
    hint_text = " ".join([caption] + [" ".join(r) for r in header_rows[:2]])
    scale, currency = detect_units(hint_text)
    return Table(caption=caption, header_rows=header_rows, rows=body, scale=scale, currency=currency)


def _walk(node: Tag) -> Iterator[Tag | str]:
    """Yield headings, tables and text-bearing leaf blocks in document order."""
    for child in node.children:
        if isinstance(child, NavigableString):
            continue
        name = (child.name or "").lower()
        if name == "table":
            yield child
        elif name in _HEADINGS:
            yield child
        elif name in _BLOCK_TAGS:
            # A block containing tables or headings is descended into; otherwise it is text.
            if child.find(["table", *_HEADINGS]):
                yield from _walk(child)
            else:
                yield child
        else:
            yield from _walk(child)


def html_to_blocks(soup: BeautifulSoup) -> list[Block]:
    root = soup.body or soup
    blocks: list[Block] = []
    try:
        for node in _walk(root):
            name = (node.name or "").lower()
            if name == "table":
                table = parse_table(node)
                if table is not None:
                    blocks.append(Block(kind=TABLE, table=table))
                else:
                    text = _text(node)
                    if text:
                        blocks.append(Block(kind=PARAGRAPH, text=text))
            elif name in _HEADINGS:
                text = _text(node)
                if text:
                    blocks.append(Block(kind=HEADING, text=text, level=_HEADINGS[name]))
            else:
                text = _text(node)
                if not text:
                    continue
                previous = blocks[-1] if blocks else None
                if previous is not None and previous.kind == TABLE and _FOOTNOTE.match(text):
                    previous.table.footnotes.append(text)  # type: ignore[union-attr]
                else:
                    blocks.append(Block(kind=PARAGRAPH, text=text))
    except RecursionError as exc:
        raise UnsafeContent("html_too_deep", "HTML is nested too deeply.") from exc

    # A paragraph directly above a table is usually its caption / units line.
    for i, block in enumerate(blocks[:-1]):
        nxt = blocks[i + 1]
        if (block.kind == PARAGRAPH and nxt.kind == TABLE and nxt.table is not None
                and not nxt.table.caption and len(block.text) < 200):
            nxt.table.caption = block.text
            scale, currency = detect_units(block.text)
            nxt.table.scale = nxt.table.scale or scale
            nxt.table.currency = nxt.table.currency or currency
            block.text = ""
    return [b for b in blocks if b.kind != PARAGRAPH or b.text]
