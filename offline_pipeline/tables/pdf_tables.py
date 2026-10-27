"""Find borderless numeric tables (financial statements) from word positions.

Most statements in filings have no ruling lines, so line-based detection finds
nothing and generic "text alignment" detection mis-reads captions and drops rows.
This detector is built for the shape these tables actually have:

* a row is a *label* (words on the left) followed by *numeric cells only*;
* the numeric cells' right edges line up into the same few columns on every row;
* at least three such rows in a run, optionally with a header line above them.

Prose that happens to contain digits fails both tests (numbers are interleaved
with words, and their positions are all over the line).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from offline_pipeline.tables.numbers import parse_cell_number

LINE_TOLERANCE = 3.0      # points: words within this vertical distance share a line
COLUMN_TOLERANCE = 22.0   # points: right edges within this belong to one column
MIN_ROWS = 3
MAX_COLUMNS = 10
_PERIOD_WORDS = {"fy", "q1", "q2", "q3", "q4", "h1", "h2", "ended", "months", "year", "quarter",
                 "restated", "change", "%", "total", "actual", "budget", "forecast"}


@dataclass
class _Line:
    top: float
    bottom: float
    words: list[dict[str, Any]]


def _is_numeric(text: str) -> bool:
    return parse_cell_number(text) is not None


def _merge_symbols(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Glue a lone ``$`` / ``(`` to the number it belongs to ("$ 1,234" is one cell)."""
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w["text"] in {"$", "€", "£", "(", "($"} and i + 1 < len(words):
            nxt = words[i + 1]
            merged.append({**nxt, "text": w["text"] + nxt["text"], "x0": w["x0"]})
            i += 2
        else:
            merged.append(w)
            i += 1
    return merged


def _group_lines(words: list[dict[str, Any]]) -> list[_Line]:
    lines: list[_Line] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(word["top"] - lines[-1].top) <= LINE_TOLERANCE:
            lines[-1].words.append(word)
            lines[-1].bottom = max(lines[-1].bottom, word["bottom"])
        else:
            lines.append(_Line(word["top"], word["bottom"], [word]))
    for line in lines:
        line.words.sort(key=lambda w: w["x0"])
        line.words = _merge_symbols(line.words)
    return lines


def _split_row(line: _Line) -> tuple[str, list[dict[str, Any]]] | None:
    """(label, numeric cells) if the line has the label-then-numbers shape, else None."""
    first_numeric = next((i for i, w in enumerate(line.words) if _is_numeric(w["text"])), None)
    if first_numeric is None:
        return None
    label_words, cells = line.words[:first_numeric], line.words[first_numeric:]
    if len(cells) < 1 or any(not _is_numeric(w["text"]) for w in cells):
        return None  # a word after the first number: this is prose, not a table row
    return " ".join(w["text"] for w in label_words).strip(), cells


def _columns(rows: list[list[dict[str, Any]]]) -> list[float]:
    """Cluster the right edges of all numeric cells into column positions."""
    edges = sorted(w["x1"] for cells in rows for w in cells)
    clusters: list[list[float]] = []
    for edge in edges:
        if clusters and edge - clusters[-1][-1] <= COLUMN_TOLERANCE:
            clusters[-1].append(edge)
        else:
            clusters.append([edge])
    return [sum(c) / len(c) for c in clusters]


def _nearest(columns: list[float], x1: float) -> int:
    return min(range(len(columns)), key=lambda i: abs(columns[i] - x1))


def _header_cells(line: _Line, columns: list[float]) -> list[str] | None:
    """Assign a header line's words to the columns it sits above, or None if it does not."""
    words = [w for w in line.words if w["text"].strip()]
    if len(words) < 2:
        return None
    cells = [""] * len(columns)
    placed = 0
    for w in words:
        idx = _nearest(columns, w["x1"])
        if abs(columns[idx] - w["x1"]) <= COLUMN_TOLERANCE * 1.5:
            cells[idx] = (cells[idx] + " " + w["text"]).strip()
            placed += 1
    return cells if placed >= max(2, len(words) // 2) else None


def detect_numeric_tables(words: list[dict[str, Any]]) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    """Return ``(bbox, grid)`` for each numeric table; the grid's first row is its header."""
    lines = _group_lines(words)
    parsed = [_split_row(line) for line in lines]

    tables = []
    i = 0
    while i < len(lines):
        if parsed[i] is None:
            i += 1
            continue
        # A run of data rows, tolerating one label-only line (a section heading such as
        # "Assets") between them.
        run = [i]
        j = i + 1
        while j < len(lines):
            if parsed[j] is not None:
                run.append(j)
                j += 1
            elif j + 1 < len(lines) and parsed[j + 1] is not None and len(lines[j].words) <= 6 \
                    and lines[j].top - lines[run[-1]].bottom < 14:
                run.append(j)
                j += 1
            else:
                break
        data = [k for k in run if parsed[k] is not None]
        if len(data) >= MIN_ROWS:
            built = _build(lines, parsed, run, data, i)
            if built is not None:
                tables.append(built)
        i = j if j > i else i + 1
    return tables


def _build(
    lines: list[_Line], parsed: list[Any], run: list[int], data: list[int], first: int
) -> tuple[tuple[float, float, float, float], list[list[str]]] | None:
    columns = _columns([parsed[k][1] for k in data])
    if not 1 <= len(columns) <= MAX_COLUMNS:
        return None
    usage = [0] * len(columns)
    for k in data:
        for cell in parsed[k][1]:
            usage[_nearest(columns, cell["x1"])] += 1
    # Real columns are used by most rows; scattered prose numbers are not.
    if sum(1 for u in usage if u >= max(2, len(data) // 2)) < max(1, len(columns) - 1) or len(columns) > 8:
        return None
    if len(columns) < 2 and len(data) < 4:
        return None

    grid: list[list[str]] = []
    header_line = lines[first - 1] if first > 0 else None
    header = _header_cells(header_line, columns) if header_line is not None else None
    top = lines[run[0]].top
    if header is not None and header_line is not None:
        grid.append([""] + header)
        top = min(top, header_line.top)

    for k in run:
        label_cells = parsed[k]
        if label_cells is None:   # label-only section line
            grid.append([" ".join(w["text"] for w in lines[k].words)] + [""] * len(columns))
            continue
        label, cells = label_cells
        row = [label] + [""] * len(columns)
        for cell in cells:
            row[1 + _nearest(columns, cell["x1"])] = cell["text"]
        grid.append(row)

    x0 = min(w["x0"] for k in run for w in lines[k].words)
    x1 = max(w["x1"] for k in run for w in lines[k].words)
    bottom = lines[run[-1]].bottom
    return (x0, top, x1, bottom), grid
