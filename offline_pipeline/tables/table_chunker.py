"""Chunk tables without ever cutting through a row.

Each chunk is self-describing: it repeats the caption, the units line and the
column headers, so a retrieved fragment of a 90-row table still says what its
numbers mean. Two representations are kept:

* the *embedded text* is linearised (``Revenue: FY2024 = 391,035; FY2023 = 365,817``)
  because that embeds and matches queries far better than pipe-separated cells;
* the payload keeps the original table (Markdown) for display, plus the parsed
  numeric cells (``numerics``) so answers can be verified against data.
"""

from __future__ import annotations

import hashlib
from typing import Any

from offline_pipeline.tables.blocks import Table
from offline_pipeline.tables.numbers import parse_cell_number, to_base_units, units_label

MAX_NUMERICS_PER_CHUNK = 400


def estimate_tokens(text: str) -> int:
    """Token estimate that errs high for number-heavy text (digits tokenise poorly)."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
    except Exception:  # noqa: BLE001 - tiktoken missing or offline: fall back
        return max(1, len(text) // 3)


def table_id(table: Table, document_key: str) -> str:
    digest = hashlib.sha256()
    digest.update(document_key.encode())
    digest.update(table.caption.encode())
    for row in table.header_rows + table.rows[:3]:
        digest.update("|".join(row).encode())
    return digest.hexdigest()[:16]


def _split_label(row: list[str], label_columns: int) -> tuple[str, list[tuple[int, str]]]:
    label = " ".join(c for c in row[:label_columns] if c).strip() or "(row)"
    cells = [(i, c) for i, c in enumerate(row) if i >= label_columns and c]
    return label, cells


def _markdown(headers: list[str], rows: list[list[str]], columns: list[int]) -> str:
    def cell(row: list[str], i: int) -> str:
        return (row[i] if i < len(row) else "").replace("|", "\\|")

    head = "| " + " | ".join(h.replace("|", "\\|") for h in (headers[i] for i in columns)) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(r, i) for i in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body])


class TableChunker:
    def __init__(self, chunk_size: int = 512) -> None:
        self.chunk_size = chunk_size

    def chunk(
        self,
        table: Table,
        base_metadata: dict[str, Any],
        *,
        document_key: str = "",
        label_columns: int = 1,
    ) -> list[dict[str, Any]]:
        """Return chunk dicts (``text`` + ``metadata``); indexes are assigned by the caller."""
        headers = table.column_headers()
        label_columns = max(1, min(label_columns, max(1, table.column_count)))
        tid = table_id(table, document_key)
        units = table.units or units_label(table.scale, table.currency)

        def preamble(columns: list[int]) -> str:
            lines = [f"Table: {table.caption}" if table.caption else "Table"]
            if units:
                lines.append(f"Units: {units}")
            lines.append("Columns: " + " | ".join(headers[i] for i in columns))
            return "\n".join(lines)

        all_columns = list(range(table.column_count))
        budget = max(64, self.chunk_size - estimate_tokens(preamble(all_columns)) - 8)

        chunks: list[dict[str, Any]] = []
        current: list[tuple[int, list[int], str]] = []   # (row index, columns, line)
        used = 0

        def flush() -> None:
            nonlocal current, used
            if not current:
                return
            columns = sorted({c for _, cols, _ in current for c in cols} | set(range(label_columns)))
            row_ids = sorted({r for r, _, _ in current})
            rows = [table.rows[r] for r in row_ids]
            text = preamble(columns) + "\n" + "\n".join(line for _, _, line in current)
            if table.footnotes:
                text += "\nNotes: " + " ".join(table.footnotes)
            chunks.append({
                "text": text,
                "metadata": {
                    **base_metadata,
                    "block_kind": "table",
                    "table_id": tid,
                    "caption": table.caption,
                    "units": units,
                    "scale": table.scale,
                    "currency": table.currency,
                    "row_start": row_ids[0],
                    "row_end": row_ids[-1],
                    "table_confidence": table.confidence,
                    "table_markdown": _markdown(headers, rows, columns),
                    "numerics": self._numerics(table, headers, row_ids, columns, label_columns),
                    **({"page": table.page} if table.page is not None else {}),
                },
            })
            current, used = [], 0

        for index, row in enumerate(table.rows):
            label, cells = _split_label(row, label_columns)
            if not cells:  # a section sub-heading row such as "Assets"
                line = label
                tokens = estimate_tokens(line)
                if used + tokens > budget:
                    flush()
                current.append((index, list(range(label_columns)), line))
                used += tokens
                continue

            groups = self._column_groups(label, cells, headers, budget)
            for group in groups:
                line = f"{label}: " + "; ".join(f"{headers[i]} = {c}" for i, c in group)
                tokens = estimate_tokens(line)
                if used + tokens > budget:
                    flush()
                current.append((index, [i for i, _ in group], line))
                used += tokens
        flush()
        return chunks

    @staticmethod
    def _column_groups(
        label: str, cells: list[tuple[int, str]], headers: list[str], budget: int
    ) -> list[list[tuple[int, str]]]:
        """Split one over-wide row by columns; the row label is repeated on each part."""
        whole = f"{label}: " + "; ".join(f"{headers[i]} = {c}" for i, c in cells)
        if estimate_tokens(whole) <= budget:
            return [cells]
        groups: list[list[tuple[int, str]]] = []
        group: list[tuple[int, str]] = []
        for pair in cells:
            trial = group + [pair]
            text = f"{label}: " + "; ".join(f"{headers[i]} = {c}" for i, c in trial)
            if group and estimate_tokens(text) > budget:
                groups.append(group)
                group = [pair]
            else:
                group = trial
        if group:
            groups.append(group)
        return groups

    @staticmethod
    def _numerics(
        table: Table, headers: list[str], row_ids: list[int], columns: list[int], label_columns: int
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in row_ids:
            row = table.rows[r]
            label = " ".join(c for c in row[:label_columns] if c).strip()
            for i in columns:
                if i < label_columns or i >= len(row):
                    continue
                parsed = parse_cell_number(row[i])
                if parsed is None or not parsed.is_value:
                    continue
                base = to_base_units(parsed, table.scale)
                out.append({
                    "row": label,
                    "column": headers[i],
                    "text": row[i],
                    "value": format(base, "f"),
                    "kind": parsed.kind,
                    "currency": parsed.currency or table.currency,
                    "scale": 0 if parsed.kind == "percent" else table.scale,
                })
                if len(out) >= MAX_NUMERICS_PER_CHUNK:
                    return out
        return out
