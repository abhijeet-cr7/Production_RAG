"""Chunk structured blocks: prose stays prose, tables stay whole, facts stay exact."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from offline_pipeline.chunkers.recursive_chunker import RecursiveTextChunker
from offline_pipeline.tables.blocks import FACTS, HEADING, PARAGRAPH, TABLE, Block, Fact
from offline_pipeline.tables.table_chunker import TableChunker, estimate_tokens

MAX_NUMERICS_PER_CHUNK = 400


class BlockChunker:
    """Walk blocks in order, keeping a heading path and flushing prose at boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self._prose = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._tables = TableChunker(chunk_size=chunk_size)

    def chunk(self, blocks: list[Block], metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        base = dict(metadata or {})
        document_key = str(base.get("doc_id") or base.get("source") or "")
        path: list[tuple[int, str]] = []
        prose: list[str] = []
        prose_page: int | None = None
        out: list[dict[str, Any]] = []

        def section() -> str:
            return " > ".join(title for _, title in path)

        def flush_prose() -> None:
            nonlocal prose, prose_page
            if not prose:
                return
            text = "\n\n".join(prose)
            heading = section()
            meta = {**base, "block_kind": "prose", **({"section_path": heading} if heading else {}),
                    **({"page": prose_page} if prose_page is not None else {})}
            for piece in self._prose.chunk(text, meta):
                if heading:
                    piece["text"] = f"[Section: {heading}]\n{piece['text']}"
                out.append(piece)
            prose, prose_page = [], None

        for block in blocks:
            if block.kind == HEADING:
                flush_prose()
                while path and path[-1][0] >= max(1, block.level):
                    path.pop()
                path.append((max(1, block.level), block.text))
            elif block.kind == PARAGRAPH:
                if prose_page is None:
                    prose_page = block.page
                prose.append(block.text)
            elif block.kind == TABLE and block.table is not None:
                flush_prose()
                meta = {**base, **({"section_path": section()} if path else {})}
                out.extend(self._tables.chunk(block.table, meta, document_key=document_key))
            elif block.kind == FACTS and block.facts:
                flush_prose()
                meta = {**base, **({"section_path": section()} if path else {})}
                out.extend(self._fact_chunks(block.facts, meta))
        flush_prose()

        for index, chunk in enumerate(out):
            chunk["metadata"]["chunk_index"] = index
            chunk["metadata"]["token_count"] = estimate_tokens(chunk["text"])
            chunk["metadata"]["chunking_strategy"] = "table_aware"
        return out

    def _fact_chunks(self, facts: list[Fact], meta: dict[str, Any]) -> list[dict[str, Any]]:
        statement = facts[0].statement
        period = facts[0].period_end or facts[0].period
        entity = next((f.entity for f in facts if f.entity), "")
        segmented = bool(facts[0].dimensions)
        title = f"XBRL facts: {statement} statement" + (" (segment detail)" if segmented else "")
        header = "\n".join(filter(None, [title, f"Entity: {entity}" if entity else "", f"Period: {period}" if period else ""]))
        budget = max(64, self.chunk_size - estimate_tokens(header) - 8)

        chunks: list[dict[str, Any]] = []
        lines: list[str] = []
        numerics: list[dict[str, Any]] = []
        used = 0

        def flush() -> None:
            nonlocal lines, numerics, used
            if not lines:
                return
            chunks.append({
                "text": header + "\n" + "\n".join(lines),
                "metadata": {**meta, "block_kind": "xbrl_facts", "statement": statement,
                             "period": period, "numerics": numerics[:MAX_NUMERICS_PER_CHUNK]},
            })
            lines, numerics, used = [], [], 0

        for fact in facts:
            dims = f" [{'; '.join(fact.dimensions)}]" if fact.dimensions else ""
            line = f"{fact.label} ({fact.concept}){dims} | {fact.period} | {fact.unit} | {fact.display}"
            tokens = estimate_tokens(line)
            if used + tokens > budget:
                flush()
            lines.append(line)
            used += tokens
            if fact.value is not None:
                numerics.append({
                    "concept": fact.concept, "row": fact.label, "column": fact.period_end or fact.period,
                    "text": fact.display, "value": fact.value, "unit": fact.unit,
                    "period": fact.period, "period_end": fact.period_end,
                    "kind": "percent" if fact.unit.lower() == "pure" else "number",
                    "currency": fact.unit if len(fact.unit) == 3 and fact.unit.isalpha() else "",
                    "scale": 0, "decimals": fact.decimals,
                })
        flush()
        return chunks
