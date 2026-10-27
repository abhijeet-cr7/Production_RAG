"""PDF guard: reject active content and bound everything, then extract visible text.

A PDF is a programmable container. Besides parser bugs (handled by running all of
this inside the sandbox subprocess) it can carry JavaScript, auto-launch actions,
embedded files, form/XFA scripts and decompression bombs. This inspects the real
object graph (not a byte grep, which obfuscated names defeat) and refuses the
file when it holds anything executable or attached. Extraction then drops text a
human could not see: tiny, off-page or white-on-white characters.
"""

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Any

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.tables.blocks import (
    PARAGRAPH,
    TABLE,
    Block,
    Table,
    blocks_to_json,
    blocks_to_text,
    infer_header_rows,
    trim_grid,
)
from offline_pipeline.tables.numbers import detect_units
from offline_pipeline.tables.pdf_tables import detect_numeric_tables

logger = logging.getLogger(__name__)

MAX_OBJECTS = 300_000

# Decoded-size ceilings, enforced with a *bounded* decompressor so a bomb is
# refused in milliseconds instead of being expanded by the text extractor first.
MAX_DECODED_STREAM = 32 * 1024 * 1024        # content streams, fonts, ...
MAX_DECODED_IMAGE = 128 * 1024 * 1024        # raster images (scans can be large)
MAX_DECODED_TOTAL = 512 * 1024 * 1024        # across the whole document
MAX_IMAGE_PIXELS = 100_000_000

# Keys whose mere presence means executable or attached content.
_DANGEROUS_KEYS = {
    "/JS", "/JavaScript", "/Launch", "/EmbeddedFile", "/EmbeddedFiles", "/RichMedia",
    "/XFA", "/SubmitForm", "/ImportData", "/Rendition", "/Movie", "/Sound", "/3D",
    "/FileAttachment", "/GoToE",
}
# Action types (/S) that are harmless navigation.
_SAFE_ACTIONS = {"/GoTo", "/URI", "/Named"}


def _limit_decompression() -> None:
    try:
        from pypdf import filters

        for name, value in {
            "ZLIB_MAX_OUTPUT_LENGTH": 20 * 1024 * 1024,
            "MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH": 20 * 1024 * 1024,
            "MAX_DECLARED_STREAM_LENGTH": 20 * 1024 * 1024,
        }.items():
            if hasattr(filters, name):
                setattr(filters, name, value)
    except Exception:  # noqa: BLE001
        logger.debug("Could not tighten pypdf stream limits.")
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = 50_000_000
        warnings.simplefilter("error", Image.DecompressionBombWarning)
    except Exception:  # noqa: BLE001
        pass


def inspect_pdf(path: Path, *, max_pages: int) -> Any:
    """Return an open pypdf reader, or raise :class:`UnsafeContent`."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    _limit_decompression()
    try:
        with open(path, "rb") as handle:
            if b"%PDF-" not in handle.read(1024):
                raise UnsafeContent("pdf_not_pdf", "File is not a PDF.")
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            raise UnsafeContent("pdf_encrypted", "Encrypted PDFs are not accepted.")
        page_count = len(reader.pages)
        if page_count > max_pages:
            raise UnsafeContent("pdf_too_many_pages", f"PDF has more than {max_pages} pages.")
        _scan_object_graph(reader)
    except UnsafeContent:
        raise
    except (PyPdfError, RecursionError, ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        raise UnsafeContent("pdf_malformed", "PDF could not be parsed safely.") from exc
    return reader


def _scan_object_graph(reader: Any) -> None:
    if reader.trailer.get("/Root") is None:
        raise UnsafeContent("pdf_malformed", "PDF has no catalog.")

    # Plain objects, plus those packed inside object streams (a favourite place
    # to hide payloads, because they are invisible to a grep for the key names).
    numbers = {(gen, num) for gen, objects in reader.xref.items() for num in objects}
    numbers |= {(0, num) for num in getattr(reader, "xref_objStm", {})}
    if len(numbers) > MAX_OBJECTS:
        raise UnsafeContent("pdf_too_complex", "PDF has too many objects.")

    budget = _DecodeBudget()
    for generation, number in numbers:
        try:
            obj = reader.get_object(_indirect(reader, number, generation))
        except Exception:  # noqa: BLE001 - an unreadable object is skipped, not trusted
            continue
        _check_object(obj)
        _check_stream_size(obj, budget)


class _DecodeBudget:
    def __init__(self) -> None:
        self.total = 0


def _check_stream_size(obj: Any, budget: _DecodeBudget) -> None:
    """Refuse Flate streams (and images) that expand beyond the ceilings."""
    import zlib

    from pypdf.generic import ArrayObject, DictionaryObject, StreamObject

    if not isinstance(obj, StreamObject):
        return
    is_image = str(obj.get("/Subtype", "")) == "/Image"
    if is_image:
        width, height = obj.get("/Width", 0), obj.get("/Height", 0)
        if isinstance(width, int) and isinstance(height, int) and width * height > MAX_IMAGE_PIXELS:
            raise UnsafeContent("pdf_bomb", "PDF contains an oversized image.")

    filters = obj.get("/Filter")
    names = [str(f) for f in filters] if isinstance(filters, ArrayObject) else [str(filters)] if filters else []
    if not names or names[0] not in ("/FlateDecode", "/Fl"):
        return
    raw = getattr(obj, "_data", None)
    if not isinstance(raw, (bytes, bytearray)):
        return
    limit = MAX_DECODED_IMAGE if is_image else MAX_DECODED_STREAM
    try:
        decoder = zlib.decompressobj()
        produced = len(decoder.decompress(bytes(raw), limit))
        if decoder.unconsumed_tail:  # more output was available than the ceiling allows
            raise UnsafeContent("pdf_bomb", "PDF contains a stream that expands too far.")
    except zlib.error:
        return  # corrupt stream: the extractor will reject or skip it
    budget.total += produced
    if budget.total > MAX_DECODED_TOTAL:
        raise UnsafeContent("pdf_bomb", "PDF expands to too much data.")


def _indirect(reader: Any, number: int, generation: int) -> Any:
    from pypdf.generic import IndirectObject

    return IndirectObject(number, generation, reader)


_DANGEROUS_SUBTYPES = {"/RichMedia", "/Movie", "/Sound", "/3D", "/Screen", "/FileAttachment"}
MAX_NESTING = 50


def _check_object(obj: Any, depth: int = 0) -> None:
    """Reject executable/attached content anywhere in *obj*, including nested
    direct dictionaries and arrays (indirect references are visited on their own)."""
    from pypdf.generic import ArrayObject, DictionaryObject

    if depth > MAX_NESTING:
        raise UnsafeContent("pdf_too_complex", "PDF objects are nested too deeply.")

    if isinstance(obj, ArrayObject):
        for item in list.__iter__(obj):  # raw items: do not resolve references
            _check_object(item, depth + 1)
        return
    if not isinstance(obj, DictionaryObject):
        return

    raw = dict(dict.items(obj))  # raw values: references stay unresolved
    for key in raw:
        if str(key) in _DANGEROUS_KEYS:
            raise UnsafeContent("pdf_active_content", "PDF contains scripts, embedded files or media.")
    if "/AA" in raw:
        raise UnsafeContent("pdf_active_content", "PDF contains automatic actions.")
    for action_key in ("/OpenAction", "/A"):
        if action_key in raw:
            _check_action(raw[action_key])
    if str(raw.get("/Subtype", "")) in _DANGEROUS_SUBTYPES or str(raw.get("/Type", "")) == "/EmbeddedFile":
        raise UnsafeContent("pdf_active_content", "PDF contains media, attachments or embedded files.")
    for value in raw.values():
        if isinstance(value, (DictionaryObject, ArrayObject)):
            _check_object(value, depth + 1)


def _check_action(raw: Any) -> None:
    from pypdf.generic import ArrayObject, DictionaryObject

    action = raw.get_object() if hasattr(raw, "get_object") else raw
    if isinstance(action, ArrayObject):  # a plain destination, e.g. [page /Fit]
        return
    if isinstance(action, DictionaryObject):
        kind = str(action.get("/S", ""))
        if kind and kind not in _SAFE_ACTIONS:
            raise UnsafeContent("pdf_active_content", f"PDF contains a '{kind}' action.")
        nxt = action.get("/Next")
        if nxt is not None:
            _check_action(nxt)


# ── visible-text extraction ──────────────────────────────────────────────────

def _is_white(color: Any) -> bool:
    if color is None:
        return False
    values = color if isinstance(color, (list, tuple)) else (color,)
    if not values or not all(isinstance(v, (int, float)) for v in values):
        return False
    if len(values) == 4:  # CMYK: no ink at all
        return all(v <= 0.05 for v in values)
    return all(v >= 0.95 for v in values)


def _covered_by_dark_rect(char: dict[str, Any], rects: list[dict[str, Any]]) -> bool:
    for rect in rects:
        if (rect["x0"] - 1 <= char["x0"] and char["x1"] <= rect["x1"] + 1
                and rect["top"] - 1 <= char["top"] and char["bottom"] <= rect["bottom"] + 1):
            return True
    return False


def extract_pdf(
    path: Path, *, max_pages: int, max_chars: int, ocr_max_pages: int
) -> dict[str, Any]:
    """Inspect then extract; returns ``{"text", "metadata", "blocks"}`` of *visible* content.

    Tables are returned as table blocks (never flattened into prose) and the page
    text between them as paragraph blocks, in reading order.
    """
    reader = inspect_pdf(path, max_pages=max_pages)
    page_count = len(reader.pages)

    blocks: list[Block] = []
    hidden_removed = 0
    ocr_pages = 0
    tables_found = 0
    total = 0
    try:
        import pdfplumber
    except ImportError:  # fall back to pypdf text, without hidden-text filtering
        pdfplumber = None  # type: ignore[assignment]
        logger.warning("pdfplumber not installed; hidden-text filtering and tables are off.")

    plumber = pdfplumber.open(str(path)) if pdfplumber is not None else None
    try:
        for index in range(page_count):
            page_blocks, removed = _page_blocks(reader, plumber, index)
            hidden_removed += removed
            if not any(b.kind == PARAGRAPH and b.text.strip() or b.kind == TABLE for b in page_blocks) \
                    and ocr_pages < ocr_max_pages:
                ocr_text = _ocr_page(reader.pages[index])
                if ocr_text.strip():
                    ocr_pages += 1
                    page_blocks = [Block(kind=PARAGRAPH, text=ocr_text, page=index + 1)]
            tables_found += sum(1 for b in page_blocks if b.kind == TABLE)
            total += sum(len(b.text) for b in page_blocks) + blocks_text_len(page_blocks)
            if total > max_chars:
                raise UnsafeContent("text_too_large", "Extracted text is too large.")
            blocks.extend(page_blocks)
    finally:
        if plumber is not None:
            plumber.close()

    return {
        "text": blocks_to_text(blocks),
        "metadata": {
            "file_type": "pdf",
            "page_count": page_count,
            "hidden_chars_removed": hidden_removed,
            "ocr_pages": ocr_pages,
            "tables_found": tables_found,
        },
        "blocks": blocks_to_json(blocks),
    }


def blocks_text_len(blocks: list[Block]) -> int:
    return sum(len(c) for b in blocks if b.table for row in b.table.header_rows + b.table.rows for c in row)


def _visible_filter(page: Any, counter: list[int]) -> Any:
    rects = [
        r for r in (page.rects or [])[:2000]
        if r.get("fill") and not _is_white(r.get("non_stroking_color"))
    ]
    x0, top, x1, bottom = page.bbox

    def visible(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        tiny = obj.get("size", 10) < 2
        off_page = obj["x1"] < x0 - 1 or obj["x0"] > x1 + 1 or obj["bottom"] < top - 1 or obj["top"] > bottom + 1
        white = _is_white(obj.get("non_stroking_color")) and not _covered_by_dark_rect(obj, rects)
        if tiny or off_page or white:
            counter[0] += 1
            return False
        return True

    return visible


def _numeric_cells(row: list[str]) -> int:
    from offline_pipeline.tables.numbers import parse_cell_number

    return sum(1 for cell in row[1:] if cell and parse_cell_number(cell) is not None)


def _find_tables(page: Any) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    """(bbox, grid) for each table: ruled tables, plus borderless numeric ones elsewhere on the page."""
    found = []
    for table in page.find_tables():
        grid = trim_grid(table.extract())
        if len(grid) >= 2 and len(grid[0]) >= 2:
            found.append((table.bbox, grid))

    words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False) or []

    def inside_ruled(word: dict[str, Any]) -> bool:
        return any(x0 - 1 <= word["x0"] and word["x1"] <= x1 + 1 and top - 1 <= word["top"] and word["bottom"] <= bottom + 1
                   for (x0, top, x1, bottom), _ in found)

    loose = [w for w in words if not inside_ruled(w)]
    if sum(1 for w in loose if any(ch.isdigit() for ch in w["text"])) >= 9:
        found.extend((bbox, trim_grid(grid)) for bbox, grid in detect_numeric_tables(loose))
    return found


def _grid_to_table(grid: list[list[str]], caption: str, page_no: int, low_confidence: bool) -> Table:
    header_rows, body = infer_header_rows(grid)
    scale, currency = detect_units(caption, *(" ".join(r) for r in header_rows[:2]))
    return Table(
        caption=caption, header_rows=header_rows, rows=body, scale=scale, currency=currency,
        page=page_no, confidence="low" if low_confidence else "high",
    )


def _page_blocks(reader: Any, plumber: Any, index: int) -> tuple[list[Block], int]:
    page_no = index + 1
    if plumber is None:
        text = reader.pages[index].extract_text() or ""
        return ([Block(kind=PARAGRAPH, text=text, page=page_no)] if text.strip() else []), 0
    try:
        page = plumber.pages[index]
        counter = [0]
        visible_page = page.filter(_visible_filter(page, counter))
        tables = sorted(_find_tables(visible_page), key=lambda t: t[0][1])

        px0, ptop, px1, pbottom = page.bbox
        blocks: list[Block] = []
        cursor = ptop
        for bbox, grid in tables:
            x0, top, x1, bottom = bbox
            top, bottom = max(ptop, top), min(pbottom, bottom)
            caption = ""
            if top - cursor > 2:
                above = _crop_text(visible_page, (px0, cursor, px1, top))
                lines = [ln.strip() for ln in above.splitlines() if ln.strip()]
                if lines and len(lines[-1]) < 160 and len(lines[-1].split()) >= 2:
                    caption = lines[-1]
                text = "\n".join(lines[:-1] if caption else lines)
                if text.strip():
                    blocks.append(Block(kind=PARAGRAPH, text=text, page=page_no))
            blocks.append(Block(kind=TABLE, table=_grid_to_table(grid, caption, page_no, False), page=page_no))
            cursor = max(cursor, bottom)
        if pbottom - cursor > 2 or not tables:
            tail = _crop_text(visible_page, (px0, cursor, px1, pbottom)) if tables else (visible_page.extract_text() or "")
            if tail.strip():
                blocks.append(Block(kind=PARAGRAPH, text=tail, page=page_no))
        return blocks, counter[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber failed on page %d (%s); using pypdf.", page_no, type(exc).__name__)
        text = reader.pages[index].extract_text() or ""
        return ([Block(kind=PARAGRAPH, text=text, page=page_no)] if text.strip() else []), 0


def _crop_text(page: Any, bbox: tuple[float, float, float, float]) -> str:
    x0, top, x1, bottom = bbox
    if bottom - top <= 0:
        return ""
    # within_bbox keeps only characters fully inside the band. crop() would also
    # keep any character merely touching the edge, pulling a table's header line
    # into the text above it and its last row into the text below.
    return page.within_bbox((x0, top, x1, bottom)).extract_text() or ""


def _page_text(reader: Any, plumber: Any, index: int) -> tuple[str, int]:
    if plumber is not None:
        try:
            page = plumber.pages[index]
            rects = [
                r for r in (page.rects or [])[:2000]
                if r.get("fill") and not _is_white(r.get("non_stroking_color"))
            ]
            x0, top, x1, bottom = page.bbox
            removed = 0

            def visible(obj: dict[str, Any]) -> bool:
                nonlocal removed
                if obj.get("object_type") != "char":
                    return True
                tiny = obj.get("size", 10) < 2
                off_page = obj["x1"] < x0 - 1 or obj["x0"] > x1 + 1 or obj["bottom"] < top - 1 or obj["top"] > bottom + 1
                white = _is_white(obj.get("non_stroking_color")) and not _covered_by_dark_rect(obj, rects)
                if tiny or off_page or white:
                    removed += 1
                    return False
                return True

            text = page.filter(visible).extract_text() or ""
            return text, removed
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber failed on page %d (%s); using pypdf.", index, type(exc).__name__)
    return reader.pages[index].extract_text() or "", 0


def _ocr_page(page: Any) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        for image_file in page.images:
            return pytesseract.image_to_string(Image.open(io.BytesIO(image_file.data)))
    except Exception as exc:  # noqa: BLE001 - includes DecompressionBombWarning
        logger.warning("OCR failed: %s", type(exc).__name__)
    return ""
