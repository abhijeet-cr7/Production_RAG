"""Parse the numbers financial documents actually contain.

Cells and prose write the same quantity many ways: ``1,234.5``, ``(1,234)``,
``-1,234``, ``$ 1,234``, ``12.5%``, ``€1.234,5``, ``—``, ``391.0 billion``. The
table chunker stores the parsed values so answers can be checked against data
instead of prose; the answer verifier reuses the same parser.

Values are ``Decimal`` so nothing is lost to float rounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_MINUS = "-−–‒"          # hyphen, minus sign, en dash, figure dash
_SUPERSCRIPTS = "¹²³⁰⁴⁵⁶⁷⁸⁹"
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

NIL_MARKERS = {"-", "—", "–", "‒", "−", "―", "nil", "--"}
NOT_A_NUMBER = {"n/a", "na", "nm", "n/m", "*", "**", "tbd", "——"}

SCALE_WORDS = {
    "thousand": 3, "thousands": 3, "k": 3,
    "million": 6, "millions": 6, "mm": 6, "mn": 6, "m": 6,
    "billion": 9, "billions": 9, "bn": 9, "b": 9,
    "trillion": 12, "trillions": 12, "tn": 12, "t": 12,
}


@dataclass(frozen=True)
class ParsedNumber:
    value: Decimal | None        # None for nil dashes and unparseable text
    text: str                    # the cell text as written
    kind: str                    # "number" | "percent" | "currency" | "nil"
    currency: str = ""           # ISO code when a symbol/code was present
    decimals: int = 0            # digits after the decimal point as written

    @property
    def is_value(self) -> bool:
        return self.value is not None


_NUMBER = re.compile(
    r"""^\(?\s*
        (?P<neg1>[%s])?\s*
        (?P<cur>[$€£¥]|USD|EUR|GBP|JPY|US\$)?\s*
        (?P<neg2>[%s])?\s*
        (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\d{1,3}(?:\.\d{3})+,\d+|\d+,\d+|\.\d+)
        \s*(?P<pct>%%)?\s*\)?\s*(?P<pct2>%%)?$
    """ % (re.escape(_MINUS), re.escape(_MINUS)),
    re.X | re.I,
)


def _normalise_digits(raw: str) -> tuple[str, int]:
    """Return (digits with '.' decimal point, number of decimals)."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", raw):       # 1.234,5  (European)
        raw = raw.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d+,\d{1,2}", raw):                  # 12,5     (decimal comma)
        raw = raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return raw, decimals


def parse_cell_number(text: str) -> ParsedNumber | None:
    """Parse a table cell. Returns None when the cell is not numeric at all."""
    original = text
    cell = text.strip()
    if not cell:
        return None
    lowered = cell.lower()
    if lowered in NOT_A_NUMBER:
        return None
    if cell in NIL_MARKERS or lowered in NIL_MARKERS:
        return ParsedNumber(None, original, "nil")

    # Footnote markers: "1,234(1)", "1,234¹", "1,234*"
    cell = re.sub(r"\s*\(\d{1,2}\)\s*$", "", cell) if re.search(r"\d\s*\(\d{1,2}\)\s*$", cell) and not cell.startswith("(") else cell
    cell = cell.rstrip(_SUPERSCRIPTS + "*†‡").strip()

    match = _NUMBER.match(cell)
    if not match:
        return None
    digits, decimals = _normalise_digits(match.group("num"))
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None

    # Accounting negatives: "(1,234)" and "(5.2)%" (the % may follow the bracket).
    bracketed = bool(re.match(r"^\(.*\)\s*%?$", cell))
    negative = bool(match.group("neg1") or match.group("neg2")) or bracketed
    if negative:
        value = -value
    symbol = match.group("cur") or ""
    currency = _CURRENCY_SYMBOLS.get(symbol, symbol.upper().replace("US$", "USD"))
    kind = "percent" if (match.group("pct") or match.group("pct2")) else "currency" if currency else "number"
    return ParsedNumber(value, original, kind, currency, decimals)


# ── units and scale ──────────────────────────────────────────────────────────

_CUR = r"(?:[$\u20ac\u00a3\u00a5]|us\$|usd|eur|gbp|jpy|chf|cad|aud)"


def _scale_pattern(words: str, extra: str = "") -> re.Pattern[str]:
    # A scale needs context: "in millions", "$ millions", "(millions)". A bare
    # "millions" ("millions of customers") must not rescale a whole table.
    return re.compile(
        rf"\bin\s+(?:{_CUR}\s*)?(?:{words})\b"
        rf"|{_CUR}\s*(?:in\s+)?(?:{words})\b"
        rf"|\(\s*(?:{words})\s*\)" + extra,
        re.I,
    )


_SCALE_PHRASES = [
    (_scale_pattern(r"thousands?", r"|\(\s*000'?s\s*\)|[$\u20ac\u00a3]\s*000'?s|\bin\s+[$\u20ac\u00a3]?000'?s?\b"), 3),
    (_scale_pattern(r"millions?"), 6),
    (_scale_pattern(r"billions?"), 9),
]
_CURRENCY_HINT = re.compile(r"\b(USD|EUR|GBP|JPY|CHF|CAD|AUD)\b|(\$|€|£|¥)", re.I)


def detect_units(*texts: str) -> tuple[int, str]:
    """Return (power-of-ten scale, ISO currency or '') hinted by captions/headers.

    ``(In thousands, except per share data)`` → (3, "").
    """
    joined = " ".join(t for t in texts if t)
    scale = 0
    for pattern, power in _SCALE_PHRASES:
        if pattern.search(joined):
            scale = power
            break
    currency = ""
    hint = _CURRENCY_HINT.search(joined)
    if hint:
        currency = hint.group(1).upper() if hint.group(1) else _CURRENCY_SYMBOLS.get(hint.group(2), "")
    return scale, currency


def units_label(scale: int, currency: str) -> str:
    words = {0: "", 3: "thousands", 6: "millions", 9: "billions", 12: "trillions"}.get(scale, f"10^{scale}")
    return " ".join(p for p in (currency, words) if p)


def to_base_units(parsed: ParsedNumber, scale: int) -> Decimal | None:
    """Apply the table's scale to a parsed cell (percentages are never scaled)."""
    if parsed.value is None:
        return None
    if parsed.kind == "percent" or scale == 0:
        return parsed.value
    return parsed.value * (Decimal(10) ** scale)


# ── numbers inside running prose (used by the answer verifier) ───────────────

_PROSE = re.compile(
    r"""(?P<neg>[%s])?\s*
        (?P<cur>[$€£¥]|USD|EUR|GBP|US\$)?\s*
        (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
        (?:\s*(?P<pct>%%|percent\b|per\s*cent\b))?
        (?:\s*(?P<scale>trillion|billion|million|thousand|bn|mn|mm|tn|k|[BMT])\b)?
    """ % re.escape(_MINUS),
    re.X | re.I,
)


@dataclass(frozen=True)
class ProseNumber:
    text: str
    value: Decimal                 # base units (scale applied; percent stays as written)
    quantum: Decimal               # half-width of what the stated precision implies
    kind: str                      # "number" | "percent" | "currency"
    currency: str
    start: int
    end: int
    negative: bool


def extract_prose_numbers(text: str) -> list[ProseNumber]:
    out: list[ProseNumber] = []
    for m in _PROSE.finditer(text):
        raw = m.group("num")
        # A trailing letter that is really part of a word ("5 Billion" is fine, "5 Mark" is not)
        has_unit = bool(m.group("pct") or m.group("scale") or m.group("cur"))
        # Look at the characters around the digits themselves: the match also
        # swallows leading whitespace and any currency symbol.
        following = text[m.end("num"): m.end("num") + 1]
        preceding = text[m.start("num") - 1: m.start("num")] if m.start("num") else ""
        # "3rd", "Q3", "A4": digits glued to letters are labels, not quantities.
        if not has_unit and (following.isalpha() or preceding.isalpha()):
            continue
        digits, decimals = _normalise_digits(raw)
        try:
            value = Decimal(digits)
        except InvalidOperation:
            continue
        scale_word = (m.group("scale") or "").lower()
        power = SCALE_WORDS.get(scale_word, 0) if scale_word else 0
        negative = bool(m.group("neg"))
        symbol = m.group("cur") or ""
        currency = _CURRENCY_SYMBOLS.get(symbol, symbol.upper().replace("US$", "USD"))
        kind = "percent" if m.group("pct") else "currency" if currency else "number"
        unit = Decimal(10) ** power
        step = Decimal(1).scaleb(-decimals) * unit  # one unit in the last stated place
        out.append(ProseNumber(
            text=m.group(0).strip(), value=(-value if negative else value) * unit,
            quantum=step / 2, kind=kind, currency=currency,
            start=m.start(), end=m.end(), negative=negative,
        ))
    return out
