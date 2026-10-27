"""XBRL facts from inline XBRL (iXBRL) filings and XBRL instance documents.

A 10-K's HTML carries machine-readable facts: each tagged number has a concept
(``us-gaap:Revenues``), a context (entity + period + dimensions), a unit, a scale
and a sign. Parsing them gives *exact* values with their period and unit, which
is what answer verification needs, rather than text scraped from a table.

Safety: parsed with the hardened XML parser (no DTD/entities/network). Taxonomy
files are never fetched: labels come from the concept name. Hidden (``ix:hidden``)
facts are kept only when numeric or on a short whitelist of ``dei:`` concepts, so
that section cannot smuggle free text into the index.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any
from xml.etree.ElementTree import Element

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.safety.xml_guard import safe_fromstring
from offline_pipeline.tables.blocks import FACTS, Block, Fact

MAX_FACTS = 100_000

IX_NAMESPACES = (
    "http://www.xbrl.org/2013/inlineXBRL",
    "http://www.xbrl.org/2008/inlineXBRL",
)
XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"
KNOWN_PREFIXES = {
    "http://fasb.org/us-gaap": "us-gaap",
    "http://xbrl.sec.gov/dei": "dei",
    "http://xbrl.ifrs.org/taxonomy": "ifrs-full",
    "http://xbrl.sec.gov/ecd": "ecd",
}
_DEI_TEXT_WHITELIST = {
    "DocumentType": "form_type",
    "EntityRegistrantName": "entity",
    "DocumentPeriodEndDate": "period_end",
    "DocumentFiscalYearFocus": "fiscal_year",
    "DocumentFiscalPeriodFocus": "fiscal_period_focus",
    "EntityCentralIndexKey": "cik",
}

_STATEMENT_PATTERNS = [
    ("cashflow", re.compile(r"NetCashProvided|NetCashUsed|PaymentsTo|PaymentsFor|ProceedsFrom|IncreaseDecreaseIn|Depreciation|AmortizationOf|ShareBasedCompensation|CashAndCashEquivalentsPeriodIncrease|RepaymentsOf", re.I)),
    ("income", re.compile(r"^(Revenue|Sales|CostOf|GrossProfit|OperatingIncome|OperatingExpenses|NetIncome|EarningsPerShare|IncomeTax|InterestExpense|InterestIncome|ResearchAndDevelopment|SellingGeneral|IncomeLoss|ProfitLoss|CostsAndExpenses|WeightedAverage|Dividends.*PerShare|OtherNonoperating)", re.I)),
    ("balance", re.compile(r"^(Assets|Liabilities|StockholdersEquity|Cash|Inventory|AccountsReceivable|AccountsPayable|PropertyPlant|Goodwill|IntangibleAssets|RetainedEarnings|CommonStock|AdditionalPaidIn|Debt|LongTermDebt|ShortTermBorrowings|DeferredRevenue|MarketableSecurities|OtherAssets|OtherLiabilities|AccruedLiabilities)", re.I)),
]


def humanise(local_name: str) -> str:
    """``RevenueFromContractWithCustomer`` → ``Revenue From Contract With Customer``."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", local_name)
    return spaced.strip()


def classify_statement(local_name: str) -> str:
    for statement, pattern in _STATEMENT_PATTERNS:
        if pattern.search(local_name):
            return statement
    return "other"


def _local(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        return ns, name
    return "", tag


def _text(el: Element) -> str:
    return "".join(el.itertext()).strip()


def _strip_safe_doctype(data: bytes) -> bytes:
    """Drop a plain ``<!DOCTYPE html ...>`` (XHTML); refuse one with an internal subset."""
    match = re.search(rb"<!DOCTYPE[^>\[]*(\[)?", data[:4096], re.I)
    if not match:
        return data
    if match.group(1):  # "[" opens an internal subset: entity declarations
        raise UnsafeContent("xml_unsafe", "XML uses DTD or entity features.")
    end = data.find(b">", match.start())
    return data[: match.start()] + data[end + 1:]


def looks_like_ixbrl(data: bytes) -> bool:
    head = data[:300_000]
    return any(ns.encode() in head for ns in IX_NAMESPACES)


def looks_like_xbrl_instance(data: bytes) -> bool:
    head = data[:5000]
    return XBRLI.encode() in head and b"xbrl" in head.lower()


# ── contexts and units ───────────────────────────────────────────────────────

def _read_contexts(root: Element) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for el in root.iter(f"{{{XBRLI}}}context"):
        cid = el.get("id")
        if not cid:
            continue
        entity = ""
        ident = el.find(f".//{{{XBRLI}}}identifier")
        if ident is not None:
            entity = _text(ident)
        start = end = instant = ""
        period = el.find(f"{{{XBRLI}}}period")
        if period is not None:
            for child in period:
                _, name = _local(child.tag)
                if name == "startDate":
                    start = _text(child)
                elif name == "endDate":
                    end = _text(child)
                elif name == "instant":
                    instant = _text(child)
        dims = tuple(sorted(
            f"{m.get('dimension', '')}={_text(m)}"
            for m in el.iter(f"{{{XBRLDI}}}explicitMember")
        ))
        contexts[cid] = {
            "entity": entity,
            "period": f"{start}..{end}" if start and end else instant,
            "period_end": end or instant,
            "dims": dims,
        }
    return contexts


def _read_units(root: Element) -> dict[str, str]:
    units: dict[str, str] = {}
    for el in root.iter(f"{{{XBRLI}}}unit"):
        uid = el.get("id")
        if not uid:
            continue
        measures = [_text(m).split(":")[-1] for m in el.iter(f"{{{XBRLI}}}measure")]
        units[uid] = "/".join(measures) if len(measures) > 1 else (measures[0] if measures else "")
    return units


# ── value parsing ────────────────────────────────────────────────────────────

def parse_ix_value(text: str, *, fmt: str, scale: str | None, sign: str | None) -> Decimal | None:
    fmt = (fmt or "").split(":")[-1].lower().replace("-", "")
    raw = text.strip()
    if fmt in {"zerodash", "fixedzero", "fixedfalse"} or raw in {"-", "—", "–"}:
        value = Decimal(0)
    elif fmt in {"numwordsen", "numwordsno"} and raw.lower() in {"no", "none", "nil", "zero"}:
        value = Decimal(0)
    else:
        if fmt in {"numcommadecimal"}:
            raw = raw.replace(".", "").replace(",", ".")
        elif fmt in {"numspacecomma"}:
            raw = raw.replace(" ", "").replace(",", ".")
        else:
            raw = raw.replace(",", "").replace(" ", "")
        raw = raw.lstrip("$€£")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            return None
    if scale:
        try:
            value = value.scaleb(int(scale))
        except (ValueError, InvalidOperation):
            return None
    if sign == "-":
        value = -value
    return value


def _display(value: Decimal | None, fallback: str) -> str:
    if value is None:
        return fallback
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    whole, _, frac = text.partition(".")
    sign = "-" if whole.startswith("-") else ""
    whole = f"{int(whole.lstrip('-') or 0):,}"
    return f"{sign}{whole}" + (f".{frac}" if frac else "")


def _concept(name: str, namespaces: dict[str, str] | None = None, ns: str = "") -> tuple[str, str]:
    if ":" in name:
        prefix, _, local = name.partition(":")
        return f"{prefix}:{local}", local
    prefix = ""
    for known, short in KNOWN_PREFIXES.items():
        if ns.startswith(known):
            prefix = short
            break
    return (f"{prefix}:{name}" if prefix else name), name


# ── public API ───────────────────────────────────────────────────────────────

def parse_ixbrl(data: bytes) -> tuple[list[Fact], dict[str, str]]:
    """Return (facts, document metadata) for an inline XBRL XHTML document."""
    root = safe_fromstring(_strip_safe_doctype(data))
    contexts, units = _read_contexts(root), _read_units(root)
    facts: list[Fact] = []
    meta: dict[str, str] = {}
    seen: set[tuple] = set()

    for el in root.iter():
        ns, name = _local(el.tag)
        if ns not in IX_NAMESPACES:
            continue
        concept_name = el.get("name", "")
        if not concept_name:
            continue
        concept, local = _concept(concept_name)

        if name == "nonNumeric":
            if concept.startswith("dei:") and local in _DEI_TEXT_WHITELIST:
                meta.setdefault(_DEI_TEXT_WHITELIST[local], _text(el)[:200])
            continue
        if name != "nonFraction":
            continue
        if el.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
            continue

        ctx = contexts.get(el.get("contextRef", ""), {})
        value = parse_ix_value(_text(el), fmt=el.get("format", ""), scale=el.get("scale"), sign=el.get("sign"))
        unit = units.get(el.get("unitRef", ""), el.get("unitRef", ""))
        key = (concept, el.get("contextRef"), unit, value)
        if key in seen:           # the same fact is often tagged twice in a filing
            continue
        seen.add(key)
        facts.append(Fact(
            concept=concept, label=humanise(local),
            value=None if value is None else format(value, "f"),
            display=_display(value, _text(el)), unit=unit,
            period=ctx.get("period", ""), period_end=ctx.get("period_end", ""),
            decimals=el.get("decimals", ""), statement=classify_statement(local),
            dimensions=ctx.get("dims", ()), entity=ctx.get("entity", ""),
        ))
        if len(facts) > MAX_FACTS:
            raise UnsafeContent("xbrl_too_many_facts", "Document has too many XBRL facts.")
    return facts, meta


def parse_xbrl_instance(data: bytes) -> tuple[list[Fact], dict[str, str]]:
    """Return (facts, metadata) for a plain XBRL instance (``<xbrli:xbrl>``)."""
    root = safe_fromstring(_strip_safe_doctype(data))
    ns, name = _local(root.tag)
    if ns != XBRLI or name != "xbrl":
        raise UnsafeContent("unsupported_xml", "XML is not an XBRL instance document.")
    contexts, units = _read_contexts(root), _read_units(root)
    facts: list[Fact] = []
    meta: dict[str, str] = {}
    for el in root:
        fact_ns, local = _local(el.tag)
        if fact_ns in (XBRLI, "http://www.xbrl.org/2003/linkbase") or not el.get("contextRef"):
            continue
        concept, local = _concept(local, ns=fact_ns)
        text = _text(el)
        if not el.get("unitRef"):
            if concept.startswith("dei:") and local in _DEI_TEXT_WHITELIST:
                meta.setdefault(_DEI_TEXT_WHITELIST[local], text[:200])
            continue
        try:
            value: Decimal | None = Decimal(text.replace(",", ""))
        except InvalidOperation:
            value = None
        ctx = contexts.get(el.get("contextRef", ""), {})
        facts.append(Fact(
            concept=concept, label=humanise(local),
            value=None if value is None else format(value, "f"), display=_display(value, text),
            unit=units.get(el.get("unitRef", ""), ""), period=ctx.get("period", ""),
            period_end=ctx.get("period_end", ""), decimals=el.get("decimals", ""),
            statement=classify_statement(local), dimensions=ctx.get("dims", ()),
            entity=ctx.get("entity", ""),
        ))
        if len(facts) > MAX_FACTS:
            raise UnsafeContent("xbrl_too_many_facts", "Document has too many XBRL facts.")
    return facts, meta


def facts_to_blocks(facts: list[Fact]) -> list[Block]:
    """Group facts by (statement, period); segment (dimensional) facts get their own groups."""
    groups: dict[tuple[str, str, bool], list[Fact]] = {}
    for fact in facts:
        groups.setdefault((fact.statement, fact.period_end or fact.period, bool(fact.dimensions)), []).append(fact)
    order = {"income": 0, "balance": 1, "cashflow": 2, "other": 3}
    blocks: list[Block] = []
    for (statement, period, segmented), items in sorted(
        groups.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][2], kv[0][1])
    ):
        blocks.append(Block(kind=FACTS, facts=items))
    return blocks
