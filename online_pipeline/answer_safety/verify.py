"""Deterministic verification of an answer against the sources it cites.

An LLM grader can say an answer "sounds grounded"; it cannot reliably tell that
$391.0 billion should have been $391.5 billion, or that a 2023 figure was quoted
for 2024. This does, with arithmetic:

* every ``[n]`` must point at a real source;
* every sentence that states a figure must carry a citation;
* every stated figure must match a value in a *cited* source, scale-aware and
  within the rounding the answer itself implies ($391.0 billion matches 391,035
  in millions, not 391,535);
* a figure that is not in the sources may still be accepted if it is a growth
  rate, difference, sum or ratio of figures that are, in the same sentence
  (``up 8.2% from $365.8B to $391.0B``): the calculator checks the model's maths;
* the period and currency of the matched source value must agree with the answer.

It reports; the gateway decides what to do (regenerate, warn or abstain).
Limit: it verifies numbers, not meaning. A correct number attached to the wrong
claim passes; that is what the semantic grader is still for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from offline_pipeline.tables.numbers import ProseNumber, extract_prose_numbers

VERIFIED = "verified"
DERIVED = "derived_verified"
NOT_FOUND = "not_found"
PERIOD_MISMATCH = "period_mismatch"
UNIT_MISMATCH = "unit_mismatch"
UNCITED = "uncited"
FAILED = {NOT_FOUND, PERIOD_MISMATCH, UNIT_MISMATCH, UNCITED}

_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'•-])|\n+")
_ABSTAIN = re.compile(
    r"(?:does\s+not|doesn't|do\s+not|don't)\s+(?:contain|provide|include|have|mention)\b|"
    r"\b(?:cannot|can't|unable\s+to|could\s+not|couldn't)\s+(?:find|determine|answer|produce|verify)\b|"
    r"\b(?:no|not\s+enough|insufficient)\s+(?:relevant\s+)?information\b|"
    r"\bnot\s+going\s+to\s+guess\b|\bi\s+don't\s+know\b",
    re.I,
)
_NON_CLAIM_TOKENS = [
    re.compile(r"\bFY\s?\d{2,4}\b", re.I),
    re.compile(r"\b[HQ][1-4]\b(?:\s*(?:19|20)\d{2})?", re.I),
    re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*(?:19|20)\d{2})?", re.I),
    re.compile(r"(?m)^\s*\d{1,2}[.)]\s"),
    re.compile(r"\b(?:Item|Note|Table|Figure|Exhibit|Section|Page|Part)\s+\d+[A-Za-z]?(?:\.\d+)?", re.I),
]
# Not \b: "FY2023" has no word boundary between the Y and the 2.
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


@dataclass
class NumberCheck:
    text: str
    status: str
    sentence: str = ""
    citations: list[int] = field(default_factory=list)
    source: str = ""        # e.g. "[2] Net sales / FY2024"


@dataclass
class Verification:
    abstained: bool = False
    has_citation: bool = True
    invalid_citations: list[int] = field(default_factory=list)
    checks: list[NumberCheck] = field(default_factory=list)

    @property
    def failed(self) -> list[NumberCheck]:
        return [c for c in self.checks if c.status in FAILED]

    @property
    def ok(self) -> bool:
        return (
            self.abstained
            or (self.has_citation and not self.invalid_citations and not self.failed)
        )

    def feedback(self) -> str:
        problems: list[str] = []
        if not self.has_citation:
            problems.append("Cite your sources with [n] after every statement that uses them.")
        if self.invalid_citations:
            problems.append("These citations do not exist: " + ", ".join(f"[{n}]" for n in self.invalid_citations) + ".")
        for check in self.failed:
            reason = {
                NOT_FOUND: "does not appear in the cited sources",
                PERIOD_MISMATCH: "is from a different period than the one you stated",
                UNIT_MISMATCH: "uses a different currency or unit than the source",
                UNCITED: "has no citation",
            }[check.status]
            problems.append(f"The figure '{check.text}' {reason}.")
        return (
            "Your previous draft failed verification. " + " ".join(problems)
            + " Rewrite it using only figures stated in the sources, with the same period and units, "
              "and cite each one as [n]."
        )

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for c in self.checks:
            counts[c.status] = counts.get(c.status, 0) + 1
        return {
            "ok": self.ok, "abstained": self.abstained, "has_citation": self.has_citation,
            "invalid_citations": self.invalid_citations, "counts": counts,
            "checks": [
                {"figure": c.text, "status": c.status, "citations": c.citations, "source": c.source}
                for c in self.checks
            ],
        }


# ── helpers ──────────────────────────────────────────────────────────────────

def _dec(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


@dataclass
class _Candidate:
    value: Decimal
    kind: str
    currency: str
    year: str
    label: str
    ref: int


def _year_of(entry: dict[str, Any]) -> str:
    end = str(entry.get("period_end") or "")
    if re.match(r"(?:19|20)\d{2}", end):
        return end[:4]
    match = _YEAR.search(str(entry.get("column") or ""))
    return match.group(0) if match else ""


def _candidates(chunk: dict[str, Any], ref: int) -> list[_Candidate]:
    out: list[_Candidate] = []
    meta = chunk.get("metadata") or {}
    for entry in meta.get("numerics") or []:
        value = _dec(entry.get("value"))
        if value is None:
            continue
        out.append(_Candidate(
            value=abs(value), kind=str(entry.get("kind", "number")),
            currency=str(entry.get("currency") or ""), year=_year_of(entry),
            label=f"[{ref}] {entry.get('row', '')} / {entry.get('column', '')}".strip(" /"), ref=ref,
        ))
    for number in extract_prose_numbers(chunk.get("text", "")):
        out.append(_Candidate(
            value=abs(number.value), kind=number.kind, currency=number.currency,
            year="", label=f"[{ref}] {number.text}", ref=ref,
        ))
    return out


def _is_claim(number: ProseNumber, years_masked: bool) -> bool:
    """Skip numbers that are labels or trivia, not quantities worth verifying."""
    plain_year = (
        number.kind == "number" and number.value == number.value.to_integral_value()
        and 1900 <= abs(number.value) <= 2100 and "," not in number.text
    )
    if plain_year:
        return False
    small_bare = (
        number.kind == "number" and abs(number.value) < 100
        and number.text.strip().replace("-", "").replace(",", "").isdigit()
    )
    return not small_bare


def _close(answer: ProseNumber, source: Decimal) -> bool:
    return abs(abs(answer.value) - source) <= answer.quantum


def _kind_compatible(answer: ProseNumber, source: _Candidate) -> bool:
    if answer.kind == "percent" or source.kind == "percent":
        return answer.kind == source.kind
    return True


def _derived_matches(target: ProseNumber, operands: list[Decimal]) -> bool:
    """Is *target* a change %, difference, sum or ratio of any two *operands*?"""
    pairs = [(a, b) for i, a in enumerate(operands) for b in operands[i + 1:]]
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            if x != 0 and target.kind == "percent":
                for candidate in ((y - x) / x * 100, (x - y) / x * 100, x / y * 100 if y else None):
                    if candidate is not None and abs(abs(target.value) - abs(candidate)) <= target.quantum:
                        return True
            if target.kind != "percent":
                for candidate in (abs(x - y), x + y):
                    if abs(abs(target.value) - candidate) <= target.quantum:
                        return True
    return False


def _clean_for_numbers(sentence: str) -> str:
    text = _CITATION.sub(" ", sentence)
    for pattern in _NON_CLAIM_TOKENS:
        text = pattern.sub(" ", text)
    return text


# ── public API ───────────────────────────────────────────────────────────────

def verify_answer(answer: str, chunks: list[dict[str, Any]], *, require_citations: bool = True) -> Verification:
    """Verify *answer* against *chunks*, numbered ``[1]..[len(chunks)]`` as in the prompt."""
    result = Verification()
    if not answer.strip() or _ABSTAIN.search(answer) and not any(
        c for c in extract_prose_numbers(_CLEAN_ABSTAIN(answer)) if _is_claim(c, True)
    ):
        result.abstained = True
        return result

    all_refs = {m for g in _CITATION.findall(answer) for m in _citation_numbers(g)}
    result.invalid_citations = sorted(n for n in all_refs if not 1 <= n <= len(chunks))
    result.has_citation = bool(all_refs) or not require_citations or not chunks

    candidates = {ref: _candidates(chunk, ref) for ref, chunk in enumerate(chunks, start=1)}

    for sentence in (s.strip() for s in _SENTENCE_SPLIT.split(answer)):
        if not sentence:
            continue
        cited = sorted({n for g in _CITATION.findall(sentence) for n in _citation_numbers(g)
                        if 1 <= n <= len(chunks)})
        text = _clean_for_numbers(sentence)
        numbers = [n for n in extract_prose_numbers(text) if _is_claim(n, True)]
        if not numbers:
            continue
        sentence_years = set(_YEAR.findall(sentence))

        pool_refs = cited or list(candidates)
        pool = [c for ref in pool_refs for c in candidates.get(ref, [])]
        verified_values: list[Decimal] = []
        pending: list[ProseNumber] = []

        for number in numbers:
            match = _find_match(number, pool, sentence_years)
            if match is None:
                pending.append(number)
                continue
            candidate, status = match
            if not cited and require_citations:
                status = UNCITED
            result.checks.append(NumberCheck(number.text, status, sentence, cited, candidate.label))
            if status in (VERIFIED,):
                verified_values.append(abs(number.value))

        for number in pending:
            if _derived_matches(number, verified_values):
                status = DERIVED if (cited or not require_citations) else UNCITED
                result.checks.append(NumberCheck(number.text, status, sentence, cited, "derived from figures in the sentence"))
            else:
                result.checks.append(NumberCheck(number.text, UNCITED if (not cited and require_citations) else NOT_FOUND,
                                                 sentence, cited, ""))
    return result


def _CLEAN_ABSTAIN(answer: str) -> str:
    return _clean_for_numbers(answer)


def _citation_numbers(group: str) -> list[int]:
    return [int(n) for n in re.split(r"[,;]", group) if n.strip().isdigit()]


def _find_match(
    number: ProseNumber, pool: list[_Candidate], sentence_years: set[str]
) -> tuple[_Candidate, str] | None:
    """Best source value for *number*, or None. Prefers a period the sentence names."""
    hits = [c for c in pool if _kind_compatible(number, c) and _close(number, c.value)]
    if not hits:
        return None
    # Currency disagreement is a unit problem even though the value matched.
    comparable = [c for c in hits if not (number.currency and c.currency and number.currency != c.currency)]
    if not comparable:
        return hits[0], UNIT_MISMATCH
    if sentence_years:
        dated = [c for c in comparable if c.year]
        in_period = [c for c in dated if c.year in sentence_years]
        if in_period:
            return in_period[0], VERIFIED
        if dated and len(dated) == len(comparable):
            return dated[0], PERIOD_MISMATCH
    return comparable[0], VERIFIED
