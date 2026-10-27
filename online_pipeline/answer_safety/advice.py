"""Detect requests for personalised investment advice.

A document Q&A system can say what filings report; it should not tell someone what
to buy or sell. Matching requests are declined before any model call when the
tenant policy asks for it. This is a keyword screen, so it is deliberately narrow:
it targets requests for a recommendation, not questions *about* ratings or targets.
"""

from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"\b(?:should|shall|would|could)\s+(?:i|we)\s+(?:buy|sell|short|invest|hold|trade|purchase|dump|avoid)\b", re.I),
    re.compile(r"\b(?:can|could|would)\s+you\s+(?:please\s+)?recommend\b", re.I),
    re.compile(r"\brecommend(?:ation)?s?\s+(?:me\s+)?(?:a|an|some|the\s+best|which)?\s*(?:good\s+)?(?:stocks?|funds?|etfs?|investments?|securities|shares|bonds?)\b", re.I),
    re.compile(r"\bwhich\s+(?:stocks?|funds?|etfs?|shares|securities)\s+(?:should|to|do\s+you)\b", re.I),
    re.compile(r"\bis\s+(?:now|it|this)\b.{0,30}\b(?:good\s+time|right\s+time)\s+to\s+(?:buy|sell|invest)\b", re.I),
    re.compile(r"\bis\s+.{1,40}\s+a\s+good\s+(?:buy|investment|stock|sell)\b", re.I),
    re.compile(r"\bwhat\s+should\s+(?:i|we)\s+(?:invest|buy|sell|do\s+with\s+my)\b", re.I),
    re.compile(r"\bgive\s+me\s+(?:some\s+)?(?:financial|investment|stock)\s+(?:advice|tips|picks)\b", re.I),
    re.compile(r"\bwhere\s+should\s+i\s+(?:put|invest)\s+my\s+money\b", re.I),
]

REFUSAL = (
    "I can't give personalised investment recommendations. I can tell you what the "
    "documents report, such as figures, trends and disclosures, if you ask about those."
)


def is_investment_advice_request(query: str) -> bool:
    return any(p.search(query) for p in _PATTERNS)
