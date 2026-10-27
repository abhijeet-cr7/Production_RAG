"""Heuristic screen for prompt-injection payloads in ingested text.

Matching content is *quarantined* (stored but excluded from retrieval until an
administrator releases it). This is a heuristic filter, not a guarantee: it
catches the common, blunt attacks; the generator prompt treats context as
untrusted data and output sanitisation closes the exfiltration channel as
further layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HIGH = "high"
MEDIUM = "medium"
LOW = "low"


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str


_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    ("instruction_override", HIGH, re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
        r"\b(?:previous|prior|above|earlier|preceding|all|any|your|the)\b[^.\n]{0,30}?"
        r"\b(?:instructions?|prompts?|rules|guidelines|directives|context|messages)\b", re.I)),
    ("new_instructions", HIGH, re.compile(
        r"\b(?:new|updated|revised|real|actual)\s+(?:system\s+)?instructions?\s*[:\-]", re.I)),
    ("system_prompt_probe", HIGH, re.compile(
        r"\b(?:reveal|show|print|repeat|output|leak|display)\b[^.\n]{0,30}?"
        r"\b(?:system|hidden|initial|original)\s+(?:prompt|instructions?|message)\b", re.I)),
    ("jailbreak_persona", HIGH, re.compile(
        r"\byou\s+are\s+now\b[^.\n]{0,40}?"
        r"\b(?:DAN|developer\s+mode|unrestricted|jailbroken|without\s+restrictions|no\s+longer\s+bound)\b|"
        r"\b(?:developer|DAN|god)\s+mode\s+(?:enabled|activated|on)\b", re.I)),
    ("role_marker_token", HIGH, re.compile(
        r"<\|?(?:im_start|im_end|system|assistant|endoftext|begin_of_text|eot_id)\|?>|"
        r"\[/?INST\]|<<\s*/?SYS\s*>>|<\s*/?\s*system\s*>", re.I)),
    ("tool_call_syntax", HIGH, re.compile(
        r"<\s*/?\s*(?:tool_call|function_call|tool_use|antml:invoke)\b|"
        r'"(?:tool_calls?|function_call)"\s*:', re.I)),
    ("exfil_markdown_image", HIGH, re.compile(
        r"!\[[^\]]*\]\(\s*https?://[^)\s]*[?&=][^)]*\)", re.I)),
    ("exfil_instruction", HIGH, re.compile(
        r"\b(?:send|post|forward|transmit|upload|exfiltrate|leak)\b[^.\n]{0,60}?"
        r"\b(?:conversation|chat|history|answer|response|data|context|secrets?|keys?)\b[^.\n]{0,60}?"
        r"\bhttps?://", re.I)),
    ("role_prefix_line", MEDIUM, re.compile(r"(?im)^\s*(?:system|assistant|developer)\s*:\s*\S")),
    ("encoded_blob", LOW, re.compile(r"[A-Za-z0-9+/]{240,}={0,2}")),
]


def scan_for_injection(text: str) -> list[Finding]:
    """Return one :class:`Finding` per rule that matched *text*."""
    return [Finding(rule, severity) for rule, severity, pattern in _RULES if pattern.search(text)]


def should_quarantine(findings: list[Finding]) -> bool:
    return any(f.severity == HIGH for f in findings)


# ── invisible Unicode ────────────────────────────────────────────────────────

# Unicode "tag" characters let an attacker write ASCII a human cannot see.
_TAG_CHARS = re.compile("[\U000e0000-\U000e007f]")
_BIDI_OVERRIDES = re.compile("[‪-‮⁦-⁩]")
# Zero-width, directional marks, soft hyphen, invisible operators, BOM, tags.
INVISIBLE_CHARS = re.compile(
    "[­؜᠎​-‏‪-‮⁠-⁤⁦-⁯﻿"
    "￹-￻\U000e0000-\U000e007f]"
)


def unicode_findings(text: str) -> list[Finding]:
    """Findings for invisible-character tricks, checked on the *raw* text."""
    found: list[Finding] = []
    if _TAG_CHARS.search(text):
        found.append(Finding("unicode_tag_smuggling", HIGH))
    if _BIDI_OVERRIDES.search(text):
        found.append(Finding("bidi_override", MEDIUM))
    return found
