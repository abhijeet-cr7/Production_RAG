"""Generation grader: checks an answer is grounded in its context and useful."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from online_pipeline.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a strict grader of retrieval-augmented answers. "
    "You receive a question, the retrieved context and a draft answer. "
    "Judge two things independently:\n"
    "1. grounded: every factual claim in the answer is supported by the context. "
    "Citation markers like [1] are fine. A claim that only comes from general "
    "knowledge, or that contradicts the context, makes the answer NOT grounded. "
    "An answer that only says the context lacks the information is grounded.\n"
    "2. answers_question: the answer actually resolves the question. A statement "
    "that the context lacks the information does NOT answer the question.\n"
    'Reply with JSON only, no prose: {"grounded": true|false, '
    '"answers_question": true|false, "unsupported_claims": ["..."]}. '
    "List at most three unsupported claims, empty when grounded."
)


@dataclass(frozen=True)
class GradeResult:
    """Verdict for one draft answer."""

    grounded: bool
    answers_question: bool
    unsupported_claims: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return self.grounded and self.answers_question

    def feedback(self) -> str:
        """Reviewer note handed to the generator for a stricter retry."""
        if not self.grounded:
            claims = "; ".join(self.unsupported_claims) or "some claims"
            return (
                "Your previous draft contained claims the context does not support "
                f"({claims}). Rewrite it using only facts stated in the context and "
                "drop anything you cannot cite."
            )
        return (
            "Your previous draft did not answer the question. Answer it directly "
            "from the context, or state exactly which information is missing."
        )


class GenerationGrader:
    """Grade a draft answer against its context with a single LLM call."""

    def __init__(self, provider: str, model: str | None = None) -> None:
        self._client = LLMClient(
            provider=provider,
            model=model or None,
            temperature=0.0,
            max_tokens=256,
        )

    def grade(self, query: str, context: str, answer: str) -> GradeResult:
        """Return the verdict, raising ValueError when the reply is unparseable."""
        user_message = (
            f"Question:\n{query}\n\nContext:\n{context}\n\nDraft answer:\n{answer}"
        )
        return self.parse(self._client.complete(_SYSTEM_PROMPT, user_message))

    @staticmethod
    def parse(raw: str) -> GradeResult:
        # Models sometimes wrap the JSON in code fences or add a preamble.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"Grader returned no JSON object: {raw[:80]!r}")
        data = json.loads(match.group(0))
        if "grounded" not in data or "answers_question" not in data:
            raise ValueError(f"Grader JSON missing verdict fields: {data}")
        claims = data.get("unsupported_claims") or []
        return GradeResult(
            grounded=_as_bool(data["grounded"]),
            answers_question=_as_bool(data["answers_question"]),
            unsupported_claims=[str(c) for c in claims][:3],
        )


def _as_bool(value: object) -> bool:
    # bool("false") is True, so string verdicts need explicit handling.
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes"}
    return bool(value)
