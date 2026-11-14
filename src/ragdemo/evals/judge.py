"""LLM judge for answer quality, from a different model family than the generator.

Two judgements, both returned as strict JSON:
  faithfulness: split the answer into atomic claims; is each one fully supported by the cited passages?
  correctness:  does the answer agree with the human-written reference answer?

Every verdict is cached on disk under a hash of (judge model, prompt version, inputs), so re-running
the evaluation after a retrieval change re-judges only the answers that actually changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

PROMPT_VERSION = "v1"

FAITHFULNESS_SYSTEM = """You are a strict fact-checker for a question-answering system.
You receive an ANSWER and the PASSAGES it cites. Split the answer into atomic factual claims
(one checkable statement each; ignore citation markers like [1]). For each claim decide whether
the passages, taken together, fully support it. A claim that goes beyond the passages, contradicts
them, or adds unsupported detail is NOT supported. Be literal: do not use outside knowledge.
Return JSON only."""

CORRECTNESS_SYSTEM = """You compare a CANDIDATE answer with a REFERENCE answer written by a human
annotator for the same QUESTION. Judge whether the candidate conveys the same essential information.
  correct:   the candidate states the key facts of the reference (wording may differ; extra detail is fine)
  partial:   the candidate is on topic and partly right but misses or muddles a key fact
  incorrect: the candidate contradicts the reference, answers a different question, or says it cannot answer
Return JSON only."""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["claim", "supported", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


class Judge(Protocol):
    name: str

    def faithfulness(self, answer: str, passages: list[str]) -> dict: ...

    def correctness(self, question: str, answer: str, references: list[str]) -> dict: ...


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def parse_faithfulness(raw: str) -> dict:
    """Tolerant parser: returns {'claims': [...], 'supported': k, 'total': n, 'score': k/n}."""
    try:
        data = json.loads(raw)
        claims = [c for c in data.get("claims", []) if isinstance(c, dict) and "supported" in c]
    except (json.JSONDecodeError, AttributeError):
        claims = []
    total = len(claims)
    supported = sum(1 for c in claims if bool(c["supported"]))
    return {
        "claims": claims,
        "supported": supported,
        "total": total,
        "score": (supported / total) if total else None,
    }


def parse_correctness(raw: str) -> dict:
    try:
        data = json.loads(raw)
        verdict = str(data.get("verdict", "")).lower()
    except (json.JSONDecodeError, AttributeError):
        data, verdict = {}, ""
    if verdict not in ("correct", "partial", "incorrect"):
        verdict = "unparsed"
    return {"verdict": verdict, "reason": str(data.get("reason", "")) if isinstance(data, dict) else ""}


class OpenAIJudge:
    def __init__(self, api_key: str, model: str = "gpt-5-mini", cache_dir: Path = Path("data/cache/judge")):
        from openai import OpenAI

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key, max_retries=6)
        self.name = model
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)

    def _call(self, system: str, user: str, schema_name: str, schema: dict) -> str:
        key = _key(self.name, PROMPT_VERSION, schema_name, system, user)
        path = self._cache / f"{key}.json"
        if path.exists():
            return path.read_text(encoding="utf-8")
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
            reasoning_effort="low",
        )
        raw = resp.choices[0].message.content or "{}"
        path.write_text(raw, encoding="utf-8")
        return raw

    def faithfulness(self, answer: str, passages: list[str]) -> dict:
        blocks = (
            "\n\n".join(f"PASSAGE {i}: {p}" for i, p in enumerate(passages, start=1)) or "PASSAGES: (none)"
        )
        user = f"{blocks}\n\nANSWER: {answer}"
        return parse_faithfulness(self._call(FAITHFULNESS_SYSTEM, user, "faithfulness", FAITHFULNESS_SCHEMA))

    def correctness(self, question: str, answer: str, references: list[str]) -> dict:
        refs = "\n".join(f"- {r}" for r in references) or "- (none)"
        user = f"QUESTION: {question}\n\nREFERENCE ANSWER(S):\n{refs}\n\nCANDIDATE ANSWER: {answer}"
        return parse_correctness(self._call(CORRECTNESS_SYSTEM, user, "correctness", CORRECTNESS_SCHEMA))


class FakeJudge:
    """Deterministic judge for tests: a claim is supported if its words appear in some passage."""

    name = "fake"

    def faithfulness(self, answer: str, passages: list[str]) -> dict:
        joined = " ".join(passages).lower()
        claims = []
        for sentence in [s.strip() for s in answer.replace("[", " [").split(".") if s.strip()]:
            words = [w for w in sentence.lower().split() if w.isalpha()]
            supported = bool(words) and all(w in joined for w in words)
            claims.append({"claim": sentence, "supported": supported, "evidence": ""})
        return parse_faithfulness(json.dumps({"claims": claims}))

    def correctness(self, question: str, answer: str, references: list[str]) -> dict:
        verdict = "correct" if any(r.lower() in answer.lower() for r in references if r) else "incorrect"
        return parse_correctness(json.dumps({"verdict": verdict, "reason": "fake"}))
