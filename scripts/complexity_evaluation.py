"""
LLM-as-a-Judge: Text Complexity Comparator

Compares two texts and judges which is more complicated,
providing detailed reasons and a confidence level.
Uses the GitHub Copilot SDK (premium requests).
"""

import asyncio
import json
import sys
from dataclasses import dataclass

from copilot import CopilotClient


def deny_all_permissions(request: dict, invocation: dict) -> dict:
    return {"kind": "denied-by-rules", "rules": []}


@dataclass
class JudgmentResult:
    """Structured result from the LLM judge."""

    text_A_complexity: float
    text_B_complexity: float
    confidence: float
    reasons: list[str]


class JudgmentParseError(RuntimeError):
    """Raised when the LLM judge output cannot be parsed."""


SYSTEM_PROMPT = """\
You are an expert linguist and vocabulary analyst acting as an impartial judge.
Your task is to compare two texts and determine which one uses LESS COMPLEX WORDS.

Focus ONLY on word-level complexity. Ignore sentence structure, grammar, punctuation,
and syntactic patterns entirely.

Evaluate word complexity along these dimensions:
1. Word rarity - how common or uncommon each word is in everyday language
2. Syllable count - longer, multi-syllabic words tend to be harder
3. Technical jargon - domain-specific or specialised terminology
4. Word origin - Latinate/Greek-derived words vs. simple Germanic/Anglo-Saxon words
5. Abstractness - concrete, tangible words vs. abstract, conceptual words

Return your answer as a JSON object with EXACTLY this schema (no markdown fences):
{
  "text_A_complexity": <float 0.0-1.0>,
  "text_B_complexity": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasons": ["reason 1", "reason 2", ...]
}

IMPORTANT: Return ONLY the raw JSON object. No markdown, no explanation, no tool use.
"""


def build_user_prompt(text_a: str, text_b: str) -> str:
    """Format the two texts into the user message."""
    return (
        f"=== TEXT A ===\n{text_a}\n\n"
        f"=== TEXT B ===\n{text_b}\n\n"
        "Compare these two texts for word-level complexity only. "
        "Provide your judgment as the specified JSON object."
    )


def _extract_json_object(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgmentParseError("No JSON object found in judge response.")
    return raw[start:end + 1]


def _parse_judgment_result(raw: str) -> JudgmentResult:
    cleaned = (raw or "").strip()
    if not cleaned:
        raise JudgmentParseError("Judge response was empty.")

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    payload = _extract_json_object(cleaned)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise JudgmentParseError(f"Judge response was not valid JSON: {exc}") from exc

    required_keys = {
        "text_A_complexity",
        "text_B_complexity",
        "confidence",
        "reasons",
    }
    missing = required_keys - data.keys()
    if missing:
        raise JudgmentParseError(f"Judge response missing keys: {sorted(missing)}")

    reasons = data["reasons"]
    if not isinstance(reasons, list):
        raise JudgmentParseError("Judge response field 'reasons' must be a list.")

    return JudgmentResult(
        text_A_complexity=float(data["text_A_complexity"]),
        text_B_complexity=float(data["text_B_complexity"]),
        confidence=float(data["confidence"]),
        reasons=[str(reason) for reason in reasons],
    )


async def judge_complexity(
    text_a: str,
    text_b: str,
    *,
    model: str = "gpt-5-mini",
    on_permission_request=deny_all_permissions,
) -> JudgmentResult:
    """
    Call the LLM via GitHub Copilot SDK to judge text complexity.

    Requires the Copilot CLI to be installed and authenticated.
    Uses your Copilot premium requests.
    """
    client = CopilotClient()
    session = None

    try:
        await client.start()

        session = await client.create_session(
            {
                "model": model,
                "system_message": {"content": SYSTEM_PROMPT},
                "on_permission_request": on_permission_request,
            }
        )

        done = asyncio.Event()
        result_content: list[str] = []

        def on_event(event):
            if event.type.value == "assistant.message":
                result_content.append(event.data.content)
            elif event.type.value == "session.idle":
                done.set()

        session.on(on_event)
        await session.send({"prompt": build_user_prompt(text_a, text_b)})
        await done.wait()

        raw = result_content[-1] if result_content else ""
        return _parse_judgment_result(raw)
    finally:
        if session is not None:
            await session.destroy()
        await client.stop()


def display_result(result: JudgmentResult) -> None:
    """Print the judgment in a human-readable format."""
    print("\n" + "=" * 60)
    print("  LLM JUDGE - Text Complexity Comparison")
    print("=" * 60)

    if result.text_A_complexity < result.text_B_complexity:
        verdict = "Text A uses LESS complex words"
    elif result.text_B_complexity < result.text_A_complexity:
        verdict = "Text B uses LESS complex words"
    else:
        verdict = "Tie (word complexity appears equal)"

    print(f"\n  Verdict   : {verdict}")
    print(f"  A score   : {result.text_A_complexity:.3f}")
    print(f"  B score   : {result.text_B_complexity:.3f}")
    print(f"  Confidence: {result.confidence:.0%}")

    print("\n  Reasons:")
    for i, reason in enumerate(result.reasons, 1):
        print(f"    {i}. {reason}")

    print("=" * 60 + "\n")


DEMO_TEXT_A = (
    "The cat sat on the mat. It was a sunny day. "
    "The cat liked to nap in the warm sunlight."
)

DEMO_TEXT_B = (
    "The epistemological ramifications of Godelian incompleteness theorems "
    "necessitate a fundamental re-evaluation of axiomatic foundations in "
    "formal systems, particularly regarding the inherent limitations of "
    "self-referential consistency proofs within Peano arithmetic and its "
    "recursively enumerable extensions."
)


async def async_main() -> None:
    """Run the complexity judge, either with CLI args or built-in demo texts."""
    if len(sys.argv) == 3:
        with open(sys.argv[1], encoding="utf-8") as f:
            text_a = f.read().strip()
        with open(sys.argv[2], encoding="utf-8") as f:
            text_b = f.read().strip()
    else:
        text_a, text_b = DEMO_TEXT_A, DEMO_TEXT_B

    print("Comparing texts ...")
    print(f"  Text A ({len(text_a)} chars): {text_a[:80]}...")
    print(f"  Text B ({len(text_b)} chars): {text_b[:80]}...")

    result = await judge_complexity(text_a, text_b)
    display_result(result)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
