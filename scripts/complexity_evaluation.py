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

# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class JudgmentResult:
    """Structured result from the LLM judge."""
    less_complicated_text: str  # "A" or "B"
    confidence: float           # 0.0 – 1.0
    reasons: list[str]          # bullet-point reasons


# ── Prompt construction ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert linguist and vocabulary analyst acting as an impartial judge.
Your task is to compare two texts and determine which one uses LESS COMPLEX WORDS.

Focus ONLY on word-level complexity. Ignore sentence structure, grammar, punctuation, \
and syntactic patterns entirely.

Evaluate word complexity along these dimensions:
1. **Word rarity** – how common or uncommon each word is in everyday language
2. **Syllable count** – longer, multi-syllabic words tend to be harder
3. **Technical jargon** – domain-specific or specialised terminology
4. **Word origin** – Latinate/Greek-derived words vs. simple Germanic/Anglo-Saxon words
5. **Abstractness** – concrete, tangible words vs. abstract, conceptual words

Return your answer as a JSON object with EXACTLY this schema (no markdown fences):
{
  "less_complicated_text": "A" | "B",
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
        "Compare these two texts. Which one is less complicated? "
        "Provide your judgment as the specified JSON object."
    )


# ── Core judge logic ────────────────────────────────────────────────────────

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

    Parameters
    ----------
    text_a, text_b : str
        The two texts to compare.
    model : str
        Model name (default: gpt-4o-mini).

    Returns
    -------
    JudgmentResult
    """
    client = CopilotClient()
    await client.start()

    session = await client.create_session(
        {
            "model": model,
            "system_message": {"content": SYSTEM_PROMPT},
            "on_permission_request": on_permission_request,
        }
    )

    # Collect the final assistant message
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

    await session.destroy()
    await client.stop()

    # Parse the response
    raw = result_content[-1] if result_content else "{}"
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    data = json.loads(raw)

    return JudgmentResult(
        less_complicated_text=data["less_complicated_text"],
        confidence=float(data["confidence"]),
        reasons=data["reasons"],
    )


# ── Pretty printer ──────────────────────────────────────────────────────────

def display_result(result: JudgmentResult) -> None:
    """Print the judgment in a human-readable format."""
    print("\n" + "=" * 60)
    print("  LLM JUDGE – Text Complexity Comparison")
    print("=" * 60)

    simpler = {
        "A": "Text A is LESS complicated",
        "B": "Text B is LESS complicated",
    }.get(result.less_complicated_text, result.less_complicated_text)

    print(f"\n  Verdict   : {simpler}")
    print(f"  Confidence: {result.confidence:.0%}")

    print("\n  Reasons:")
    for i, reason in enumerate(result.reasons, 1):
        print(f"    {i}. {reason}")

    print("=" * 60 + "\n")


# ── CLI entry point ─────────────────────────────────────────────────────────

DEMO_TEXT_A = (
    "The cat sat on the mat. It was a sunny day. "
    "The cat liked to nap in the warm sunlight."
)

DEMO_TEXT_B = (
    "The epistemological ramifications of Gödelian incompleteness theorems "
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

    print("Comparing texts …")
    print(f"  Text A ({len(text_a)} chars): {text_a[:80]}…")
    print(f"  Text B ({len(text_b)} chars): {text_b[:80]}…")

    result = await judge_complexity(text_a, text_b)
    display_result(result)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
