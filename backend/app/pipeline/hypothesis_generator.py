"""
hypothesis_generator.py

The one place in the pipeline where an LLM is allowed to speculate.
Everything upstream (report_generator.py) states only what the
numbers directly show. This module takes those same grounded
statistics and asks an LLM to propose candidate explanations --
explicitly labeled as unvalidated AI-generated hypotheses, never
mixed into the factual report.

Requires ANTHROPIC_API_KEY in the environment. If it's not set, the
function returns an empty list rather than raising, so the rest of
the app keeps working without this feature configured.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class Hypothesis:
    text: str
    based_on: str  # which finding motivated it, for traceability
    label: str = "AI-GENERATED HYPOTHESIS — NOT VALIDATED"


SYSTEM_PROMPT = """You are assisting a researcher by proposing candidate \
scientific hypotheses from statistical patterns already found in their data \
(clusters, feature importances, variance structure). You are NOT confirming \
anything -- you are suggesting directions worth testing experimentally.

Rules:
- Ground every hypothesis in the specific numbers provided. Do not invent \
findings that aren't in the input.
- Prefer mechanistic or biological/physical plausibility over vague statements.
- Explicitly note what evidence would be needed to test each hypothesis.
- Output strict JSON: a list of objects with keys "hypothesis", "based_on", \
and "how_to_test". No prose outside the JSON.
"""


def generate_hypotheses(
    report_findings: list[str],
    domain_context: str = "",
    max_hypotheses: int = 3,
    model: str = "claude-sonnet-4-6",
) -> list[Hypothesis]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    import anthropic  # noqa: WPS433

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = (
        f"Domain context: {domain_context or 'not specified'}\n\n"
        f"Grounded findings from the analysis pipeline:\n"
        + "\n".join(f"- {f}" for f in report_findings)
        + f"\n\nPropose up to {max_hypotheses} hypotheses as JSON."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    raw_text = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    hypotheses = []
    for item in parsed[:max_hypotheses]:
        hypotheses.append(
            Hypothesis(
                text=f"{item.get('hypothesis', '').strip()} "
                     f"(how to test: {item.get('how_to_test', 'not specified').strip()})",
                based_on=item.get("based_on", "").strip(),
            )
        )
    return hypotheses
