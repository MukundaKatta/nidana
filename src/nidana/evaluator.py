"""LLM-as-judge clinical evaluator for grading model responses."""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel

from nidana.models import ModelAdapter, ModelResponse
from nidana.scorer import ClinicalScore
from nidana.vignettes.generator import ClinicalVignette

_SYSTEM_PROMPT = """\
You are an expert clinical evaluator for a medical AI benchmark. You will receive:
1. A clinical vignette (the case)
2. The correct diagnosis and differential
3. A model's response to the vignette
4. Dangerous diagnoses that should not be missed

Your task is to evaluate the model's clinical reasoning and produce a structured JSON score.

Scoring rubric:

**correct_diagnosis** (0.0-1.0):
- 1.0: Exact match or clinically equivalent diagnosis
- 0.75: Correct diagnosis using an acceptable synonym or slightly different phrasing
- 0.5: Partially correct (right organ system or disease category, but wrong specific diagnosis)
- 0.25: Correct diagnosis appears in the differential but not as the primary
- 0.0: Completely wrong diagnosis

**differential_quality** (0.0-1.0):
- 1.0: Comprehensive, well-ordered differential including all key possibilities
- 0.75: Good differential with most relevant diagnoses in reasonable order
- 0.5: Adequate differential but missing important considerations or poorly ordered
- 0.25: Sparse differential with only 1-2 relevant diagnoses
- 0.0: No differential provided or entirely irrelevant

**reasoning_quality** (0.0-1.0):
- 1.0: Expert-level synthesis of all clinical data with clear logical reasoning
- 0.75: Good reasoning connecting most clinical findings to the diagnosis
- 0.5: Basic reasoning present but missing key connections or contains inaccuracies
- 0.25: Minimal reasoning, mostly listing findings without synthesis
- 0.0: No reasoning provided or entirely incorrect reasoning

**safety_score** (0.0-1.0):
- 1.0: All dangerous diagnoses explicitly identified and discussed
- 0.75: Most dangerous diagnoses mentioned
- 0.5: Some dangerous diagnoses mentioned
- 0.25: Only one dangerous diagnosis mentioned in passing
- 0.0: No dangerous diagnoses addressed despite clear risks

Respond with ONLY a JSON object (no markdown fences) with these exact keys:
{
  "correct_diagnosis": <float>,
  "differential_quality": <float>,
  "reasoning_quality": <float>,
  "safety_score": <float>,
  "dangerous_misses_identified": [<list of dangerous diagnoses the model correctly flagged>],
  "dangerous_misses_missed": [<list of dangerous diagnoses the model failed to mention>],
  "evaluator_notes": "<brief justification for scores>"
}
"""


def _build_eval_prompt(vignette: ClinicalVignette, model_output: str) -> str:
    return (
        f"## Clinical Vignette\n{vignette.to_prompt()}\n\n"
        f"## Correct Diagnosis\n{vignette.correct_diagnosis}\n\n"
        f"## Expected Differential\n"
        + "\n".join(f"- {d}" for d in vignette.differential)
        + f"\n\n## Dangerous Diagnoses Not to Miss\n"
        + "\n".join(f"- {d}" for d in vignette.dangerous_misses)
        + f"\n\n## Model Response\n{model_output}\n\n"
        "Please evaluate the model response and return the JSON score."
    )


def _parse_eval_json(raw: str) -> dict:
    """Extract JSON from the evaluator response, tolerating markdown fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    return json.loads(cleaned)


class ClinicalEvaluator:
    """Grades model responses using an LLM-as-judge approach.

    The evaluator uses a strong LLM (the *judge*) to assess four dimensions
    of clinical reasoning: diagnostic accuracy, differential quality,
    reasoning depth, and safety awareness.
    """

    def __init__(self, judge: ModelAdapter) -> None:
        self._judge = judge

    def evaluate(
        self,
        vignette: ClinicalVignette,
        model_output: str,
        model_id: str,
    ) -> ClinicalScore:
        """Evaluate a single model response against a vignette."""
        eval_prompt = _build_eval_prompt(vignette, model_output)
        response: ModelResponse = self._judge.generate(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=eval_prompt,
        )
        try:
            data = _parse_eval_json(response.raw_text)
        except (json.JSONDecodeError, KeyError) as exc:
            # Fallback: return zero scores with error note
            return ClinicalScore(
                vignette_id=vignette.id,
                model_id=model_id,
                specialty=vignette.specialty,
                correct_diagnosis=0.0,
                differential_quality=0.0,
                reasoning_quality=0.0,
                safety_score=0.0,
                evaluator_notes=f"Failed to parse evaluator response: {exc}",
            )

        return ClinicalScore(
            vignette_id=vignette.id,
            model_id=model_id,
            specialty=vignette.specialty,
            correct_diagnosis=float(data.get("correct_diagnosis", 0.0)),
            differential_quality=float(data.get("differential_quality", 0.0)),
            reasoning_quality=float(data.get("reasoning_quality", 0.0)),
            safety_score=float(data.get("safety_score", 0.0)),
            dangerous_misses_identified=data.get("dangerous_misses_identified", []),
            dangerous_misses_missed=data.get("dangerous_misses_missed", []),
            evaluator_notes=data.get("evaluator_notes", ""),
        )

    def evaluate_batch(
        self,
        vignettes: list[ClinicalVignette],
        model_outputs: list[str],
        model_id: str,
    ) -> list[ClinicalScore]:
        """Evaluate a batch of responses."""
        if len(vignettes) != len(model_outputs):
            raise ValueError("Vignette and output lists must be the same length")
        return [
            self.evaluate(v, o, model_id)
            for v, o in zip(vignettes, model_outputs)
        ]
