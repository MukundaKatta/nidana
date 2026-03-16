"""Clinical scoring models for the Nidana benchmark."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from nidana.vignettes.specialties import MedicalSpecialty


class ClinicalScore(BaseModel):
    """Scores for a single vignette evaluation."""

    vignette_id: str
    model_id: str
    specialty: MedicalSpecialty

    # Core metrics (0.0 -- 1.0)
    correct_diagnosis: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "1.0 = exact match, 0.75 = equivalent/synonym, 0.5 = partially correct "
            "(right category), 0.25 = in the differential but not primary, 0.0 = wrong"
        ),
    )
    differential_quality: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Evaluates ranked differential: completeness, relevance, ordering. "
            "1.0 = comprehensive and well-ordered, 0.0 = absent or irrelevant"
        ),
    )
    reasoning_quality: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Quality of clinical reasoning: logical synthesis of history, exam, labs, "
            "imaging. 1.0 = expert-level reasoning, 0.0 = no reasoning provided"
        ),
    )
    safety_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "1.0 = all dangerous diagnoses identified and addressed, "
            "0.5 = partially addressed, 0.0 = dangerous misses not mentioned"
        ),
    )

    # Optional metadata
    dangerous_misses_identified: list[str] = Field(default_factory=list)
    dangerous_misses_missed: list[str] = Field(default_factory=list)
    evaluator_notes: str = ""

    @property
    def composite_score(self) -> float:
        """Weighted composite: diagnosis 40%, differential 20%, reasoning 25%, safety 15%."""
        return (
            self.correct_diagnosis * 0.40
            + self.differential_quality * 0.20
            + self.reasoning_quality * 0.25
            + self.safety_score * 0.15
        )


class SpecialtyAggregateScore(BaseModel):
    """Aggregated scores for a model within a single specialty."""

    model_id: str
    specialty: MedicalSpecialty
    n_vignettes: int

    mean_correct_diagnosis: float
    mean_differential_quality: float
    mean_reasoning_quality: float
    mean_safety_score: float
    mean_composite: float

    @staticmethod
    def from_scores(
        model_id: str,
        specialty: MedicalSpecialty,
        scores: list[ClinicalScore],
    ) -> "SpecialtyAggregateScore":
        n = len(scores)
        if n == 0:
            return SpecialtyAggregateScore(
                model_id=model_id,
                specialty=specialty,
                n_vignettes=0,
                mean_correct_diagnosis=0.0,
                mean_differential_quality=0.0,
                mean_reasoning_quality=0.0,
                mean_safety_score=0.0,
                mean_composite=0.0,
            )
        return SpecialtyAggregateScore(
            model_id=model_id,
            specialty=specialty,
            n_vignettes=n,
            mean_correct_diagnosis=sum(s.correct_diagnosis for s in scores) / n,
            mean_differential_quality=sum(s.differential_quality for s in scores) / n,
            mean_reasoning_quality=sum(s.reasoning_quality for s in scores) / n,
            mean_safety_score=sum(s.safety_score for s in scores) / n,
            mean_composite=sum(s.composite_score for s in scores) / n,
        )


class ModelAggregateScore(BaseModel):
    """Overall aggregate scores for a model across all specialties."""

    model_id: str
    total_vignettes: int
    specialty_scores: list[SpecialtyAggregateScore]

    mean_correct_diagnosis: float
    mean_differential_quality: float
    mean_reasoning_quality: float
    mean_safety_score: float
    mean_composite: float

    @staticmethod
    def from_clinical_scores(
        model_id: str,
        scores: list[ClinicalScore],
    ) -> "ModelAggregateScore":
        from collections import defaultdict

        by_specialty: dict[MedicalSpecialty, list[ClinicalScore]] = defaultdict(list)
        for s in scores:
            by_specialty[s.specialty].append(s)

        specialty_aggs = [
            SpecialtyAggregateScore.from_scores(model_id, sp, sp_scores)
            for sp, sp_scores in sorted(by_specialty.items(), key=lambda x: x[0].value)
        ]

        n = len(scores)
        if n == 0:
            return ModelAggregateScore(
                model_id=model_id,
                total_vignettes=0,
                specialty_scores=[],
                mean_correct_diagnosis=0.0,
                mean_differential_quality=0.0,
                mean_reasoning_quality=0.0,
                mean_safety_score=0.0,
                mean_composite=0.0,
            )

        return ModelAggregateScore(
            model_id=model_id,
            total_vignettes=n,
            specialty_scores=specialty_aggs,
            mean_correct_diagnosis=sum(s.correct_diagnosis for s in scores) / n,
            mean_differential_quality=sum(s.differential_quality for s in scores) / n,
            mean_reasoning_quality=sum(s.reasoning_quality for s in scores) / n,
            mean_safety_score=sum(s.safety_score for s in scores) / n,
            mean_composite=sum(s.composite_score for s in scores) / n,
        )
