"""Tests for the clinical scoring module."""

from __future__ import annotations

import pytest

from nidana.scorer import ClinicalScore, ModelAggregateScore, SpecialtyAggregateScore
from nidana.vignettes.specialties import MedicalSpecialty


class TestClinicalScore:
    """Tests for individual clinical scores."""

    def test_composite_score_calculation(self) -> None:
        score = ClinicalScore(
            vignette_id="test001",
            model_id="test-model",
            specialty=MedicalSpecialty.CARDIOLOGY,
            correct_diagnosis=1.0,
            differential_quality=0.8,
            reasoning_quality=0.9,
            safety_score=0.7,
        )
        # 1.0*0.40 + 0.8*0.20 + 0.9*0.25 + 0.7*0.15 = 0.40 + 0.16 + 0.225 + 0.105 = 0.89
        assert abs(score.composite_score - 0.89) < 0.001

    def test_composite_score_perfect(self) -> None:
        score = ClinicalScore(
            vignette_id="test002",
            model_id="test-model",
            specialty=MedicalSpecialty.NEUROLOGY,
            correct_diagnosis=1.0,
            differential_quality=1.0,
            reasoning_quality=1.0,
            safety_score=1.0,
        )
        assert abs(score.composite_score - 1.0) < 0.001

    def test_composite_score_zero(self) -> None:
        score = ClinicalScore(
            vignette_id="test003",
            model_id="test-model",
            specialty=MedicalSpecialty.PULMONOLOGY,
            correct_diagnosis=0.0,
            differential_quality=0.0,
            reasoning_quality=0.0,
            safety_score=0.0,
        )
        assert score.composite_score == 0.0

    def test_score_bounds_validation(self) -> None:
        with pytest.raises(Exception):
            ClinicalScore(
                vignette_id="test004",
                model_id="test-model",
                specialty=MedicalSpecialty.CARDIOLOGY,
                correct_diagnosis=1.5,  # out of bounds
                differential_quality=0.5,
                reasoning_quality=0.5,
                safety_score=0.5,
            )

    def test_dangerous_misses_fields(self) -> None:
        score = ClinicalScore(
            vignette_id="test005",
            model_id="test-model",
            specialty=MedicalSpecialty.EMERGENCY_MEDICINE,
            correct_diagnosis=0.75,
            differential_quality=0.6,
            reasoning_quality=0.7,
            safety_score=0.5,
            dangerous_misses_identified=["Aortic dissection"],
            dangerous_misses_missed=["Tension pneumothorax"],
            evaluator_notes="Good reasoning but missed critical safety item.",
        )
        assert len(score.dangerous_misses_identified) == 1
        assert "Aortic dissection" in score.dangerous_misses_identified
        assert "Tension pneumothorax" in score.dangerous_misses_missed


class TestSpecialtyAggregateScore:
    """Tests for specialty-level aggregation."""

    def _make_scores(self, n: int, dx: float = 0.8) -> list[ClinicalScore]:
        return [
            ClinicalScore(
                vignette_id=f"v{i}",
                model_id="model-a",
                specialty=MedicalSpecialty.CARDIOLOGY,
                correct_diagnosis=dx,
                differential_quality=0.7,
                reasoning_quality=0.6,
                safety_score=0.9,
            )
            for i in range(n)
        ]

    def test_aggregation_with_scores(self) -> None:
        scores = self._make_scores(5)
        agg = SpecialtyAggregateScore.from_scores(
            "model-a", MedicalSpecialty.CARDIOLOGY, scores
        )
        assert agg.n_vignettes == 5
        assert abs(agg.mean_correct_diagnosis - 0.8) < 0.001
        assert abs(agg.mean_differential_quality - 0.7) < 0.001

    def test_aggregation_empty(self) -> None:
        agg = SpecialtyAggregateScore.from_scores(
            "model-a", MedicalSpecialty.CARDIOLOGY, []
        )
        assert agg.n_vignettes == 0
        assert agg.mean_composite == 0.0


class TestModelAggregateScore:
    """Tests for model-level aggregation."""

    def test_aggregation_across_specialties(self) -> None:
        scores = [
            ClinicalScore(
                vignette_id="c1",
                model_id="model-x",
                specialty=MedicalSpecialty.CARDIOLOGY,
                correct_diagnosis=1.0,
                differential_quality=0.8,
                reasoning_quality=0.9,
                safety_score=1.0,
            ),
            ClinicalScore(
                vignette_id="n1",
                model_id="model-x",
                specialty=MedicalSpecialty.NEUROLOGY,
                correct_diagnosis=0.5,
                differential_quality=0.6,
                reasoning_quality=0.5,
                safety_score=0.5,
            ),
        ]
        agg = ModelAggregateScore.from_clinical_scores("model-x", scores)
        assert agg.total_vignettes == 2
        assert len(agg.specialty_scores) == 2
        assert abs(agg.mean_correct_diagnosis - 0.75) < 0.001

    def test_aggregation_empty(self) -> None:
        agg = ModelAggregateScore.from_clinical_scores("model-x", [])
        assert agg.total_vignettes == 0
        assert agg.mean_composite == 0.0
