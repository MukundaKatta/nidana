"""Leaderboard generation and ranking for the Nidana benchmark."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from nidana.benchmark import BenchmarkResult
from nidana.scorer import ModelAggregateScore, SpecialtyAggregateScore
from nidana.vignettes.specialties import MedicalSpecialty


class LeaderboardEntry(BaseModel):
    """A single row on the leaderboard."""

    rank: int
    model_id: str
    total_vignettes: int
    mean_composite: float
    mean_correct_diagnosis: float
    mean_differential_quality: float
    mean_reasoning_quality: float
    mean_safety_score: float


class SpecialtyLeaderboardEntry(BaseModel):
    """A single row on a specialty-specific leaderboard."""

    rank: int
    model_id: str
    specialty: MedicalSpecialty
    n_vignettes: int
    mean_composite: float
    mean_correct_diagnosis: float
    mean_safety_score: float


class Leaderboard(BaseModel):
    """Complete leaderboard with overall and per-specialty rankings."""

    overall: list[LeaderboardEntry]
    by_specialty: dict[str, list[SpecialtyLeaderboardEntry]]

    @staticmethod
    def from_results(results: list[BenchmarkResult]) -> "Leaderboard":
        """Build a leaderboard from one or more benchmark results."""
        # Overall ranking
        aggregates: list[ModelAggregateScore] = [r.aggregate for r in results]
        aggregates.sort(key=lambda a: a.mean_composite, reverse=True)

        overall = [
            LeaderboardEntry(
                rank=i + 1,
                model_id=agg.model_id,
                total_vignettes=agg.total_vignettes,
                mean_composite=round(agg.mean_composite, 4),
                mean_correct_diagnosis=round(agg.mean_correct_diagnosis, 4),
                mean_differential_quality=round(agg.mean_differential_quality, 4),
                mean_reasoning_quality=round(agg.mean_reasoning_quality, 4),
                mean_safety_score=round(agg.mean_safety_score, 4),
            )
            for i, agg in enumerate(aggregates)
        ]

        # Per-specialty ranking
        specialty_map: dict[str, list[SpecialtyAggregateScore]] = {}
        for agg in aggregates:
            for sp_score in agg.specialty_scores:
                key = sp_score.specialty.value
                specialty_map.setdefault(key, []).append(sp_score)

        by_specialty: dict[str, list[SpecialtyLeaderboardEntry]] = {}
        for sp_key, sp_scores in sorted(specialty_map.items()):
            sp_scores.sort(key=lambda s: s.mean_composite, reverse=True)
            by_specialty[sp_key] = [
                SpecialtyLeaderboardEntry(
                    rank=i + 1,
                    model_id=s.model_id,
                    specialty=s.specialty,
                    n_vignettes=s.n_vignettes,
                    mean_composite=round(s.mean_composite, 4),
                    mean_correct_diagnosis=round(s.mean_correct_diagnosis, 4),
                    mean_safety_score=round(s.mean_safety_score, 4),
                )
                for i, s in enumerate(sp_scores)
            ]

        return Leaderboard(overall=overall, by_specialty=by_specialty)
