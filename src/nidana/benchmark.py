"""NidanaBench -- orchestration of benchmark runs across models."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from nidana.evaluator import ClinicalEvaluator
from nidana.models import ModelAdapter
from nidana.scorer import ClinicalScore, ModelAggregateScore
from nidana.vignettes.generator import ClinicalVignette, VignetteBank
from nidana.vignettes.specialties import MedicalSpecialty

_SYSTEM_PROMPT = (
    "You are a board-certified physician. Given a clinical vignette, provide:\n"
    "1. Your most likely diagnosis\n"
    "2. A ranked differential diagnosis list (up to 5 diagnoses)\n"
    "3. Your clinical reasoning explaining how you arrived at your diagnosis\n"
    "4. Any critical diagnoses that must not be missed in this presentation\n\n"
    "Be thorough, precise, and demonstrate expert-level clinical reasoning."
)


class BenchmarkResult(BaseModel):
    """Complete result of a benchmark run for one model."""

    model_id: str
    run_timestamp: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    scores: list[ClinicalScore]
    aggregate: ModelAggregateScore
    raw_responses: dict[str, str] = Field(
        default_factory=dict,
        description="Map of vignette_id -> raw model output",
    )


class NidanaBench:
    """Main benchmark orchestrator.

    Runs clinical vignettes against one or more models, collects responses,
    evaluates them with an LLM judge, and produces aggregate scores.
    """

    def __init__(
        self,
        judge: ModelAdapter,
        vignette_bank: Optional[VignetteBank] = None,
        console: Optional[Console] = None,
    ) -> None:
        self._judge = judge
        self._bank = vignette_bank or VignetteBank()
        self._evaluator = ClinicalEvaluator(judge=judge)
        self._console = console or Console()

    def run(
        self,
        model: ModelAdapter,
        specialties: Optional[list[MedicalSpecialty]] = None,
        max_vignettes: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run the full benchmark pipeline for a single model.

        Parameters
        ----------
        model:
            The model adapter to benchmark.
        specialties:
            Limit to specific specialties. ``None`` runs all available.
        max_vignettes:
            Cap the total number of vignettes (useful for quick testing).
        """
        # Select vignettes
        if specialties:
            vignettes: list[ClinicalVignette] = []
            for sp in specialties:
                vignettes.extend(self._bank.by_specialty(sp))
        else:
            vignettes = self._bank.all

        if max_vignettes is not None:
            vignettes = vignettes[:max_vignettes]

        self._console.print(
            f"\n[bold cyan]Nidana Benchmark[/bold cyan] -- "
            f"model=[bold]{model.model_id}[/bold]  "
            f"vignettes={len(vignettes)}  "
            f"specialties={len({v.specialty for v in vignettes})}"
        )

        scores: list[ClinicalScore] = []
        raw_responses: dict[str, str] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Running vignettes...", total=len(vignettes))
            for vignette in vignettes:
                progress.update(
                    task,
                    description=f"[{vignette.specialty.display_name}] {vignette.chief_complaint[:50]}...",
                )
                # Step 1: get model response
                prompt = vignette.to_prompt()
                response = model.generate(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                raw_responses[vignette.id] = response.raw_text

                # Step 2: evaluate with judge
                score = self._evaluator.evaluate(
                    vignette=vignette,
                    model_output=response.raw_text,
                    model_id=model.model_id,
                )
                scores.append(score)
                progress.advance(task)

        aggregate = ModelAggregateScore.from_clinical_scores(model.model_id, scores)

        return BenchmarkResult(
            model_id=model.model_id,
            scores=scores,
            aggregate=aggregate,
            raw_responses=raw_responses,
        )

    def run_multiple(
        self,
        models: list[ModelAdapter],
        specialties: Optional[list[MedicalSpecialty]] = None,
        max_vignettes: Optional[int] = None,
    ) -> list[BenchmarkResult]:
        """Run the benchmark across multiple models sequentially."""
        results: list[BenchmarkResult] = []
        for model in models:
            result = self.run(model, specialties=specialties, max_vignettes=max_vignettes)
            results.append(result)
        return results
