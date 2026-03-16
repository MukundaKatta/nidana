"""Rich terminal reports and JSON export for Nidana benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nidana.benchmark import BenchmarkResult
from nidana.leaderboard import Leaderboard
from nidana.scorer import ClinicalScore


class NidanaReporter:
    """Generates rich terminal output and JSON exports for benchmark results."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self._console = console or Console()

    # ------------------------------------------------------------------
    # Single-model result report
    # ------------------------------------------------------------------

    def print_result(self, result: BenchmarkResult) -> None:
        """Print a detailed result report for a single model."""
        c = self._console
        agg = result.aggregate

        c.print()
        c.print(
            Panel(
                f"[bold]Model:[/bold] {result.model_id}\n"
                f"[bold]Vignettes:[/bold] {agg.total_vignettes}\n"
                f"[bold]Timestamp:[/bold] {result.run_timestamp}",
                title="[bold cyan]Nidana Benchmark Results[/bold cyan]",
                border_style="cyan",
            )
        )

        # Overall scores
        overall_table = Table(title="Overall Scores", show_header=True, header_style="bold magenta")
        overall_table.add_column("Metric", style="bold")
        overall_table.add_column("Score", justify="right")
        overall_table.add_row("Correct Diagnosis", f"{agg.mean_correct_diagnosis:.3f}")
        overall_table.add_row("Differential Quality", f"{agg.mean_differential_quality:.3f}")
        overall_table.add_row("Reasoning Quality", f"{agg.mean_reasoning_quality:.3f}")
        overall_table.add_row("Safety Score", f"{agg.mean_safety_score:.3f}")
        overall_table.add_row(
            Text("Composite", style="bold green"),
            Text(f"{agg.mean_composite:.3f}", style="bold green"),
        )
        c.print(overall_table)

        # Per-specialty breakdown
        if agg.specialty_scores:
            sp_table = Table(
                title="Scores by Specialty",
                show_header=True,
                header_style="bold blue",
            )
            sp_table.add_column("Specialty", style="bold")
            sp_table.add_column("N", justify="right")
            sp_table.add_column("Diagnosis", justify="right")
            sp_table.add_column("Differential", justify="right")
            sp_table.add_column("Reasoning", justify="right")
            sp_table.add_column("Safety", justify="right")
            sp_table.add_column("Composite", justify="right", style="green")

            for sp in agg.specialty_scores:
                sp_table.add_row(
                    sp.specialty.display_name,
                    str(sp.n_vignettes),
                    f"{sp.mean_correct_diagnosis:.3f}",
                    f"{sp.mean_differential_quality:.3f}",
                    f"{sp.mean_reasoning_quality:.3f}",
                    f"{sp.mean_safety_score:.3f}",
                    f"{sp.mean_composite:.3f}",
                )
            c.print(sp_table)

        # Individual vignette scores
        self._print_vignette_details(result.scores)

    def _print_vignette_details(self, scores: list[ClinicalScore]) -> None:
        """Print per-vignette score details."""
        table = Table(
            title="Individual Vignette Scores",
            show_header=True,
            header_style="bold yellow",
        )
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Specialty")
        table.add_column("Dx", justify="right")
        table.add_column("Diff", justify="right")
        table.add_column("Reason", justify="right")
        table.add_column("Safety", justify="right")
        table.add_column("Comp", justify="right", style="green")

        for s in scores:
            table.add_row(
                s.vignette_id[:10],
                s.specialty.display_name[:18],
                f"{s.correct_diagnosis:.2f}",
                f"{s.differential_quality:.2f}",
                f"{s.reasoning_quality:.2f}",
                f"{s.safety_score:.2f}",
                f"{s.composite_score:.2f}",
            )
        self._console.print(table)

    # ------------------------------------------------------------------
    # Leaderboard report
    # ------------------------------------------------------------------

    def print_leaderboard(self, leaderboard: Leaderboard) -> None:
        """Print the overall leaderboard."""
        c = self._console
        c.print()
        table = Table(
            title="[bold cyan]Nidana Leaderboard[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Rank", justify="right", style="bold")
        table.add_column("Model", style="bold")
        table.add_column("Vignettes", justify="right")
        table.add_column("Diagnosis", justify="right")
        table.add_column("Differential", justify="right")
        table.add_column("Reasoning", justify="right")
        table.add_column("Safety", justify="right")
        table.add_column("Composite", justify="right", style="bold green")

        for entry in leaderboard.overall:
            table.add_row(
                str(entry.rank),
                entry.model_id,
                str(entry.total_vignettes),
                f"{entry.mean_correct_diagnosis:.3f}",
                f"{entry.mean_differential_quality:.3f}",
                f"{entry.mean_reasoning_quality:.3f}",
                f"{entry.mean_safety_score:.3f}",
                f"{entry.mean_composite:.3f}",
            )
        c.print(table)

    def print_specialty_leaderboard(
        self,
        leaderboard: Leaderboard,
        specialty: Optional[str] = None,
    ) -> None:
        """Print specialty-specific leaderboards."""
        c = self._console
        specialties = [specialty] if specialty else sorted(leaderboard.by_specialty.keys())

        for sp_key in specialties:
            entries = leaderboard.by_specialty.get(sp_key, [])
            if not entries:
                continue
            display = entries[0].specialty.display_name if entries else sp_key
            table = Table(
                title=f"[bold blue]{display}[/bold blue]",
                show_header=True,
                header_style="bold",
            )
            table.add_column("Rank", justify="right")
            table.add_column("Model")
            table.add_column("N", justify="right")
            table.add_column("Diagnosis", justify="right")
            table.add_column("Safety", justify="right")
            table.add_column("Composite", justify="right", style="green")

            for e in entries:
                table.add_row(
                    str(e.rank),
                    e.model_id,
                    str(e.n_vignettes),
                    f"{e.mean_correct_diagnosis:.3f}",
                    f"{e.mean_safety_score:.3f}",
                    f"{e.mean_composite:.3f}",
                )
            c.print(table)

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def export_json(
        self,
        results: list[BenchmarkResult],
        output_path: str | Path,
    ) -> Path:
        """Export results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "benchmark": "nidana",
            "version": "0.1.0",
            "results": [r.model_dump(mode="json") for r in results],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        self._console.print(f"\n[green]Results exported to {path}[/green]")
        return path

    def export_leaderboard_json(
        self,
        leaderboard: Leaderboard,
        output_path: str | Path,
    ) -> Path:
        """Export leaderboard to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(leaderboard.model_dump(mode="json"), indent=2, default=str))
        self._console.print(f"\n[green]Leaderboard exported to {path}[/green]")
        return path
