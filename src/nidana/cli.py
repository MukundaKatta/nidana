"""Click CLI for the Nidana benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from nidana.benchmark import BenchmarkResult, NidanaBench
from nidana.leaderboard import Leaderboard
from nidana.models import ClaudeAdapter, ModelConfig, OllamaAdapter, OpenAIAdapter
from nidana.report import NidanaReporter
from nidana.vignettes.generator import VignetteBank
from nidana.vignettes.specialties import MedicalSpecialty

console = Console()


@click.group()
@click.version_option(package_name="nidana-bench")
def cli() -> None:
    """Nidana -- LLM Clinical Reasoning Benchmark across 20 Medical Specialties."""


# ---------------------------------------------------------------------------
# nidana run
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["claude", "openai", "ollama"], case_sensitive=False),
    required=True,
    help="LLM provider to benchmark.",
)
@click.option("--model", required=True, help="Model identifier (e.g. claude-sonnet-4-20250514, gpt-4o, llama3).")
@click.option("--judge-provider", default="claude", help="Provider for the LLM judge (default: claude).")
@click.option("--judge-model", default="claude-sonnet-4-20250514", help="Model for the LLM judge.")
@click.option(
    "--specialty",
    multiple=True,
    help="Limit to specific specialties (can be repeated). Use enum values like 'cardiology'.",
)
@click.option("--max-vignettes", type=int, default=None, help="Maximum number of vignettes to run.")
@click.option("--output", "-o", type=click.Path(), default=None, help="JSON output path for results.")
def run(
    provider: str,
    model: str,
    judge_provider: str,
    judge_model: str,
    specialty: tuple[str, ...],
    max_vignettes: Optional[int],
    output: Optional[str],
) -> None:
    """Run the Nidana benchmark against a model."""
    # Build model adapter
    model_adapter = ModelConfig(provider=provider, model=model).to_adapter()

    # Build judge adapter
    judge_adapter = ModelConfig(provider=judge_provider, model=judge_model).to_adapter()

    # Parse specialties
    selected_specialties: Optional[list[MedicalSpecialty]] = None
    if specialty:
        selected_specialties = []
        for s in specialty:
            try:
                selected_specialties.append(MedicalSpecialty(s))
            except ValueError:
                console.print(f"[red]Unknown specialty: {s}[/red]")
                console.print(f"Available: {', '.join(sp.value for sp in MedicalSpecialty)}")
                raise SystemExit(1)

    # Run benchmark
    bench = NidanaBench(judge=judge_adapter, console=console)
    result = bench.run(
        model=model_adapter,
        specialties=selected_specialties,
        max_vignettes=max_vignettes,
    )

    # Display results
    reporter = NidanaReporter(console=console)
    reporter.print_result(result)

    # Export if requested
    if output:
        reporter.export_json([result], output)


# ---------------------------------------------------------------------------
# nidana score
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
def score(results_file: str) -> None:
    """Display detailed scores from a results JSON file."""
    path = Path(results_file)
    data = json.loads(path.read_text())

    results = [BenchmarkResult.model_validate(r) for r in data.get("results", [])]
    if not results:
        console.print("[red]No results found in file.[/red]")
        raise SystemExit(1)

    reporter = NidanaReporter(console=console)
    for result in results:
        reporter.print_result(result)


# ---------------------------------------------------------------------------
# nidana leaderboard
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("results_files", nargs=-1, type=click.Path(exists=True))
@click.option("--specialty", default=None, help="Show leaderboard for a specific specialty.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Export leaderboard to JSON.")
def leaderboard(
    results_files: tuple[str, ...],
    specialty: Optional[str],
    output: Optional[str],
) -> None:
    """Generate a leaderboard from one or more result files."""
    if not results_files:
        console.print("[red]Provide at least one results JSON file.[/red]")
        raise SystemExit(1)

    all_results: list[BenchmarkResult] = []
    for rf in results_files:
        data = json.loads(Path(rf).read_text())
        all_results.extend(
            BenchmarkResult.model_validate(r) for r in data.get("results", [])
        )

    if not all_results:
        console.print("[red]No results found in provided files.[/red]")
        raise SystemExit(1)

    lb = Leaderboard.from_results(all_results)
    reporter = NidanaReporter(console=console)
    reporter.print_leaderboard(lb)

    if specialty:
        reporter.print_specialty_leaderboard(lb, specialty=specialty)
    else:
        reporter.print_specialty_leaderboard(lb)

    if output:
        reporter.export_leaderboard_json(lb, output)


# ---------------------------------------------------------------------------
# nidana list-specialties
# ---------------------------------------------------------------------------

@cli.command("list-specialties")
def list_specialties() -> None:
    """List all 20 medical specialties and vignette counts."""
    from rich.table import Table

    bank = VignetteBank()
    table = Table(title="Nidana Medical Specialties", show_header=True, header_style="bold cyan")
    table.add_column("Specialty", style="bold")
    table.add_column("Enum Value")
    table.add_column("Vignettes", justify="right")
    table.add_column("Description", max_width=60)

    for sp in MedicalSpecialty:
        count = len(bank.by_specialty(sp))
        style = "green" if count > 0 else "dim"
        table.add_row(
            sp.display_name,
            sp.value,
            str(count),
            sp.description[:60] + ("..." if len(sp.description) > 60 else ""),
            style=style,
        )
    console.print(table)
    console.print(f"\n[bold]Total vignettes:[/bold] {len(bank)}")


# ---------------------------------------------------------------------------
# nidana list-vignettes
# ---------------------------------------------------------------------------

@cli.command("list-vignettes")
@click.option("--specialty", default=None, help="Filter by specialty.")
def list_vignettes(specialty: Optional[str]) -> None:
    """List all built-in clinical vignettes."""
    from rich.table import Table

    bank = VignetteBank()
    vignettes = bank.all
    if specialty:
        try:
            sp = MedicalSpecialty(specialty)
            vignettes = bank.by_specialty(sp)
        except ValueError:
            console.print(f"[red]Unknown specialty: {specialty}[/red]")
            raise SystemExit(1)

    table = Table(title="Clinical Vignettes", show_header=True, header_style="bold")
    table.add_column("ID", max_width=12, style="dim")
    table.add_column("Specialty")
    table.add_column("Difficulty")
    table.add_column("Age/Sex", justify="center")
    table.add_column("Chief Complaint", max_width=50)
    table.add_column("Diagnosis", max_width=40)

    for v in vignettes:
        diff_style = {"easy": "green", "moderate": "yellow", "hard": "red"}.get(v.difficulty, "")
        table.add_row(
            v.id[:10],
            v.specialty.display_name[:18],
            f"[{diff_style}]{v.difficulty}[/{diff_style}]",
            f"{v.patient_age}{v.patient_sex}",
            v.chief_complaint[:50],
            v.correct_diagnosis[:40],
        )
    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(vignettes)} vignettes")


if __name__ == "__main__":
    cli()
