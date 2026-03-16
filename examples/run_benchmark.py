#!/usr/bin/env python3
"""Example: run the Nidana benchmark against a model.

Usage
-----
    # Set your API key
    export ANTHROPIC_API_KEY="sk-ant-..."

    # Run against Claude with 5 vignettes (quick test)
    python examples/run_benchmark.py --provider claude --model claude-sonnet-4-20250514 --max-vignettes 5

    # Run against OpenAI
    export OPENAI_API_KEY="sk-..."
    python examples/run_benchmark.py --provider openai --model gpt-4o --specialty cardiology

    # Run against local Ollama
    python examples/run_benchmark.py --provider ollama --model llama3 --max-vignettes 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path for local development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console

from nidana.benchmark import NidanaBench
from nidana.leaderboard import Leaderboard
from nidana.models import ModelConfig
from nidana.report import NidanaReporter
from nidana.vignettes.specialties import MedicalSpecialty


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Nidana clinical reasoning benchmark")
    parser.add_argument("--provider", required=True, choices=["claude", "openai", "ollama"])
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--judge-provider", default="claude", help="LLM judge provider")
    parser.add_argument("--judge-model", default="claude-sonnet-4-20250514", help="LLM judge model")
    parser.add_argument("--specialty", action="append", help="Limit to specialty (repeatable)")
    parser.add_argument("--max-vignettes", type=int, default=None, help="Max vignettes to run")
    parser.add_argument("--output", "-o", default="results/benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    console = Console()

    # Build adapters
    model_adapter = ModelConfig(provider=args.provider, model=args.model).to_adapter()
    judge_adapter = ModelConfig(provider=args.judge_provider, model=args.judge_model).to_adapter()

    # Parse specialties
    specialties = None
    if args.specialty:
        specialties = [MedicalSpecialty(s) for s in args.specialty]

    # Run benchmark
    bench = NidanaBench(judge=judge_adapter, console=console)
    result = bench.run(
        model=model_adapter,
        specialties=specialties,
        max_vignettes=args.max_vignettes,
    )

    # Report
    reporter = NidanaReporter(console=console)
    reporter.print_result(result)

    # Leaderboard (single model)
    lb = Leaderboard.from_results([result])
    reporter.print_leaderboard(lb)

    # Export
    reporter.export_json([result], args.output)

    console.print("\n[bold green]Benchmark complete.[/bold green]")


if __name__ == "__main__":
    main()
