# nidana

Python scaffold for an LLM clinical reasoning benchmark concept. Work in progress.

## What's Here

A Nidana class with stub methods for processing, analyzing, and transforming. These methods increment a counter and return `{ "ok": True }` without performing real operations.

- src/core.py - Main Nidana class with placeholder methods
- src/__main__.py - CLI entry point
- src/nidana/ - Package directory with module files:
  - benchmark.py, evaluator.py, scorer.py - Evaluation stubs
  - models.py, leaderboard.py, report.py - Data/reporting stubs
  - cli.py - CLI module
  - vignettes/ - Clinical vignettes directory
- examples/ - Example scripts
- tests/ - Test directory

## Tech Stack

- Python
- Standard library only (time, logging, json, typing)

## Status

AI-generated scaffold. The class methods are stubs that do not perform actual clinical reasoning evaluation, LLM benchmarking, or medical vignette processing. No clinical data, medical knowledge base, or LLM integration exists despite the description claiming 2000+ clinical vignettes across 20 specialties.

## Setup

```bash
pip install -r requirements.txt
python -m src.nidana
```
