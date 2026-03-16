# Nidana

**LLM Clinical Reasoning Benchmark across 20 Medical Specialties**

Nidana evaluates large language models on their ability to perform clinical diagnostic reasoning using realistic patient vignettes. Named after the Sanskrit term for "diagnosis" (*nidana*), the benchmark measures four dimensions of clinical competence:

- **Diagnostic accuracy** -- Can the model identify the correct diagnosis?
- **Differential quality** -- Does it generate a comprehensive, well-ordered differential?
- **Reasoning quality** -- Does it synthesize clinical findings with expert-level logic?
- **Safety awareness** -- Does it flag dangerous diagnoses that must not be missed?

## Specialties

Nidana covers 20 medical specialties with 40+ expert-authored clinical vignettes:

Cardiology, Pulmonology, Gastroenterology, Nephrology, Neurology, Endocrinology, Rheumatology, Hematology-Oncology, Infectious Disease, Dermatology, Psychiatry, OB/GYN, Pediatrics, Emergency Medicine, Orthopedics, Urology, Ophthalmology, Otolaryngology (ENT), Allergy & Immunology, General Surgery.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# List available specialties and vignette counts
nidana list-specialties

# List all vignettes
nidana list-vignettes --specialty cardiology

# Run benchmark (requires API key)
export ANTHROPIC_API_KEY="sk-ant-..."
nidana run --provider claude --model claude-sonnet-4-20250514 --max-vignettes 5

# View scores from a results file
nidana score results.json

# Generate leaderboard from multiple result files
nidana leaderboard results_claude.json results_gpt4.json -o leaderboard.json
```

## Scoring

Each vignette is scored on four axes (0.0-1.0):

| Metric | Weight | Description |
|--------|--------|-------------|
| Correct Diagnosis | 40% | Accuracy of the primary diagnosis |
| Differential Quality | 20% | Completeness and ordering of the differential |
| Reasoning Quality | 25% | Depth and correctness of clinical reasoning |
| Safety Score | 15% | Identification of dangerous diagnoses |

The **composite score** is the weighted average of all four metrics.

## Architecture

- **VignetteBank**: 40+ built-in clinical vignettes with patient demographics, vitals, labs, imaging, correct diagnosis, differential, and dangerous misses
- **ModelAdapter**: Unified interface for Claude, OpenAI, and Ollama (local) models
- **ClinicalEvaluator**: LLM-as-judge grading using a structured rubric
- **NidanaBench**: Orchestrates vignette presentation, model querying, and evaluation
- **Leaderboard**: Ranks models overall and by specialty

## Testing

```bash
pytest tests/ -v
```

## Author

Mukunda Katta

## License

Apache 2.0
