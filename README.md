# nidana

**Nidana — LLM Clinical Reasoning Benchmark. Testing AI diagnostic accuracy across 20 medical specialties with 2000+ clinical vignettes.**

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-proprietary-red)

## Install
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from src.core import Nidana
 instance = Nidana()
r = instance.process(input="test")
```

## CLI
```bash
python -m src status
python -m src run --input "data"
```

## API
| Method | Description |
|--------|-------------|
| `process()` | Process |
| `analyze()` | Analyze |
| `transform()` | Transform |
| `validate()` | Validate |
| `export()` | Export |
| `get_stats()` | Get stats |
| `get_stats()` | Get stats |
| `reset()` | Reset |

## Test
```bash
pytest tests/ -v
```

## License
(c) 2026 Officethree Technologies. All Rights Reserved.
