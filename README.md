# AGI SOTA Prototypes: The 4 Pillars of General Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 27 pass](https://img.shields.io/badge/tests-27%20pass-brightgreen)](#)

MVP implementations of State-of-the-Art AGI concepts, validated across two distinct domains: spatial program synthesis (**ARC-AGI**) and deep symbolic reasoning (**Zork**).

## Architecture: The 4 Core Pillars 🏛️

All agents inherit from the abstract interfaces defined in `core/`:

| Pillar | Interface | Purpose |
|:--|:--|:--|
| **Feedback Loops** | `core.feedback.Environment` | Execute actions, observe state changes |
| **Approximability** | `core.approximability.Heuristic` | Evaluate distance between states to guide search |
| **Abstraction** | `core.abstraction.ActionGrammar` | Compose primitives into novel programs |
| **Exploration** | `core.exploration.SearchAlgorithm` | Traverse combinatorial state/program spaces |

---

## Domains 🌍

### ARC-AGI (`domains/arc/`)
Program synthesis over the [Official ARC-AGI Training Dataset](https://github.com/fchollet/ARC-AGI) (400 tasks).

- **DSL**: 19 primitives — `rotate90`, `mirror_x/y`, `transpose`, `crop`, `pad`, `fill_box`, `replace_color`, `paint`, `tile`, `overlay`, `scale_up`, `stack_v`, `stack_h`, `largest_object`, `count_color`, `most_common_color`
- **Search**: Evolutionary beam search with mutation, crossover, and heuristic-guided selection
- **Heuristic**: Pixel edit distance (0.0 = exact match)

### Zork (`domains/zork/`)
Deep symbolic exploration of [Infocom Zork I](https://en.wikipedia.org/wiki/Zork) (350 total points) using Microsoft's `jericho` engine.

- **Search**: A* best-first with composite reward shaping: `score×10 + inventory×2 + rooms×1`
- **Macro-Actions**: `take all`, `look`, `open all`, `inventory`, `drop all` injected alongside Jericho's atomic actions
- **Abstraction**: Few-shot NLP extraction mapping text to semantic JSON
- **State**: Byte-level hashing for deduplication across thousands of game states

---

## Quick Start 🚀

```bash
# 1. Clone
git clone https://github.com/vibhor-77/agi-sota-prototypes.git
cd agi-sota-prototypes

# 2. Install dependencies
pip install -r requirements.txt

# 3. Clone the Official ARC Dataset
git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI
```

### Dependencies

| Package | Purpose |
|:--|:--|
| `numpy`, `scipy` | Numerical computation |
| `jericho` | Zork I game engine (bundles SpaCy internally) |
| `pytest`, `pytest-xdist` | Parallel test execution |

---

## Running Tests 🧪

```bash
# Sequential (standard unittest)
python -m unittest discover -s tests -t .

# Parallel across all CPU cores (recommended, 2-3× faster)
python -m pytest tests/ -n auto -q
```

> Parallel tests require `pytest` + `pytest-xdist` (included in requirements.txt).

---

## CLI Usage 🖥️

### Interactive Mode
```bash
python main.py interactive --domain zork --level 2
python main.py interactive --domain arc --level 3
```

### Benchmark Mode
```bash
# Basic benchmark
python main.py benchmark --domain arc --level 3 --trials 10

# Parallel with configurable workers
python main.py benchmark --domain zork --level 2 --trials 10 --workers 5

# High-compute ARC: wider beam + more generations
python main.py benchmark --domain arc --level 3 --trials 50 --beam-width 200 --max-gens 50

# High-compute Zork: larger state budget
python main.py benchmark --domain zork --level 2 --trials 10 --budget 2000

# Verbose mode: detailed per-action logging for debugging
python main.py benchmark --domain zork --level 2 --trials 1 --verbose
python main.py benchmark --domain arc --level 3 --trials 1 --verbose
```

### CLI Reference

| Flag | Domain | Default | Description |
|:--|:--|:--|:--|
| `--level` | Both | 3 | Difficulty (1=easy, 2=medium, 3=hard) |
| `--trials` | Both | 5 | Number of benchmark trials |
| `--workers` | Both | CPU count | Parallel worker threads/processes |
| `--budget` | Zork | 1000 (L2) | Max states to expand per trial |
| `--beam-width` | ARC | 100 | Beam width for evolutionary search |
| `--max-gens` | ARC | 30 | Max generations for evolution |
| `--verbose` / `-v` | Both | off | Detailed logging for debugging |

---

## Project Structure

```
agi-sota-prototypes/
├── core/                    # Abstract interfaces (4 Pillars)
│   ├── feedback.py          # Environment base class
│   ├── approximability.py   # Heuristic base class
│   ├── abstraction.py       # StateRepresentation, ActionGrammar
│   └── exploration.py       # SearchAlgorithm base class
├── domains/
│   ├── arc/                 # ARC-AGI domain
│   │   ├── dsl.py           # 19-primitive DSL + AST nodes + evolutionary operators
│   │   ├── env.py           # Grid, BoundingBox, task loading
│   │   ├── heuristics.py    # Pixel edit distance
│   │   └── search.py        # Evolutionary beam search (parallel)
│   └── zork/                # Zork domain
│       ├── env.py           # Jericho wrapper + macro-actions + inventory API
│       ├── agent.py         # A* reward-shaped exploration
│       └── semantics.py     # NLP semantic parser
├── tests/                   # 34 unit + integration tests
├── data/                    # ARC-AGI dataset + Zork ROM
├── main.py                  # Unified CLI
└── requirements.txt
```
