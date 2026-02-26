# AGI SOTA Prototypes: The 4 Pillars of General Learning

This repository contains Minimum Viable Product (MVP) implementations of advanced State-of-the-Art (SOTA) Artificial General Intelligence (AGI) concepts across two distinct domains: Continuous spatial search (ARC) and Deep dependency discrete pathways (Zork).

The codebase is explicitly architected around the **4 Core Pillars of General Learning**:

1. **Feedback Loops (`core.feedback.Environment`)**: The ability to execute actions in a dynamic world and observe the resulting state changes.
2. **Approximability (`core.approximability.Heuristic`)**: The ability to evaluate mathematical distance between abstract states to guide search and avoid exponential complexity.
3. **Abstraction & Composability (`core.abstraction.StateRepresentation`, `ActionGrammar`)**: The ability to parse raw semantics into structured concepts and dynamically compose primitive functions into novel programs.
4. **Exploration (`core.exploration.SearchAlgorithm`)**: The systematic topological traversal of combinatorial state and program spaces to deduce logic.

## 1. Domain: ARC-AGI-2 SOTA (`domains/arc/`)
A spatial reasoning agent capable of program synthesis via Beam Search. 
- **Feedback**: Manipulates 2D `numpy` grids.
- **Abstraction**: Features an enriched Domain Specific Language (DSL) including `rotate90`, `mirror`, and topological bounding box scaling.
- **Approximability**: Uses `Pixel Edit Distance` heuristics to efficiently guide the beam search towards functional composition.
- **Exploration**: Employs Beam Search to prioritize and mutate the top-scoring AST programs.
- *Result*: Demonstrates zero-shot generalization to unseen complex geometrical transformations.

## 2. Domain: Zork Deep RL SOTA (`domains/zork/`)
A deep-horizon symbolic reasoning agent operating in a multi-state stochastic text dependency environment.
- **Feedback**: Multi-step lock-and-key design including lighting (Lamp), combat (Sword), and environmental gating (Keys & Doors).
- **Abstraction**: Uses few-shot entity extraction to map Natural Language commands (`"attack troll with sword"`) into semantic JSON graphs.
- **Approximability / Exploration**: Employs deterministic BFS replay to deduce win parameters through partial observability state-aliasing prevention.
- *Result*: Dynamically computes a perfect >8 step deep execution pathway without hardcoded logic.

## Quick Start & Setup

The project relies on very light dependencies (numpy/scipy) for the abstract computation.

```bash
# 1. Clone the repository
git clone https://github.com/vibhor-77/agi-sota-prototypes.git
cd agi-sota-prototypes

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run the blazing fast unit tests
python -m unittest discover tests/
```

## Running the Agents (`main.py`)
The architecture is fully modularized and exposed via a unified CLI. 

Both agents dynamically support scaling difficulties from `Level 1` (linear task mapping) to `Level 3` (deep structural dependencies requiring full search topologies).

### 1. Interactive Demonstrations
Watch the agent explore the environment, build its abstraction graph, and deduce the logical execution sequence step-by-step.
```bash
python main.py interactive --domain zork --level 3
```
```bash
python main.py interactive --domain arc --level 2
```

### 2. High-Performance Benchmarks
Run the agents headless to evaluate the true speed and generalized power of the 4 Pillars computation.
```bash
python main.py benchmark --domain arc --level 3 --trials 10
```
```bash
python main.py benchmark --domain zork --level 3 --trials 10
```
