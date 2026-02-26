# AGI SOTA Prototypes

This repository contains Minimum Viable Product (MVP) implementations of advanced State-of-the-Art (SOTA) Artificial General Intelligence (AGI) concepts across two distinct domains: Continuous spatial search (ARC) and Deep dependency discrete pathways (Zork).

## 1. ARC-AGI-2 SOTA (`arc_sota.py`)
A spatial reasoning agent capable of program synthesis via Beam Search. 
- Features an enriched Domain Specific Language (DSL) including `rotate90`, `mirror`, and topological bounding box scaling.
- Uses `Pixel Edit Distance` heuristics to efficiently guide the beam search towards functional composition.
- Demonstrates zero-shot generalization to unseen complex geometrical transformations.

## 2. Zork Deep RL SOTA (`zork_sota.py`)
A deep-horizon symbolic reasoning agent operating in a multi-state stochastic text dependency environment.
- **Deep Dependencies**: Multi-step lock-and-key design including lighting (Lamp), combat (Sword), and environmental gating (Keys & Doors).
- **Semantic LLM-style Parsing**: Uses few-shot entity extraction to map Natural Language commands (`"attack troll with sword"`) into semantic JSON graphs.
- **BFS State-Space Graph Search**: Employs deterministic BFS replay to deduce win parameters through partial observability state-aliasing prevention.
- Achieves perfect generalization in reaching the goal state without hardcoded pathways.

## Running
Both agents are self-contained zero-dependency files (barring `numpy`). Run them purely as python scripts.

```bash
python arc_sota.py
python zork_sota.py
```
