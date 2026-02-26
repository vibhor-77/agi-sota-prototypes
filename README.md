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

## Running the Agents
The architecture is fully modularized. Run the entry point scripts for each domain:

```bash
# Run the Spatial Composition Engine (ARC)
python -m domains.arc.run

# Run the Deep Symbolic Pathway Planner (Zork)
python -m domains.zork.run
```
