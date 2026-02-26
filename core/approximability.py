"""
Pillar 2: Approximability

To prevent exponential runtime in search spaces, learning systems must be able
to define heuristic distances between abstract states to approximate the optimal logic.
"""
from abc import ABC, abstractmethod
from typing import Any

class Heuristic(ABC):
    """
    Evaluates how close a candidate state or program is to the desired outcome.
    Allows for guided search over combinatorial explosions.
    """
    @abstractmethod
    def evaluate(self, candidate: Any, target: Any) -> float:
        """Returns a scalar score. Higher or lower depending on maximization formulation."""
        pass
