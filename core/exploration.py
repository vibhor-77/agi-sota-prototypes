"""
Pillar 4: Exploration

The agent must systematically search the abstract state space, discovering logical topologies,
using the grammar (Composability) guided by the heuristic distance (Approximability).
"""
from abc import ABC, abstractmethod
from typing import Any

class SearchAlgorithm(ABC):
    """
    Systematic discovery and pathfinding over dynamic environmental graphs or program spaces.
    """
    @abstractmethod
    def search(self, start_state: Any, target: Any, **kwargs) -> Any:
        """Returns a discovered optimal path or synthesized program."""
        pass
