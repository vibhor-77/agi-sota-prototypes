"""
Pillar 3: Abstraction and Composability

The agent must be able to decompose complex tasks into fundamental primitives,
and compose them dynamically to invent novel solutions.
"""
from abc import ABC, abstractmethod
from typing import Any, List

class StateRepresentation(ABC):
    """
    Extracts semantics from raw observations (e.g. text parsing, visual grouping)
    to form abstract categorical concepts.
    """
    @abstractmethod
    def parse(self, raw_observation: Any) -> Any:
        """Translates a raw environment observation into a structured, abstracted graph or format."""
        pass

class ActionGrammar(ABC):
    """
    Defines the set of fundamental operations that can be composed to solve tasks.
    """
    @property
    @abstractmethod
    def primitives(self) -> List[Any]:
        """Returns the primitive functions or operators available."""
        pass
        
    @abstractmethod
    def compose(self, *args) -> Any:
        """Dynamically builds larger operations from the primitives."""
        pass
