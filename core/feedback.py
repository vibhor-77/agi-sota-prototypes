"""
Pillar 1: Feedback Loops

In general learning, the agent must be able to interact with an environment,
execute actions, and receive an observation (feedback) of the resulting state.
"""
from abc import ABC, abstractmethod
from typing import Any

class Environment(ABC):
    """
    Abstract representation of an interactive environment.
    Provides the feedback loop where actions cause observable state changes.
    """
    @abstractmethod
    def get_observation(self) -> Any:
        """Returns the current observable state of the environment."""
        pass
        
    @abstractmethod
    def execute_action(self, action: Any) -> Any:
        """Executes an action and returns the next observation/feedback."""
        pass
