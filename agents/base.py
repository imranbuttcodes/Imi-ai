# ==============================================================
# agents/base.py — Abstract Base Class for All Agents
# ==============================================================
# Every agent in the system MUST inherit from BaseAgent.
# This enforces the contract:
#   - Define: name, description, capabilities
#   - Implement: run(imi_state) -> dict
#   - Registration happens automatically on __init__
#
# Usage (in any agent):
#   from agents.base import BaseAgent
#
#   class KnowledgeAgent(BaseAgent):
#       name         = "knowledge"
#       description  = "Searches your private documents."
#       capabilities = ["rag", "document-qa"]
#
#       def run(self, imi_state):
#           ...
# ==============================================================

from abc import ABC, abstractmethod
from core.registry import registry
from core.state import ImiState


class BaseAgent(ABC):
    """
    Abstract base class that every Imi AI agent must inherit from.

    Responsibilities:
    - Enforces that every agent has a run() method (via ABC).
    - Auto-registers the agent in the global registry on instantiation.
    - Provides a consistent interface for the master graph to call.
    """

    # ── Subclasses MUST define these three class attributes ────
    name:         str       # e.g., "knowledge"
    description:  str       # e.g., "Searches your private documents"
    capabilities: list[str] # e.g., ["rag", "document-qa"]

    def __init__(self):
        """
        Called automatically when an agent is instantiated.
        Registers this agent into the global registry.
        """
        self._register()

    def _register(self) -> None:
        """Registers this agent into the global singleton registry."""
        registry.register(
            name=self.name,
            agent=self,
            capabilities=self.capabilities,
            description=self.description,
        )

    @abstractmethod
    def run(self, imi_state: ImiState) -> dict:
        """
        Main execution method for the agent.
        Takes the full master graph state, executes specialist tasks,
        and returns a partial state update dictionary.

        Args:
            imi_state: The current master graph state.

        Returns:
            A dict with the fields this agent updated.
            At minimum: {"agent_outputs": {"<name>": "<answer>"}}
        """
        ...
