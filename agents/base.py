# ==============================================================
# agents/base.py — Abstract Base Class for All Agents
# ==============================================================
# Every agent in the system MUST inherit from BaseAgent.
# This enforces the contract:
#   - Define: name, description, capabilities
#   - Implement: run(nexus_state) -> dict
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
#       def run(self, nexus_state):
#           ...
# ==============================================================

from abc import ABC, abstractmethod
from core.registry import registry
from core.state import NexusState


class BaseAgent(ABC):
    """
    Abstract base class that every Nexus AI agent must inherit from.

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
    def run(self, nexus_state: NexusState) -> dict:
        """
        The main entry point called by the master graph.

        Every agent MUST implement this method. It receives the
        full NexusState, does its specialist work, and returns
        a partial state update dict.

        Args:
            nexus_state: The current master graph state.

        Returns:
            A dict with the fields this agent updated.
            At minimum: {"agent_outputs": {"<name>": "<answer>"}}
        """
        ...
