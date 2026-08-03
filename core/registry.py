# ==============================================================
# core/registry.py — Agent Registry
# ==============================================================
# The global in-memory directory of all available agents.
# Agents self-register here when they boot up.
# The master graph reads from here to build nodes dynamically.
#
# Usage:
#   from core.registry import registry, discover_agents
#   discover_agents()               # auto-import all agents
#   all_agents = registry.list_all()
#   agent = registry.get("knowledge")
# ==============================================================

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEntry:
    """A single agent's registration record."""
    name:         str
    agent:        Any         # The live agent instance (has a .run() method)
    capabilities: list[str]   # e.g., ["rag", "document-qa"]
    description:  str = ""    # Used in the Main AI's dynamic prompt


class AgentRegistry:
    """
    Plug-and-play agent registry.

    Agents self-register here on boot via self.register().
    The master graph reads here to dynamically build nodes.
    The Main AI reads here to build its routing prompt.
    """

    def __init__(self):
        self._agents: dict[str, AgentEntry] = {}

    def register(
        self,
        name:         str,
        agent:        Any,
        capabilities: list[str],
        description:  str = ""
    ) -> None:
        """Register an agent. Called by BaseAgent.register() automatically."""
        self._agents[name] = AgentEntry(
            name=name,
            agent=agent,
            capabilities=capabilities,
            description=description
        )

    def get(self, name: str) -> Any:
        """Get the live agent instance by name."""
        entry = self._agents.get(name)
        if not entry:
            raise ValueError(f"Agent '{name}' is not registered.")
        return entry.agent

    def list_all(self) -> list[AgentEntry]:
        """Return all registered agents. Used by Main AI and the graph builder."""
        return list(self._agents.values())

    def is_registered(self, name: str) -> bool:
        """Check if an agent exists by name. Used for validation."""
        return name in self._agents

    def find_by_capability(self, capability: str) -> list[str]:
        """Find all agent names that have a specific capability."""
        return [
            name for name, entry in self._agents.items()
            if capability in entry.capabilities
        ]


# ==============================================================
# Global singleton — import this everywhere
# Only ONE registry exists for the entire application lifetime.
# ==============================================================
registry = AgentRegistry()


def discover_agents() -> None:
    """
    Scans the agents/ directory and auto-imports every agent.py found.

    Importing an agent.py triggers the module-level instantiation
    (e.g. _knowledge_agent = KnowledgeAgent()), which calls __init__,
    which calls self.register(), which populates the global registry.

    This means: drop a new agent folder in → it is automatically live.
    No manual imports needed anywhere in the core OS.
    """
    import importlib
    from pathlib import Path

    agents_dir = Path(__file__).parent.parent / "agents"

    if not agents_dir.exists():
        return

    for path in sorted(agents_dir.iterdir()):
        # Skip: not a directory, is __pycache__, has no agent.py
        if not path.is_dir():
            continue
        if path.name.startswith("_"):
            continue
        if not (path / "agent.py").exists():
            continue

        try:
            importlib.import_module(f"agents.{path.name}.agent")
        except Exception as e:
            print(f"[Registry] Warning: Could not load agent '{path.name}': {e}")
