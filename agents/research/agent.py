# ==============================================================
# agents/research/agent.py — Research Agent
# ==============================================================

from agents.base import BaseAgent
from agents.research.graph import build_research_graph
from core.state import ImiState
from core.constants import AgentName


class ResearchAgent(BaseAgent):
    name         = AgentName.RESEARCH
    description  = "Performs deep web research by breaking down complex queries and synthesizing a detailed report from live internet data."
    capabilities = ["web-search", "deep-research", "live-data"]

    def __init__(self):
        self._graph = build_research_graph()
        super().__init__()

    def run(self, imi_state: ImiState) -> dict:
        """
        Entry point called by the master graph.
        """
        agent_state = {
            "query":          imi_state["query"],
            "web_queries":    [],
            "search_results": [],
            "report":         "",
        }

        result = self._graph.invoke(agent_state)

        # ── Output Transform ───────────────────────────────────
        return {
            "agent_outputs": {
                self.name: result["report"]
            }
        }


# ── Module-level instantiation ─────────────────────────────────
# This triggers auto-registration when discovered by registry.py
_research_agent = ResearchAgent()
