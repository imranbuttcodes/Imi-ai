# ==============================================================
# agents/knowledge/agent.py — Knowledge Agent
# ==============================================================
# The public face of the Knowledge Agent.
# This is what the master graph calls via agent.run().
#
# Responsibilities:
#   1. Define name, description, capabilities (for registry)
#   2. Transform NexusState → KnowledgeState (input)
#   3. Run the internal graph
#   4. Transform KnowledgeState → NexusState update (output)
# ==============================================================

from agents.base import BaseAgent
from agents.knowledge.graph import build_knowledge_graph
from core.state import NexusState
from core.constants import AgentName


class KnowledgeAgent(BaseAgent):
    name         = AgentName.KNOWLEDGE
    description  = "Answers questions from your private documents and internal knowledge base using RAG."
    capabilities = ["rag", "document-qa", "knowledge-retrieval"]

    def __init__(self):
        self._graph = build_knowledge_graph()
        super().__init__()   # triggers auto-registration in registry

    def run(self, nexus_state: NexusState) -> dict:
        """
        Entry point called by the master graph.

        Transforms NexusState → KnowledgeState,
        runs the internal pipeline,
        transforms result → NexusState partial update.
        """
        # ── Input Transform ────────────────────────────────────
        agent_state = {
            "query":            nexus_state["query"],
            "retrieved_chunks": [],
            "answer":           "",
        }

        # ── Run Internal Graph ─────────────────────────────────
        result = self._graph.invoke(agent_state)

        # ── Output Transform ───────────────────────────────────
        return {
            "agent_outputs": {
                self.name: result["answer"]
            }
        }


# ── Module-level instantiation ─────────────────────────────────
# This line runs when discover_agents() imports this file.
# It creates the agent AND triggers auto-registration via super().__init__()
_knowledge_agent = KnowledgeAgent()
