# ==============================================================
# agents/knowledge/graph.py — Knowledge Agent Internal Graph
# ==============================================================
# Wires the Knowledge Agent's nodes into a mini LangGraph.
# This graph is completely self-contained. The master graph
# only sees agent.run() — it never knows this exists.
# ==============================================================

from langgraph.graph import StateGraph, START, END
from agents.knowledge.state import KnowledgeState
from agents.knowledge.nodes import retrieve_node, answer_node


def build_knowledge_graph():
    """
    Builds and compiles the Knowledge Agent's internal graph.

    Flow:
        START → retrieve_node → answer_node → END
    """
    graph = StateGraph(KnowledgeState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer",   answer_node)

    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer",   END)

    return graph.compile()
