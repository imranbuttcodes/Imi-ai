# ==============================================================
# agents/research/graph.py — Research Agent Internal Graph
# ==============================================================

from langgraph.graph import StateGraph, START, END
from agents.research.state import ResearchState
from agents.research.nodes import plan_node, search_node, report_node

def build_research_graph():
    """
    Builds and compiles the Research Agent's internal graph.
    Flow: START → plan → search → report → END
    """
    graph = StateGraph(ResearchState)

    graph.add_node("plan",   plan_node)
    graph.add_node("search", search_node)
    graph.add_node("report", report_node)

    graph.add_edge(START,    "plan")
    graph.add_edge("plan",   "search")
    graph.add_edge("search", "report")
    graph.add_edge("report", END)

    return graph.compile()
