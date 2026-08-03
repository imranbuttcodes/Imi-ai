# ==============================================================
# core/router.py — The Train Switch
# ==============================================================
# This is a LangGraph conditional edge function.
# It reads the state returned by Main AI, and tells the graph
# which node to run next.
#
# If Main AI said "next_agent": "knowledge", it returns "knowledge".
# If Main AI said "next_agent": "END", it returns END.
# ==============================================================

from langgraph.graph import END
from core.state import NexusState
from core.registry import registry

def route_request(state: NexusState) -> str:
    """
    Decides the next node in the master graph based on Main AI's decision.
    """
    next_agent = state.get("next_agent")

    # If the Main AI answered directly, we are done.
    if not next_agent or next_agent == "END":
        return END

    # Verify the agent actually exists (failsafe)
    if not registry.is_registered(next_agent):
        print(f"[Router Warning] Main AI hallucinated agent '{next_agent}'. Defaulting to END.")
        return END

    # Return the name of the agent node to run next
    return next_agent
