# ==============================================================
# core/state.py — Master NexusState
# ==============================================================
# The single shared dictionary passed through the entire
# master graph. Every node reads from and writes back to this.
#
# Design: LEAN. Only what the master graph needs.
#         Agents have their own private state internally.
#
# Node responsibilities:
#   query          → set by app.py before graph starts
#   next_agent     → set by main_ai_node
#   agent_outputs  → set by specialist agents
#   final_response → set by generate_response_node OR main_ai_node
#   errors         → set by any node that catches an exception
#   metadata       → optional, set by anyone
# ==============================================================

from typing import TypedDict, Any


class NexusState(TypedDict):
    query:          str               # Original user query
    next_agent:     str               # Routing decision: agent name, "END", or "fallback"
    agent_outputs:  dict[str, str]    # {"knowledge": "...", "research": "..."}
    final_response: str               # The final answer shown to the user
    errors:         list[str]         # Errors from any agent
    metadata:       dict[str, Any]    # Optional extra data (timestamps, tokens used, etc.)
