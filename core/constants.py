# ==============================================================
# core/constants.py — Project-Wide Constants
# ==============================================================
# NEVER type raw strings for agent names or routing labels.
# Always import from here. One change here = changes everywhere.
#
# Usage:
#   from core.constants import AgentName, Routing
#   name = AgentName.KNOWLEDGE
# ==============================================================


class AgentName:
    """Canonical names for every agent in the system."""
    KNOWLEDGE = "knowledge"
    RESEARCH  = "research"
    # Add future agents here:
    # CODING    = "coding"
    FILESYSTEM = "filesystem"


class Routing:
    """Special routing labels used by the Router."""
    END      = "END"       # Main AI answered directly — stop here
    FALLBACK = "fallback"  # No valid agent found — use fallback node


class NodeName:
    """LangGraph node name suffixes and static node names."""
    AGENT_SUFFIX      = "_agent"     # e.g., "knowledge" + "_agent" = "knowledge_agent"
    MAIN_AI           = "main_ai"
    GENERATE_RESPONSE = "generate_response"
    FALLBACK          = "fallback"
