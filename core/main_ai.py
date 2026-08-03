# ==============================================================
# core/main_ai.py — The Front Door
# ==============================================================
# The Main AI is the first LLM the user talks to.
# It reads the registry to see what agents are available.
#
# It makes exactly ONE decision:
# 1. Answer directly (if it's a general question)
# 2. Route to a specialist (e.g., "knowledge", "research")
#
# Usage (called by the master graph):
#   from core.main_ai import main_ai_node
#   state = main_ai_node(state)
# ==============================================================

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from core.state import NexusState
from core.registry import registry
from infrastructure.llm.factory import get_llm


# ── Structured Output Schema ───────────────────────────────────
class RoutingDecision(BaseModel):
    next_agent: str = Field(
        ...,
        description="The name of the agent to route to, or 'END' to answer directly."
    )
    direct_answer: str = Field(
        default="",
        description="If next_agent is 'END', put your answer here. Otherwise leave empty."
    )


def build_system_prompt() -> str:
    """
    Dynamically builds the system prompt based on currently registered agents.
    If you add a new agent, it automatically appears here.
    """
    agents = registry.list_all()
    
    if not agents:
        agent_descriptions = "No specialist agents are currently available."
        valid_agents = "'END'"
    else:
        desc_lines = []
        for a in agents:
            desc_lines.append(f"- {a.name}: {a.description} (Capabilities: {', '.join(a.capabilities)})")
        agent_descriptions = "\n".join(desc_lines)
        valid_agents = ", ".join([f"'{a.name}'" for a in agents]) + ", or 'END'"

    return f"""You are the Nexus Main AI.
Your job is to analyze the user's request and decide who should handle it.

Available Specialist Agents:
{agent_descriptions}

Rules:
1. If a specialist agent is better suited for the request, set next_agent to their name.
2. If no agent is suited, OR it's a simple greeting/general question, handle it yourself.
   Set next_agent to 'END' and provide your response in direct_answer.
3. next_agent MUST be exactly one of: {valid_agents}.
"""


def main_ai_node(state: NexusState) -> dict:
    """
    The master graph calls this node first.
    It reads the user's query and decides where to route it.
    """
    system_prompt = build_system_prompt()
    query = state["query"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])

    # We use a fast, smart model for routing
    llm = get_llm()
    chain = prompt | llm.with_structured_output(RoutingDecision)
    
    decision: RoutingDecision = chain.invoke({"query": query})

    return {
        "next_agent": decision.next_agent,
        "final_response": decision.direct_answer
    }
