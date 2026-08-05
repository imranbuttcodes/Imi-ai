# ==============================================================
# graphs/nexus_graph.py — The Master Graph
# ==============================================================
# This is the blueprint for the entire Nexus AI system.
# It wires the Main AI to all the specialist agents dynamically.
#
# If you add a new agent to the registry, this graph will
# automatically build a node and a routing edge for it.
# No hardcoding required!
# ==============================================================

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from core.state import NexusState
from core.main_ai import main_ai_node, build_agent_tools
from core.registry import discover_agents

def route_after_main(state: NexusState) -> str:
    """
    Decides whether to route to the Tools node or finish the turn.
    """
    messages = state.get("messages", [])
    if not messages:
        return "summarize"
    
    last_message = messages[-1]
    # If the LLM made a tool call, route to the tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
        
    # Otherwise, the LLM has responded to the user, so summarize and end
    return "summarize"


def build_master_graph():
    """
    Builds the master LangGraph using a ReAct Orchestrator loop.
    
    Flow:
        START → main_ai_node ↔ tools (Specialist Agents)
                             ↳ summarize → END
    """
    # 1. Ensure all agents are loaded into the registry first
    discover_agents()
    
    # 2. Initialize the master graph state
    graph = StateGraph(NexusState)

    # 3. Add the Main AI (the ReAct Orchestrator)
    graph.add_node("main_ai", main_ai_node)
    graph.add_edge(START, "main_ai")

    # 4. Add the Tools Node (Wraps all Specialist Agents)
    tools = build_agent_tools()
    if tools:
        graph.add_node("tools", ToolNode(tools))
        # After executing tools, always loop back to the Main AI to reason again
        graph.add_edge("tools", "main_ai")

    # 5. Add the Summarizer Node (Context Guard)
    from infrastructure.memory.summarizer import summarize_node
    graph.add_node("summarize", summarize_node)
    graph.add_edge("summarize", END)

    # 6. Add the conditional routing from Main AI
    graph.add_conditional_edges(
        "main_ai",          
        route_after_main,   
        {"tools": "tools", "summarize": "summarize"} if tools else {"summarize": "summarize"}
    )

    # Compile with SQLite persistent checkpointer
    import sqlite3
    import os
    from core.config import settings
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    os.makedirs(settings.memory_dir, exist_ok=True)
    working_db_path = os.path.join(settings.memory_dir, "nexus_working.db")
    conn = sqlite3.connect(working_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    return graph.compile(checkpointer=checkpointer)
