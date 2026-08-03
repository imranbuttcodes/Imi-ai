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
from core.state import NexusState
from core.main_ai import main_ai_node
from core.router import route_request
from core.registry import registry, discover_agents


def build_master_graph():
    """
    Builds the master LangGraph dynamically based on registered agents.
    
    Flow:
        START → main_ai_node → route_request → [Specialist Agent] → END
                                             → END (direct answer)
    """
    # 1. Ensure all agents are loaded into the registry first
    discover_agents()
    
    # 2. Initialize the master graph state
    graph = StateGraph(NexusState)

    # 3. Add the Main AI (the front door)
    graph.add_node("main_ai", main_ai_node)
    graph.add_edge(START, "main_ai")

    # 4. Dynamically add a node for every registered agent
    all_agents = registry.list_all()
    agent_names = []
    
    for entry in all_agents:
        agent_name = entry.name
        # Note: We pass the agent's run method as the node function
        graph.add_node(agent_name, entry.agent.run)
        
        # After any specialist agent finishes, the graph ends
        graph.add_edge(agent_name, END)
        
        agent_names.append(agent_name)

    # 5. Build the conditional routing map
    # Maps what the router returns (string) to the actual node name
    route_map = {name: name for name in agent_names}
    route_map[END] = END  # Handle direct answers

    # 6. Add the conditional edges from Main AI
    graph.add_conditional_edges(
        "main_ai",          # The node we are routing FROM
        route_request,      # The function that decides where to go
        route_map           # The dictionary mapping choices to nodes
    )

    # Compile and return the executable application
    return graph.compile()
