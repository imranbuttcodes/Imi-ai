from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from infrastructure.llm.factory import get_llm
from agents.filesystem.schemas import FileSystemState
from core.config import settings

def build_filesystem_graph(tools):
    """
    Builds the tool-calling StateGraph for the FileSystem MCP Agent.
    It takes the async MCP tools injected from the Agent's event loop session.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    async def agent_node(state: FileSystemState):
        from langchain_core.messages import SystemMessage
        
        # Give the LLM a robust, generic system prompt mapping intent to absolute paths
        system_msg = SystemMessage(
            content=(
                "You are an autonomous File System assistant. "
                f"You have strict access to the following base directories: {settings.mcp_filesystem_allowed_dirs}. "
                "When a user refers to a generic folder name (e.g., 'desktop', 'documents', 'my project'), "
                "you must infer the correct absolute path from the allowed directories list above. "
                "Never guess or hallucinate absolute paths outside of these allowed directories. "
                "If you are unsure of the directory structure, use your available tools (like list_allowed_directories or search_files) to explore first."
            )
        )
        
        # We must await the async ainvoke so it properly resolves within the event loop
        response = await llm_with_tools.ainvoke([system_msg] + state["messages"])
        return {"messages": [response]}
        
    graph = StateGraph(FileSystemState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    from langgraph.checkpoint.memory import MemorySaver
    
    # Flow: START -> agent <---> tools -> END
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "   ")
    
    # Must use a checkpointer to support interrupt() state preservation
    return graph.compile(checkpointer=MemorySaver())
