from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from core.state import ImiState
from core.registry import registry
from infrastructure.llm.factory import get_llm
from infrastructure.memory.manager import memory_manager
from tools.memory_tools import save_to_memory, search_memory



def _create_agent_tool(entry):
    """Factory function to properly scope the agent instance without leaking it into the JSON schema."""
    agent_instance = entry.agent
    agent_name = entry.name
    
    def agent_tool_func(query: str) -> str:
        try:
            state = {"query": query, "messages": [("user", query)]}
            result = agent_instance.run(state)
            raw_output = result.get("agent_outputs", {}).get(agent_name, "Task completed but no output returned.")
            
            # TRUNCATION FIX: Prevent massive file reads from blowing up the context window!
            if isinstance(raw_output, str) and len(raw_output) > 10000:
                return raw_output[:10000] + "\n\n...[OUTPUT TRUNCATED TO PREVENT CONTEXT WINDOW EXPLOSION]..."
            return raw_output
        except Exception as e:
            return f"Error executing {agent_name} agent: {str(e)}"
        
    # Standardize the function name and docstring for LangChain Tool
    agent_tool_func.__name__ = f"call_{agent_name}_agent"
    agent_tool_func.__doc__ = (
        f"Use this tool to delegate tasks to the {agent_name} agent. "
        f"Capabilities: {', '.join(entry.capabilities)}. "
        f"Description: {entry.description}. "
        f"The query must be a fully detailed standalone instruction."
    )
    
    return tool(agent_tool_func)

def build_agent_tools() -> list:
    """
    Dynamically creates LangChain tools from registered agents and core tools.
    """
    tools = [save_to_memory, search_memory]
    for entry in registry.list_all():
        tools.append(_create_agent_tool(entry))
    return tools

def main_ai_node(state: ImiState) -> dict:
    """
    The master Orchestrator node using a ReAct loop.
    It binds the agent tools and decides whether to use them or respond to the user.
    """
    tools = build_agent_tools()
    
    system_prompt = """You are the Imi Main AI Orchestrator.
Your job is to analyze the user's request and fulfill it.
If you need to perform actions (like web search, file reads, or memory retrieval), you MUST use the available tools.
You can use tools sequentially to gather information before answering.
Once you have all the information, synthesize it into a final conversational response to the user.
"""
    
    messages = list(state.get("messages", []))
    summary = state.get("summary", "")
    
    # --- MEMORY INJECTION ---
    semantic_facts = memory_manager.retrieve("User profile and facts")
    
    memory_context = ""
    if semantic_facts:
        memory_context += f"Known User Facts (Semantic Memory):\n{semantic_facts}\n\n"
    if summary:
        memory_context += f"Conversation Summary (Working Memory):\n{summary}\n\n"
        
    if memory_context:
        memory_msg = SystemMessage(
            content=f"--- CONTEXT ---\n{memory_context}Use this context to inform your responses."
        )
        messages.insert(0, memory_msg)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm = get_llm(role="router")
    
    if tools:
        llm = llm.bind_tools(tools)
        
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    # DEBUG TRACE
    if hasattr(response, "tool_calls") and response.tool_calls:
        from rich.console import Console
        console = Console()
        for call in response.tool_calls:
            console.print(f"[bold yellow]⚙️ Main AI Action:[/bold yellow] Triggering [cyan]{call['name']}[/cyan]...")
            
    return {"messages": [response]}
