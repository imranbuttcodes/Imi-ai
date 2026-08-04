import asyncio
from typing import Any
from agents.base import BaseAgent
from agents.filesystem.schemas import FileSystemState
from agents.filesystem.graph import build_filesystem_graph
from langchain_mcp_adapters.client import MultiServerMCPClient
from core.config import settings
from core.constants import AgentName
from tools.filesystem import delete_path

class FileSystemAgent(BaseAgent):
    name = AgentName.FILESYSTEM
    description = (
        "Interacts with the local file system using the Anthropic MCP server. "
        "Can read files (text, media, multiple), write/edit files, create directories, "
        "list/tree directories, move/rename files, search files, and get file info. "
        "Can also delete files and directories securely (requires explicit user confirmation)."
    )
    capabilities = [
        "read-file", "read-text-file", "read-media-file", "read-multiple-files",
        "write-file", "edit-file", "create-directory", "list-directory",
        "list-directory-with-sizes", "directory-tree", "move-file",
        "search-files", "get-file-info", "list-allowed-directories", "delete-path"
    ]
    
    def __init__(self):
        super().__init__()
        
    def run(self, nexus_state: dict[str, Any]) -> dict[str, Any]:
        """
        The Master Graph calls this synchronous run method.
        We spawn an asyncio event loop to handle the MCP background process lifecycle.
        """
        return asyncio.run(self._arun(nexus_state))
        
    async def _arun(self, nexus_state: dict[str, Any]) -> dict[str, Any]:
        """
        Maintains the MCP server connection open for the entire duration of the
        tool-calling graph, ensuring tools execute properly when requested.
        """
        # Parse the comma-separated config string into a list of paths
        allowed_dirs = [d.strip() for d in settings.mcp_filesystem_allowed_dirs.split(",") if d.strip()]
        
        import sys
        npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
        
        # Initialize the MCP Server (langchain-mcp-adapters 0.1.0+ removes async with)
        client = MultiServerMCPClient({
            "filesystem": {
                "command": npx_cmd,
                "args": ["-y", "@modelcontextprotocol/server-filesystem", *allowed_dirs],
                "transport": "stdio"
            }
        })
        
        # Fetch tools dynamically and inject our native Python tools
        mcp_tools = await client.get_tools()
        tools = mcp_tools + [delete_path]
        
        # Build the state graph with live MCP tools
        graph = build_filesystem_graph(tools)
        
        # Provide a thread_id config so MemorySaver can track state across interruptions
        config = {"configurable": {"thread_id": "filesystem_session"}}
        
        # Construct the initial state
        agent_state = {
            "query": nexus_state["query"],
            "messages": [("user", nexus_state["query"])]
        }
        
        # Run the graph until completion or until it hits an interrupt()
        result = await graph.ainvoke(agent_state, config)
        
        # State machine interruption loop
        from langgraph.types import Command
        while True:
            snapshot = graph.get_state(config)
            
            # If there are no pending tasks, the graph is completely finished
            if not snapshot.next:
                break
                
            # If the graph paused because our delete tool called interrupt()
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                interrupt_msg = snapshot.tasks[0].interrupts[0].value
                
                # Prompt the human in the terminal
                print(interrupt_msg, end="")
                user_res = input().strip()
                
                # Feed the answer directly back into the state machine to resume execution!
                result = await graph.ainvoke(Command(resume=user_res), config)
            else:
                # Failsafe break
                break
        
        # Extract final answer
        final_message = result["messages"][-1].content
        
        return {
            "agent_outputs": {
                self.name: final_message
            } 
        }

# Auto-registration when discover_agents() imports this file

_filesystem_agent = FileSystemAgent()
