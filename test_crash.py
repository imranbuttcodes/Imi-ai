import asyncio
from core.config import settings
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys

async def main():
    allowed_dirs = [d.strip() for d in settings.mcp_filesystem_allowed_dirs.split(",") if d.strip()]
    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    
    print(f"Using command: {npx_cmd}")
    print(f"Args: {['-y', '@modelcontextprotocol/server-filesystem', *allowed_dirs]}")
    
    client = MultiServerMCPClient({
        "filesystem": {
            "command": npx_cmd,
            "args": ["-y", "@modelcontextprotocol/server-filesystem", *allowed_dirs],
            "transport": "stdio"
        }
    })
    
    try:
        print("Fetching tools...")
        tools = await client.get_tools()
        print(f"Got {len(tools)} tools.")
        
        # Manually invoke a tool to see if it crashes during execution
        for t in tools:
            if t.name == "list_directory":
                print("Invoking list_directory...")
                res = await t.ainvoke({"path": allowed_dirs[-1]})
                print(f"Result: {res}")
                break
                
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
