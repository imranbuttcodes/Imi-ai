# ==============================================================
# test_filesystem_mcp.py — Standalone MCP Filesystem Server Test
# ==============================================================
# Run this script to verify that your Node/NVM setup, MCP server,
# and langchain-mcp-adapters client work correctly on Ubuntu.
#
# Usage:
#   python test_filesystem_mcp.py
# ==============================================================

import asyncio
import os
import shutil
import sys
from pathlib import Path

# 1. Ensure Node.js and MCP binaries (from NVM or PATH) are accessible
def ensure_node_in_path():
    """Ensures Node.js and MCP binaries are in os.environ['PATH']."""
    if shutil.which("node") and shutil.which("mcp-server-filesystem"):
        return

    # Check NVM node version directories on Linux/macOS
    nvm_node_dirs = sorted(Path.home().glob(".nvm/versions/node/*/bin"))
    if nvm_node_dirs:
        latest_node_bin = str(nvm_node_dirs[-1])
        current_path = os.environ.get("PATH", "")
        if latest_node_bin not in current_path:
            os.environ["PATH"] = f"{latest_node_bin}:{current_path}"

ensure_node_in_path()

print("=" * 60)
print(" MCP FILESYSTEM SERVER SYSTEM DIAGNOSTICS")
print("=" * 60)
print(f" Python Executable : {sys.executable}")
print(f" Operating System  : {sys.platform}")
print(f" Node Path         : {shutil.which('node') or 'NOT FOUND'}")
print(f" NPX Path          : {shutil.which('npx') or 'NOT FOUND'}")
print(f" MCP Server Path   : {shutil.which('mcp-server-filesystem') or 'NOT FOUND'}")

from langchain_mcp_adapters.client import MultiServerMCPClient

async def run_mcp_tests():
    # Target testing directory (allowed directory)
    test_dir = os.path.abspath(os.path.dirname(__file__))
    print(f" Allowed Directory : {test_dir}")
    print("-" * 60)

    # Determine command (use direct mcp-server-filesystem binary if available)
    mcp_binary = shutil.which("mcp-server-filesystem")
    if mcp_binary:
        command = mcp_binary
        args = [test_dir]
    else:
        command = "npx.cmd" if sys.platform == "win32" else (shutil.which("npx") or "npx")
        args = ["-y", "@modelcontextprotocol/server-filesystem", test_dir]

    print(f"\n[1] Starting MCP Server Process...")
    print(f"    Command: {command}")
    print(f"    Args   : {args}\n")

    client = MultiServerMCPClient({
        "filesystem": {
            "command": command,
            "args": args,
            "transport": "stdio"
        }
    })

    # Fetch Tools
    tools = await client.get_tools()
    print(f"[2] Discovered {len(tools)} MCP Tools:")
    tool_map = {t.name: t for t in tools}
    for name in tool_map:
        print(f"    - {name}")

    print("\n" + "=" * 60)
    print(" EXECUTING WORKING EXAMPLES")
    print("=" * 60)

    # Example 1: List Directory
    if "list_directory" in tool_map:
        print("\n--- Example 1: list_directory ---")
        try:
            res = await tool_map["list_directory"].ainvoke({"path": test_dir})
            output_snippet = str(res)[:500]
            print(f"Result:\n{output_snippet}")
        except Exception as e:
            print(f"Error executing list_directory: {e}")

    # Example 2: Write File
    test_file_path = os.path.join(test_dir, "_mcp_test_file.txt")
    if "write_file" in tool_map:
        print(f"\n--- Example 2: write_file ({test_file_path}) ---")
        try:
            res = await tool_map["write_file"].ainvoke({
                "path": test_file_path,
                "content": "Hello from Imi AI MCP Test Script!\nStatus: Operational."
            })
            print(f"Result: {res}")
        except Exception as e:
            print(f"Error executing write_file: {e}")

    # Example 3: Read File
    if "read_file" in tool_map:
        print(f"\n--- Example 3: read_file ({test_file_path}) ---")
        try:
            res = await tool_map["read_file"].ainvoke({"path": test_file_path})
            print(f"Result:\n{res}")
        except Exception as e:
            print(f"Error executing read_file: {e}")

    # Example 4: File Info
    if "get_file_info" in tool_map:
        print(f"\n--- Example 4: get_file_info ({test_file_path}) ---")
        try:
            res = await tool_map["get_file_info"].ainvoke({"path": test_file_path})
            print(f"Result:\n{res}")
        except Exception as e:
            print(f"Error executing get_file_info: {e}")

    # Clean up test file
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"\n[Cleanup] Removed temporary test file: {test_file_path}")

    print("\n" + "=" * 60)
    print(" MCP FILESYSTEM SERVER TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_mcp_tests())
