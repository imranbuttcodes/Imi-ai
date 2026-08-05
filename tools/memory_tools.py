from langchain_core.tools import tool
from infrastructure.memory.manager import memory_manager, MemoryType

@tool("save_to_memory")
def save_to_memory(fact: str) -> str:
    """Use this tool to permanently save rules, user preferences, or important facts to your long-term semantic memory."""
    memory_manager.store(MemoryType.SEMANTIC, [fact])
    return f"Successfully saved to long-term memory: {fact}"

@tool("search_memory")
def search_memory(query: str) -> str:
    """Use this tool to search your long-term semantic memory and archive history for past facts, preferences, or events."""
    semantic_results = memory_manager.retrieve(query)
    archive_results = memory_manager.retrieve_archive(query)
    
    result = "Search Results:\n"
    if semantic_results:
        result += f"\nSemantic Memory:\n{semantic_results}\n"
    if archive_results:
        result += f"\nArchive Memory:\n"
        for row in archive_results:
            result += f"- [{row['timestamp']}] {row['role']}: {row['content'][:100]}...\n"
            
    if result == "Search Results:\n":
        return "No results found in memory."
    return result
