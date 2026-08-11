# ==============================================================
# infrastructure/memory/summarizer.py — Context Window Guard
# ==============================================================
# Prevents context overflow by automatically summarizing old 
# messages and archiving them.
# ==============================================================

import tiktoken
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_core.messages.utils import trim_messages
# Optional: some langchain versions have this natively, otherwise we fallback
try:
    from langchain_core.messages.utils import count_tokens_approximately
except ImportError:
    # Safe fallback if the version doesn't export it
    def count_tokens_approximately(msgs):
        return sum(len(str(m.content)) // 4 for m in msgs)

from core.state import ImiState
from infrastructure.memory.manager import memory_manager, MemoryType
from infrastructure.llm.factory import get_llm

MAX_CONTEXT_TOKENS = 4000  # Limit before we trigger summarization (Adjustable)


def summarize_node(state: ImiState) -> dict:
    """
    Checks token count. If over MAX_CONTEXT_TOKENS, it:
    1. Archives old messages
    2. Summarizes them
    3. Removes them from Working Memory
    """
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    token_count = count_tokens_approximately(messages)
    
    # Under limit? Do nothing, let the graph proceed normally.
    if token_count <= MAX_CONTEXT_TOKENS:
        return {}
        
    # Over limit! Use LangChain's trim_messages to get the safe subset
    # We ask trim_messages to keep the last 2000 tokens (leaving room for the summary)
    messages_to_keep = trim_messages(
        messages, 
        max_tokens=2000, 
        token_counter=count_tokens_approximately, 
        strategy="last",
        include_system=True
    )
    
    # Identify which messages got trimmed out
    kept_ids = {m.id for m in messages_to_keep if hasattr(m, 'id')}
    messages_to_archive = [m for m in messages if hasattr(m, 'id') and m.id not in kept_ids]
    
    if not messages_to_archive:
        return {}
    
    # 1. Archive the old messages safely so we never lose history
    memory_manager.store(MemoryType.ARCHIVE, messages_to_archive)
    
    # 2. Summarize
    old_summary = state.get("summary", "")
    
    if old_summary:
        prompt = f"""
Current Summary:

{old_summary}

Extend this summary using the new conversation.

Keep:
- User information
- Preferences
- Important facts
- Previous discussions
- Long-term memory
"""
    else:
        prompt = """
Create a concise summary of this conversation.

Keep:
- User information
- Preferences
- Important facts
"""

    prompt += "\nRecent Messages to compress:\n"
    for msg in messages_to_archive:
        role = msg.type if hasattr(msg, 'type') else "Unknown"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        prompt += f"{role}: {content}\n"
        
    llm = get_llm(role="memory")
    summary_response = llm.invoke(prompt)
    
    # 3. Create RemoveMessage instructions
    delete_commands = [RemoveMessage(id=m.id) for m in messages_to_archive if hasattr(m, 'id') and m.id]
    
    # 4. Return state update explicitly using the `summary` field
    return {
        "summary": summary_response.content,
        "messages": delete_commands
    }
