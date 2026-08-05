# ==============================================================
# infrastructure/memory/reflection.py — Background Extraction
# ==============================================================
# Silently runs in the background to extract permanent facts
# from the conversation and store them in Semantic Memory.
# ==============================================================

from infrastructure.llm.factory import get_llm
from infrastructure.memory.manager import memory_manager, MemoryType
import logging

from pydantic import BaseModel, Field

logging.basicConfig(
    filename="reflection_logs.log",
    level=logging.INFO
)


class ExtractedFacts(BaseModel):
    has_facts: bool = Field(description="True if the user revealed new permanent facts about themselves, False otherwise.")
    facts: list[str] = Field(description="A list of newly extracted facts. Empty if has_facts is False.")

def extract_and_store_facts(messages: list):
    """
    Asynchronous background job that checks the recent turn for 
    new, permanent facts about the user and saves them.
    """
    if not messages:
        return
        
    # We only care about the most recent interaction (usually the last 2-4 messages)
    # We don't want to re-extract facts we already extracted.
    recent_messages = messages[-4:]
    
    conversation_text = ""
    for msg in recent_messages:
        role = msg.type if hasattr(msg, 'type') else "Unknown"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        conversation_text += f"{role}: {content}\n"

    # Pull existing facts from ChromaDB to prevent fuzzy duplicates!
    # We query using the conversation text so Chroma pulls the most relevant existing facts.
    existing_facts = memory_manager.retrieve(conversation_text)
    
    known_facts_context = ""
    if existing_facts:
        known_facts_context = f"\nEXISTING KNOWN FACTS ABOUT THE USER:\n{existing_facts}\n\nCRITICAL: Do NOT extract any facts that are already covered by or semantically identical to the Existing Known Facts above. Only extract NET-NEW information.\n"

    prompt = f"""
You are an advanced memory extraction engine.
Analyze the following recent conversation snippet. 

Did the user explicitly state any NEW, PERMANENT facts about themselves?
Examples of permanent facts:
- "I am building Nexus AI"
- "I code in Python and Windows"
- "I prefer dark mode"
- "I work as a software engineer"
- or whatever you think could be the fact you should remember about the user

Do NOT extract transient facts like "I am tired today", "I am eating pizza", or general conversation flow.{known_facts_context}
Conversation Snippet:
{conversation_text}
"""

    try:
        llm = get_llm(role="memory")
        structured_llm = llm.with_structured_output(ExtractedFacts)
        result: ExtractedFacts = structured_llm.invoke(prompt)
        
        if result.has_facts and result.facts:
            # Store in ChromaDB!
            memory_manager.store(MemoryType.SEMANTIC, result.facts)
            logging.info(f"Background Reflection extracted {len(result.facts)} facts.")
            
    except Exception as e:
        logging.error(f"Failed to extract facts in background: {e}")
