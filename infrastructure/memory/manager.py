# ==============================================================
# infrastructure/memory/manager.py — The Memory Manager
# ==============================================================
# This is the core infrastructure service for memory.
# It provides a clean API (store, retrieve, retrieve_archive)
# so agents don't have to manage databases manually.
# ==============================================================

import sqlite3
import json
from enum import Enum
from typing import Any
from pathlib import Path
from langsmith import traceable


class MemoryType(Enum):
    WORKING = "working"     # Managed directly by LangGraph SqliteSaver
    SEMANTIC = "semantic"   # Managed via ChromaDB (Long-term facts)
    ARCHIVE = "archive"     # Managed via SQLite (Deep episodic history)


class MemoryManager:
    def __init__(self, db_path: str = None):
        import os
        from core.config import settings
        
        # Ensure the clean data directory exists
        os.makedirs(settings.memory_dir, exist_ok=True)
        
        # Use provided path or default to the memory_dir
        self.db_path = db_path or os.path.join(settings.memory_dir, "nexus_archive.db")
        self._init_archive_db()

    def _init_archive_db(self):
        """Initializes the SQLite Archive table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def store(self, memory_type: MemoryType, data: Any):
        """
        Stores memory based on its designated layer.
        """
        if memory_type == MemoryType.WORKING:
            # Working memory is handled natively by LangGraph's checkpointer.
            pass
            
        elif memory_type == MemoryType.ARCHIVE:
            # Data should be a list of LangChain Message objects
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            for msg in data:
                # msg is typically an AnyMessage (HumanMessage, AIMessage, etc.)
                role = msg.type if hasattr(msg, 'type') else "unknown"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # Check for metadata/id
                msg_id = msg.id if hasattr(msg, 'id') else None
                meta_json = json.dumps({"msg_id": msg_id}) if msg_id else "{}"
                
                cursor.execute(
                    "INSERT INTO archive (role, content, metadata) VALUES (?, ?, ?)",
                    (role, content, meta_json)
                )
            
            conn.commit()
            conn.close()
            
        elif memory_type == MemoryType.SEMANTIC:
            # Data should be a list of strings (facts)
            if not data:
                return
                
            from langchain_chroma import Chroma
            from infrastructure.retrieval.embeddings import get_embeddings
            from core.config import settings
            import os
            import uuid
            
            semantic_dir = os.path.join(settings.memory_dir, "nexus_semantic_db")
            
            # Lazy initialize the Semantic Vector DB cleanly in data/memory
            vectorstore = Chroma(
                collection_name="nexus_semantic",
                persist_directory=semantic_dir,
                embedding_function=get_embeddings()
            )
            
            # Create unique IDs for facts to prevent duplicates if they match exactly
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, fact)) for fact in data]
            
            # We add texts. If a fact ID already exists, Chroma ignores or overwrites it
            vectorstore.add_texts(texts=data, ids=ids)

    @traceable(run_type="retriever", name="SemanticMemoryRetrieval", metadata={"layer": "semantic_memory", "database": "chroma_db"})
    def retrieve(self, query: str = "User profile and preferences") -> str:
        """
        Fast Retrieval from Semantic Memory (Phase 3).
        Takes ~5ms to pull the top facts about the user.
        """
        try:
            from langchain_chroma import Chroma
            from infrastructure.retrieval.embeddings import get_embeddings
            from core.config import settings
            import os
            
            semantic_dir = os.path.join(settings.memory_dir, "nexus_semantic_db")
            
            vectorstore = Chroma(
                collection_name="nexus_semantic",
                persist_directory=semantic_dir,
                embedding_function=get_embeddings()
            )
            
            # Retrieve top 5 most relevant facts
            results = vectorstore.similarity_search(query, k=5)
            
            if not results:
                return ""
                
            facts = [doc.page_content for doc in results]
            return "\n".join(f"- {fact}" for fact in facts)
        except Exception:
            return ""

    @traceable(run_type="retriever", name="ArchiveMemoryRetrieval", metadata={"layer": "episodic_archive", "database": "sqlite_db"})
    def retrieve_archive(self, query: str) -> list[dict]:
        """
        Deep episodic retrieval from the SQLite archive.
        Useful when the user asks about a conversation from 3 months ago.
        """
        # Basic keyword search for now (we can upgrade to FTS5 later)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        # SQLite LIKE is case-insensitive by default
        cursor.execute(
            "SELECT timestamp, role, content FROM archive WHERE content LIKE ? ORDER BY timestamp DESC LIMIT 50",
            (f"%{query}%",)
        )
        
        results = [{"timestamp": row[0], "role": row[1], "content": row[2]} for row in cursor.fetchall()]
        conn.close()
        return results

# Singleton instance for the system to use
memory_manager = MemoryManager()
