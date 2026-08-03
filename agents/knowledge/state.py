# ==============================================================
# agents/knowledge/state.py — Knowledge Agent Private State
# ==============================================================
# This state is ONLY used inside the Knowledge Agent's
# internal graph. The master graph never sees it.
#
# Design: contains only what THIS agent's pipeline needs.
# ==============================================================

from typing import TypedDict


class KnowledgeState(TypedDict):
    query:             str         # The user's original question
    retrieved_chunks:  list[str]   # Raw text chunks from the vector store
    answer:            str         # The final synthesized answer
