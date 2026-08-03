# ==============================================================
# agents/research/state.py — Research Agent Private State
# ==============================================================

from typing import TypedDict

class ResearchState(TypedDict):
    query:          str         # The user's original question
    web_queries:    list[str]   # The planned search queries
    search_results: list[str]   # The accumulated raw text from Tavily
    report:         str         # The final synthesized markdown report
