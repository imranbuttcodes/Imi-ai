# ==============================================================
# agents/knowledge/prompts.py — Knowledge Agent Prompts
# ==============================================================
# All LLM prompt templates for the Knowledge Agent live here.
# Nodes import from here — they never build prompts inline.
# ==============================================================

from langchain_core.prompts import ChatPromptTemplate


# ── Answer Generation Prompt ───────────────────────────────────
# Used by answer_node to synthesize an answer from retrieved chunks.

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise and helpful Knowledge Assistant.
Your job is to answer the user's question using ONLY the context provided below.

Rules:
- Base your answer strictly on the provided context.
- If the context does not contain enough information to answer, say so clearly.
- Do NOT make up information or use outside knowledge.
- Be concise and direct.
- STRICT RULE: You MUST cite your sources at the end of your answer. The context below will include metadata (like file names and page numbers). If you use a piece of context, you must cite its source (e.g., "[Source: policy.pdf, page 4]").

Context:
{context}"""
    ),
    (
        "human",
        "Question: {query}"
    )
])
