# ==============================================================
# agents/knowledge/nodes.py — Knowledge Agent Node Functions
# ==============================================================
# Pipeline: retrieve_node → answer_node
#
# retrieve_node chooses the RAG strategy from settings.rag_strategy:
#   "basic"    → direct hybrid retrieval (fast)
#   "crag"     → Corrective RAG (web fallback for bad chunks)
#   "self_rag" → Self-RAG (hallucination-checked, self-correcting)
#
# answer_node synthesizes the final answer from retrieved chunks.
# If self_rag already generated the answer, answer_node passes it through.
# ==============================================================

from agents.knowledge.state import KnowledgeState
from agents.knowledge.prompts import ANSWER_PROMPT
from infrastructure.llm.factory import get_llm
from core.config import settings


def retrieve_node(state: KnowledgeState) -> dict:
    """
    Retrieves relevant document chunks using the configured RAG strategy.
    Set RAG_STRATEGY in .env to switch strategies.
    """
    query    = state["query"]
    strategy = settings.rag_strategy

    # ── CRAG ──────────────────────────────────────────────────
    if strategy == "crag":
        from infrastructure.retrieval.crag import run_crag
        chunks = run_crag(query)
        return {"retrieved_chunks": chunks}

    # ── Self-RAG ──────────────────────────────────────────────
    if strategy == "self_rag":
        from infrastructure.retrieval.self_rag import run_self_rag
        # Self-RAG generates the full answer internally
        answer = run_self_rag(query)
        return {
            "retrieved_chunks": [],
            "answer": answer        # answer_node will detect this and pass through
        }

    # ── Basic RAG (default fallback) ───────────────────────────
    from infrastructure.retrieval.vector_store import get_retriever
    retriever = get_retriever()
    docs      = retriever.invoke(query)
    chunks    = [
        f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    ]
    return {"retrieved_chunks": chunks}


def answer_node(state: KnowledgeState) -> dict:
    """
    Synthesizes a final answer from retrieved chunks.
    If Self-RAG already generated an answer, passes it through unchanged.
    """
    # Self-RAG already produced a complete answer — skip generation
    if state.get("answer"):
        return {"answer": state["answer"]}

    chunks  = state.get("retrieved_chunks", [])
    query   = state["query"]
    context = "\n\n".join(chunks) if chunks else "No context available."

    llm      = get_llm()
    chain    = ANSWER_PROMPT | llm
    response = chain.invoke({"context": context, "query": query})

    return {"answer": response.content}
