# ==============================================================
# infrastructure/retrieval/crag.py — Corrective RAG Pipeline
# ==============================================================
# Your CRAG pipeline adapted for Imi AI.
#
# Flow:
#   retrieve → eval_each_doc → (CORRECT → refine)
#                            → (INCORRECT/AMBIGUOUS → rewrite → web_search → refine)
#                            → generate
#
# Usage:
#   from infrastructure.retrieval.crag import run_crag
#   chunks = run_crag("What is the Q3 budget?")  # returns list[str]
# ==============================================================

import nltk
import os
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langsmith import traceable
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from infrastructure.llm.factory import get_llm
from infrastructure.retrieval.vector_store import get_retriever
from core.config import settings

# ── Thresholds ─────────────────────────────────────────────────
LOWER_THRESHOLD = 0.3
UPPER_THRESHOLD = 0.7


# ── State ──────────────────────────────────────────────────────
class CRAGState(TypedDict, total=False):
    question:       str
    docs:           List[Document]
    good_docs:      List[Document]
    web_docs:       List[Document]
    strips:         List[str]
    kept_strips:    List[str]
    refine_context: str
    web_query:      str
    VERDICT:        str
    reason:         str


# ── Pydantic Schemas ───────────────────────────────────────────
class DocScoreSchema(BaseModel):
    score: float = Field(description="Relevance score 0.0–1.0", ge=0.0, le=1.0)

class WebQuery(BaseModel):
    query: str

from typing import Literal

class KeepOrDrop(BaseModel):
    keep: Literal["yes", "no"] = Field(description="Answer 'yes' to keep, 'no' to drop")


# ── Prompt Templates ───────────────────────────────────────────
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict retrieval evaluator for RAG. "
               "Score the chunk's relevance to the question between 0.0 and 1.0. "
               "1.0 = chunk alone fully answers it. 0.0 = completely irrelevant. Be conservative."),
    ("human", "Question: {question}\n\nChunk: {chunk}")
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user question into a web search query of 6–14 keywords. "
               "If recency is implied, add a constraint like (last 30 days). "
               "Do NOT answer the question."),
    ("human", "Question: {question}")
])

filter_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return keep=true ONLY if the sentence directly helps answer the question."),
    ("human", "Question: {question}\n\nSentence: {sentence}")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. "
               "Answer ONLY using the provided context. "
               "If context is insufficient, say: 'I don't know.'"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


# ── Nodes ──────────────────────────────────────────────────────
@traceable(run_type="retriever", name="CRAG_RetrieveNode", metadata={"layer": "retrieval", "source": "tavily_search"})
def retrieve_node(state: CRAGState) -> dict:
    retriever = get_retriever()
    return {"docs": retriever.invoke(state["question"])}


def eval_each_doc_node(state: CRAGState) -> dict:
    eval_llm   = get_llm(role="evaluator")
    eval_chain = eval_prompt | eval_llm.with_structured_output(DocScoreSchema)

    scores, good_docs = [], []
    for doc in state.get("docs", []):
        score = eval_chain.invoke({"question": state["question"], "chunk": doc.page_content}).score
        scores.append(score)
        if score > LOWER_THRESHOLD:
            good_docs.append(doc)

    if not scores:
        verdict, reason = "incorrect", "No documents retrieved"
    elif any(s > UPPER_THRESHOLD for s in scores):
        verdict, reason = "correct", f"At least one doc scored above {UPPER_THRESHOLD}"
    elif all(s < LOWER_THRESHOLD for s in scores):
        verdict, reason = "incorrect", f"All docs scored below {LOWER_THRESHOLD}"
    else:
        verdict, reason = "ambiguous", "Mixed scores — triggering web search"

    return {"good_docs": good_docs, "VERDICT": verdict, "reason": reason}


def rewrite_query_node(state: CRAGState) -> dict:
    rewrite_llm   = get_llm(role="evaluator")
    rewrite_chain = rewrite_prompt | rewrite_llm.with_structured_output(WebQuery)
    web_query = rewrite_chain.invoke({"question": state["question"]}).query
    return {"web_query": web_query}


def web_search_node(state: CRAGState) -> dict:
    from langchain_tavily import TavilySearch
    query = state.get("web_query") or state["question"]
    tool  = TavilySearch(max_results=3, search_depth="advanced",
                         tavily_api_key=settings.tavily_api_key)
    results  = tool.invoke(query)
    web_docs = [
        Document(
            page_content=f"TITLE: {r.get('title','')}\nURL: {r.get('url','')}\nCONTENT: {r.get('content','')}",
            metadata={"title": r.get("title",""), "url": r.get("url","")}
        )
        for r in results.get("results", [])
    ]
    return {"web_docs": web_docs}


def refine_node(state: CRAGState) -> dict:
    filter_llm   = get_llm(role="evaluator")
    filter_chain = filter_prompt | filter_llm.with_structured_output(KeepOrDrop)

    verdict = state.get("VERDICT")
    good_docs = state.get("good_docs", [])
    web_docs = state.get("web_docs", [])

    docs_to_use = (
        good_docs if verdict == "correct"
        else web_docs if verdict == "incorrect"
        else good_docs + web_docs
    )

    strips_with_meta = []
    for d in docs_to_use:
        source = d.metadata.get("source") or d.metadata.get("url") or "Unknown"
        page = d.metadata.get("page")
        meta_str = f"[Source: {source}{f', Page: {page}' if page is not None else ''}]"
        
        try:
            doc_strips = nltk.sent_tokenize(d.page_content.strip())
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            doc_strips = nltk.sent_tokenize(d.page_content.strip())
            
        for s in doc_strips:
            strips_with_meta.append((s, meta_str))

    kept = []
    for s, meta_str in strips_with_meta:
        if filter_chain.invoke({"question": state["question"], "sentence": s}).keep == "yes":
            # Re-attach the citation to the surviving sentence
            kept.append(f"{meta_str}\n{s}")

    return {"strips": [s for s, m in strips_with_meta], "kept_strips": kept, "refine_context": "\n\n".join(kept)}


# ── Routing ────────────────────────────────────────────────────
def route_after_eval(state: CRAGState) -> str:
    return "refine" if state.get("VERDICT") == "correct" else "rewrite_query"


# ── Graph ──────────────────────────────────────────────────────
def _build_crag_graph():
    g = StateGraph(CRAGState)
    g.add_node("retrieve",      retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search",    web_search_node)
    g.add_node("refine",        refine_node)

    g.add_edge(START,          "retrieve")
    g.add_edge("retrieve",     "eval_each_doc")
    g.add_conditional_edges("eval_each_doc", route_after_eval,
                            {"refine": "refine", "rewrite_query": "rewrite_query"})
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search",    "refine")
    g.add_edge("refine",        END)
    return g.compile()

_crag_graph = None

def run_crag(query: str) -> list[str]:
    """
    Run the CRAG pipeline.
    Returns kept_strips: a list of filtered, relevant sentences.
    The Knowledge Agent's answer_node synthesizes the final answer from these.
    """
    global _crag_graph
    if _crag_graph is None:
        _crag_graph = _build_crag_graph()

    result = _crag_graph.invoke({"question": query})
    return result.get("kept_strips", [])
