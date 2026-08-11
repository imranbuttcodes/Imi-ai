# ==============================================================
# infrastructure/retrieval/self_rag.py — Self-RAG Pipeline
# ==============================================================
# Your Self-RAG pipeline adapted for Imi AI.
#
# Flow:
#   should_retrieve → NO  → generate_direct → END
#                  → YES → retrieve → is_relevant
#                        → generate_from_context → is_sup → is_use
#                        → (retry loop with revise/rewrite)
#
# Usage:
#   from infrastructure.retrieval.self_rag import run_self_rag
#   answer = run_self_rag("What is the company policy?")  # returns str
# ==============================================================

from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from infrastructure.llm.factory import get_llm
from infrastructure.retrieval.vector_store import get_retriever
from core.config import settings


# ── Constants ──────────────────────────────────────────────────
MAX_RETRIES        = 10
MAX_REWRITE_TRIES  = 3


# ── State ──────────────────────────────────────────────────────
class SelfRAGState(TypedDict):
    question:       str
    need_retrieval: bool
    docs:           List[Document]
    relevant_docs:  List[Document]
    answer:         str
    retries:        int
    context:        str
    issup:          Literal["fully_supported", "partially_supported", "no_support"]
    evidence:       List[str]
    isuse:          Literal["useful", "not_useful"]
    use_reason:     str
    retrieval_query: str
    rewrite_tries:  int


# ── Pydantic Schemas ───────────────────────────────────────────
class ShouldRetrieve(BaseModel):
    should_retrieve: Literal["yes", "no"] = Field(..., description="Answer 'yes' if external documents are needed, 'no' if not.")

class RelevanceDecision(BaseModel):
    is_relevant: Literal["yes", "no"] = Field(..., description="Answer 'yes' if doc helps answer the question, 'no' if not.")

class IsSUPDecision(BaseModel):
    issup:    Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

class IsUSEDecision(BaseModel):
    isuse:     Literal["useful", "not_useful"]
    reason:    str = Field(..., description="Short reason in 1 line.")

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(..., description="Rewritten query optimized for vector retrieval.")


# ── Prompt Templates ───────────────────────────────────────────
decide_retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", "Decide whether retrieval is needed.\n"
               "should_retrieve=True: requires specific facts, citations, or info not in the model.\n"
               "should_retrieve=False: general explanations or definitions. If unsure, choose True."),
    ("human", "Question: {question}")
])

direct_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only your general knowledge. "
               "If unsure, say: 'I don't know based on my general knowledge.'"),
    ("human", "{question}")
])

is_relevant_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return is_relevant=true if the document contains info useful for answering the question."),
    ("human", "Question:\n{question}\n\nDocument:\n{document}")
])

rag_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY using the provided context. "
               "If context is insufficient, say: 'No relevant document found.'\n"
               "STRICT RULE: You MUST cite your sources at the end of your answer. If you use a piece of context, you must cite its source (e.g., [Source: policy.pdf, page 4])."),
    ("human", "Question:\n{question}\n\nContext:\n{context}")
])

issup_prompt = ChatPromptTemplate.from_messages([
    ("system", "Verify whether the ANSWER is grounded in the CONTEXT.\n"
               "issup: fully_supported / partially_supported / no_support.\n"
               "fully_supported: every claim is explicitly in CONTEXT with no unsupported interpretation.\n"
               "partially_supported: core facts are there but contains interpretive/qualitative phrasing.\n"
               "no_support: key claims are not in CONTEXT.\n"
               "Be strict. Include up to 3 direct quotes as evidence."),
    ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}")
])

isuse_prompt = ChatPromptTemplate.from_messages([
    ("system", "Judge USEFULNESS of the ANSWER for the QUESTION.\n"
               "useful: directly answers what was asked.\n"
               "not_useful: generic, off-topic, or only background. Keep reason to 1 line."),
    ("human", "Question:\n{question}\n\nAnswer:\n{answer}")
])

revise_prompt = ChatPromptTemplate.from_messages([
    ("system", "Output ONLY direct bullet-point quotes from CONTEXT that answer the question.\n"
               "Format: - <direct quote>\n"
               "Do NOT add any words beyond the quotes and bullet dashes."),
    ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}")
])

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question into a 6–16 word query optimized for vector retrieval "
               "over internal company PDFs. Add 2–5 high-signal keywords. Remove filler words."),
    ("human", "QUESTION:\n{question}\n\nPrevious query:\n{retrieval_query}\n\nAnswer (if any):\n{answer}")
])


# ── Nodes ──────────────────────────────────────────────────────
def should_retriever_node(state: SelfRAGState) -> dict:
    llm   = get_llm(role="evaluator")
    chain = decide_retrieval_prompt | llm.with_structured_output(ShouldRetrieve)
    result = chain.invoke({"question": state["question"]})
    return {"need_retrieval": result.should_retrieve == "yes"}


def generate_direct(state: SelfRAGState) -> dict:
    llm   = get_llm(role="specialist")
    chain = direct_gen_prompt | llm
    out   = chain.invoke({"question": state["question"]})
    return {"answer": out.content}


@traceable(run_type="retriever", name="SelfRAG_RetrieveNode", metadata={"layer": "retrieval", "source": "tavily_search"})
def retrieve_node(state: SelfRAGState) -> dict:
    retriever = get_retriever()
    query     = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(query)}


def is_relevant_node(state: SelfRAGState) -> dict:
    llm   = get_llm(role="evaluator")
    chain = llm.with_structured_output(RelevanceDecision)
    relevant_docs = []
    for doc in state["docs"]:
        msgs = is_relevant_prompt.format_messages(
            question=state["question"], document=doc.page_content
        )
        if chain.invoke(msgs).is_relevant == "yes":
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


def generate_from_context(state: SelfRAGState) -> dict:
    chunks = []
    for d in state.get("relevant_docs", []):
        source = d.metadata.get("source") or d.metadata.get("url") or "Unknown"
        page = d.metadata.get("page")
        meta_str = f"[Source: {source}{f', Page: {page}' if page is not None else ''}]"
        chunks.append(f"{meta_str}\n{d.page_content}")

    context = "\n\n---\n\n".join(chunks).strip()
    if not context:
        return {"answer": "No relevant document found.", "context": ""}
    llm   = get_llm(role="specialist")
    chain = rag_gen_prompt | llm
    out   = chain.invoke({"question": state["question"], "context": context})
    return {"answer": out.content, "context": context}


def no_relevant_docs(state: SelfRAGState) -> dict:
    return {"answer": "No relevant document found.", "context": ""}


def no_answer_found(state: SelfRAGState) -> dict:
    return {"answer": "No answer found.", "context": ""}


def is_sup_node(state: SelfRAGState) -> dict:
    llm   = get_llm(role="evaluator")
    chain = issup_prompt | llm.with_structured_output(IsSUPDecision)
    result = chain.invoke({
        "question": state["question"],
        "answer":   state.get("answer", ""),
        "context":  state.get("context", "")
    })
    return {"issup": result.issup, "evidence": result.evidence}


def revise_answer(state: SelfRAGState) -> dict:
    llm   = get_llm(role="specialist")
    chain = revise_prompt | llm
    out   = chain.invoke({
        "question": state["question"],
        "answer":   state.get("answer", ""),
        "context":  state.get("context", "")
    })
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}


def is_use_node(state: SelfRAGState) -> dict:
    llm   = get_llm(role="evaluator")
    chain = isuse_prompt | llm.with_structured_output(IsUSEDecision)
    result = chain.invoke({"question": state["question"], "answer": state.get("answer", "")})
    return {"isuse": result.isuse, "use_reason": result.reason}


def rewrite_question(state: SelfRAGState) -> dict:
    llm   = get_llm(role="evaluator")
    chain = rewrite_for_retrieval_prompt | llm.with_structured_output(RewriteDecision)
    result = chain.invoke({
        "question":       state["question"],
        "retrieval_query": state.get("retrieval_query", ""),
        "answer":          state.get("answer", "")
    })
    return {
        "retrieval_query": result.retrieval_query,
        "rewrite_tries":   state.get("rewrite_tries", 0) + 1,
        "docs":            [],
        "relevant_docs":   [],
        "context":         "",
    }


# ── Routing ────────────────────────────────────────────────────
def route_after_decide(state: SelfRAGState) -> str:
    return "retrieve" if state["need_retrieval"] else "generate_direct"

def route_after_relevance(state: SelfRAGState) -> str:
    return "generate_from_context" if state.get("relevant_docs") else "no_relevant_docs"

def route_after_issup(state: SelfRAGState) -> str:
    if state.get("issup") == "fully_supported":
        return "accept"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept"
    return "revise"

def route_after_isuse(state: SelfRAGState) -> str:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"


# ── Graph ──────────────────────────────────────────────────────
def _build_self_rag_graph():
    g = StateGraph(SelfRAGState)

    g.add_node("should_retrieve",       should_retriever_node)
    g.add_node("generate_direct",       generate_direct)
    g.add_node("retrieve",              retrieve_node)
    g.add_node("is_relevant",           is_relevant_node)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_relevant_docs",      no_relevant_docs)
    g.add_node("is_sup",                is_sup_node)
    g.add_node("revise_answer",         revise_answer)
    g.add_node("is_use",                is_use_node)
    g.add_node("rewrite_question",      rewrite_question)
    g.add_node("no_answer_found",       no_answer_found)

    g.add_edge(START, "should_retrieve")
    g.add_conditional_edges("should_retrieve", route_after_decide,
                            {"retrieve": "retrieve", "generate_direct": "generate_direct"})
    g.add_edge("retrieve", "is_relevant")
    g.add_conditional_edges("is_relevant", route_after_relevance,
                            {"generate_from_context": "generate_from_context",
                             "no_relevant_docs": "no_relevant_docs"})
    g.add_edge("generate_from_context", "is_sup")
    g.add_conditional_edges("is_sup", route_after_issup,
                            {"accept": "is_use", "revise": "revise_answer"})
    g.add_edge("revise_answer", "is_sup")
    g.add_conditional_edges("is_use", route_after_isuse,
                            {"END": END, "no_answer_found": "no_answer_found",
                             "rewrite_question": "rewrite_question"})
    g.add_edge("rewrite_question",  "retrieve")
    g.add_edge("no_relevant_docs",  END)
    g.add_edge("no_answer_found",   END)
    g.add_edge("generate_direct",   END)
    return g.compile()

_self_rag_graph = None

def run_self_rag(query: str) -> str:
    """
    Run the Self-RAG pipeline.
    Returns a complete final answer string.
    The Knowledge Agent skips answer_node and uses this directly.
    """
    global _self_rag_graph
    if _self_rag_graph is None:
        _self_rag_graph = _build_self_rag_graph()

    result = _self_rag_graph.invoke({
        "question":       query,
        "need_retrieval": False,
        "docs":           [],
        "relevant_docs":  [],
        "answer":         "",
        "retries":        0,
        "context":        "",
        "issup":          "no_support",
        "evidence":       [],
        "isuse":          "not_useful",
        "use_reason":     "",
        "retrieval_query": "",
        "rewrite_tries":   0,
    })
    return result.get("answer", "No answer found.")
