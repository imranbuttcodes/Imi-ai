# ==============================================================
# agents/research/nodes.py — Research Agent Node Functions
# ==============================================================

from agents.research.state import ResearchState
from agents.research.prompts import PLANNING_PROMPT, REPORT_PROMPT
from agents.research.schemas import SearchPlan
from infrastructure.llm.factory import get_llm
from core.config import settings

def plan_node(state: ResearchState) -> dict:
    """Uses the LLM to break the query into search queries."""
    llm = get_llm(settings.eval_model)
    chain = PLANNING_PROMPT | llm.with_structured_output(SearchPlan)
    
    plan = chain.invoke({"query": state["query"]})
    return {"web_queries": plan.queries}


def search_node(state: ResearchState) -> dict:
    """Executes the search queries using Tavily."""
    from langchain_tavily import TavilySearch
    
    tool = TavilySearch(
        max_results=3, 
        search_depth="advanced",
        tavily_api_key=settings.tavily_api_key
    )
    
    results = []
    for query in state.get("web_queries", []):
        try:
            search_response = tool.invoke(query)
            for r in search_response.get("results", []):
                # Format with citations
                url = r.get("url", "Unknown URL")
                content = r.get("content", "")
                results.append(f"[Source: {url}]\n{content}")
        except Exception as e:
            print(f"Tavily search failed for query '{query}': {e}")
            
    return {"search_results": results}


def report_node(state: ResearchState) -> dict:
    """Synthesizes the search results into a report."""
    chunks = state.get("search_results", [])
    context = "\n\n---\n\n".join(chunks) if chunks else "No search results found."
    
    llm = get_llm(settings.gen_model)
    chain = REPORT_PROMPT | llm
    
    response = chain.invoke({
        "query": state["query"],
        "context": context
    })
    
    return {"report": response.content}
