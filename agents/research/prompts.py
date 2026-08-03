# ==============================================================
# agents/research/prompts.py — Research Agent Prompts
# ==============================================================

from langchain_core.prompts import ChatPromptTemplate

PLANNING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Research Planner. "
               "Break the user's question down into 2-3 highly specific Google search queries. "
               "Focus on retrieving the most factual, up-to-date, and comprehensive information."),
    ("human", "Question: {query}")
])

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Research Analyst. "
               "Write a comprehensive, highly-structured markdown report based ONLY on the provided search results.\n\n"
               "Rules:\n"
               "- Use clear headings and bullet points.\n"
               "- STRICT RULE: You MUST cite your sources using the [Source: URL] tags provided in the context.\n"
               "- Do not make up information. If the search results are insufficient, state that clearly."),
    ("human", "Original Request: {query}\n\nSearch Results:\n{context}")
])
