# ==============================================================
# agents/research/schemas.py — Research Agent Schemas
# ==============================================================

from pydantic import BaseModel, Field

class SearchPlan(BaseModel):
    queries: list[str] = Field(description="A list of 2-3 specific Google search queries to gather information.")
