# ==============================================================
# infrastructure/llm/factory.py — Provider-Agnostic LLM Factory
# ==============================================================
# Agents NEVER import ChatGroq or ChatOpenAI directly.
# They always call get_llm() from here.
#
# Uses LangChain's init_chat_model — pass any model string
# and it auto-detects the provider.
#
# Usage:
#   from infrastructure.llm.factory import get_llm
#
#   get_llm()                                   # default from .env
#   get_llm("llama-3.3-70b-versatile")          # Groq (auto-detected)
#   get_llm("gpt-4o")                           # OpenAI (auto-detected)
#   get_llm("gemini-2.0-flash")                 # Google (auto-detected)
#   get_llm("groq:llama-3.1-8b-instant")        # Explicit provider prefix
# ==============================================================

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from core.config import settings

def get_llm(model: str | None = None, provider: str | None = None) -> BaseChatModel:
    """
    Returns a LangChain-compatible chat model instance.
    """
    _model = model or settings.default_model

    if ":" in _model:
        return init_chat_model(_model)

    _provider = provider or settings.default_provider
    return init_chat_model(
        _model, 
        model_provider=_provider
    )