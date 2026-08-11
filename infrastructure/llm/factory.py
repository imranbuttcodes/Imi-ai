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
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from core.config import settings
import os
from dotenv import load_dotenv


load_dotenv()


def get_llm(role: str = "default", model: str | None = None, provider: str | None = None) -> BaseChatModel:
    """
    Returns a LangChain-compatible chat model instance based on its role in the system.
    """
    # Pure Speed & Smarter JSON classification
    if role == "router":
        # return ChatGroq(
        #     model="llama-3.3-70b-versatile",   # 
        #     api_key=os.getenv('GROQ_API_KEY', 'MISSING_KEY')
        # )

        # hf_llm = HuggingFaceEndpoint(
        #     repo_id="Qwen/Qwen2.5-7B-Instruct",
        #     task="text-generation",
        #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        # )
        # return ChatHuggingFace(llm=hf_llm)
          
        #  return ChatOpenAI(
        #              model="openrouter/free",
        #              base_url="https://openrouter.ai/api/v1",
        #              api_key=os.getenv('OPENROUTER_API_KEY', 'MISSING_KEY')
        #          )


        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            api_key = os.getenv("DEEPSEEK_API_KEY"),
            model="deepseek-chat",
            temperature=0
        )
        
    # Heavy Tool Calling & Coding
    elif role == "specialist":
        # hf_llm = HuggingFaceEndpoint(
        #     repo_id="Qwen/Qwen2.5-7B-Instruct",
        #     task="text-generation",
        #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        # )
        # return ChatHuggingFace(llm=hf_llm)
        # or use this 

        # return ChatOpenAI(
        #     model="meta-llama/llama-4-maverick",
        #     api_key=os.getenv("OPENROUTER_API_KEY"),
        #     base_url="https://openrouter.ai/api/v1",
        # )

        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            api_key = os.getenv("DEEPSEEK_API_KEY"),
            model="deepseek-chat",
            temperature=0
        )

    # 3. CRAG & Self-RAG Evaluators (0.20s) - Extremely logical and fast document grading
    elif role == "evaluator":
        # return ChatGroq(
        #     model="qwen/qwen3.6-27b",
        #     api_key=os.getenv('GROQ_API_KEY', 'MISSING_KEY')
        # )

        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            api_key = os.getenv("DEEPSEEK_API_KEY"),
            model="deepseek-chat",
            temperature=0
        )
        
    # 4. Security Guardrails (0.22s) - Checks for prompt injection/jailbreaks
    elif role == "guardrail":
        return ChatGroq(
            model="meta-llama/llama-prompt-guard-2-86m",
            api_key=os.getenv('GROQ_API_KEY', 'MISSING_KEY')
        )

    # 5. Background Memory Summarizer
    elif role == "memory":
        # hf_llm = HuggingFaceEndpoint(
        #     repo_id="Qwen/Qwen2.5-7B-Instruct",
        #     task="text-generation",
        #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        # )
        # return ChatHuggingFace(llm=hf_llm)

        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            api_key = os.getenv("DEEPSEEK_API_KEY"),
            model="deepseek-chat",
            temperature=0
        )
        
    # 6. Fallback (For manual Overrides)
    _model = model or settings.default_model
    if ":" in _model:
        return init_chat_model(_model)
    return init_chat_model(_model, model_provider=provider or settings.default_provider)