# ==============================================================
# core/config.py — Application Configuration
# ==============================================================
# Pydantic Settings reads from the .env file automatically.
# All API keys and config values live here.
# Import the global `settings` object anywhere you need a key.
#
# Usage:
#   from core.config import settings
#   api_key = settings.groq_api_key
# ==============================================================

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Ensure .env is actually loaded into os.environ for LangChain
load_dotenv()


class Settings(BaseSettings):
    # --- LLM Providers ---
    groq_api_key:    str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key:  str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key:  str = Field(default="", alias="GOOGLE_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    # --- Default Model Selection ---
    default_provider: str = Field(default="groq",                    alias="DEFAULT_PROVIDER")
    default_model:    str = Field(default="llama-3.3-70b-versatile",  alias="DEFAULT_MODEL")
    eval_model:       str = Field(default="llama-3.3-70b-versatile",  alias="EVAL_MODEL")      # fast model for grading
    gen_model:        str = Field(default="llama-3.3-70b-versatile",  alias="GEN_MODEL")       # powerful model for answers

    # --- Embedding Settings ---
    embedding_provider: str = Field(default="cohere",               alias="EMBEDDING_PROVIDER")
    cohere_api_key:     str = Field(default="",                     alias="COHERE_API_KEY")

    # --- RAG Settings ---
    rag_strategy:   str = Field(default="crag",                     alias="RAG_STRATEGY")      # "basic", "crag", "self_rag"
    chroma_db_path: str = Field(default="./data/chroma",            alias="CHROMA_DB_PATH")
    retrieval_k:    int = Field(default=4,                          alias="RETRIEVAL_K")
    tavily_api_key: str = Field(default="",                         alias="TAVILY_API_KEY")

    # --- App Settings ---
    app_name:    str = Field(default="Imi AI", alias="APP_NAME")
    debug:      bool = Field(default=False,      alias="DEBUG")
    memory_dir:  str = Field(default="data/memory", alias="MEMORY_DIR")

    # --- MCP Settings ---
    mcp_filesystem_allowed_dirs: str = Field(default="D:\\", alias="MCP_FILESYSTEM_ALLOWED_DIRS")

    class Config:
        env_file         = ".env"
        env_file_encoding = "utf-8"
        extra            = "ignore"   # silently ignore unknown keys in .env


# Global singleton — import this everywhere
settings = Settings()
