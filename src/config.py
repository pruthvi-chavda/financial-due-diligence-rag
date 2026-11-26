import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_PRIMARY_MODEL = os.getenv("OLLAMA_PRIMARY_MODEL", "qwen3.5:4b")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_PRIMARY_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financial_docs")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RETRIEVAL_TOP_K = 15
RERANK_TOP_N = 5

DOCLING_DO_OCR = os.getenv("DOCLING_DO_OCR", "true").lower() == "true"
DOCLING_FORCE_FULL_PAGE_OCR = os.getenv("DOCLING_FORCE_FULL_PAGE_OCR", "true").lower() == "true"
DOCLING_DO_TABLE_STRUCTURE = os.getenv("DOCLING_DO_TABLE_STRUCTURE", "true").lower() == "true"

API_URL = os.getenv("API_URL", "http://localhost:8000")

chunk_size = 1000
chunk_overlap = 150


def get_model_name(role: str = "qa", provider: str | None = None) -> str:
    """
    Returns the right model name based on active provider and task role.
    - role="fast" -> lightweight model (table summaries, fast fallback)
    - role="qa"   -> primary reasoning model for final financial synthesis
    """
    active_provider = (provider or LLM_PROVIDER).lower()
    use_fast = role == "fast"
    if active_provider == "groq":
        return GROQ_FALLBACK_MODEL if use_fast else GROQ_PRIMARY_MODEL
    return OLLAMA_FALLBACK_MODEL if use_fast else OLLAMA_PRIMARY_MODEL


if __name__ == "__main__":
    print(f"Base Dir          : {BASE_DIR}")
    print(f"Default Provider  : {LLM_PROVIDER}")
    print(f"Active QA Model   : {get_model_name('qa')}")
    print(f"Groq QA Model     : {get_model_name('qa', provider='groq')}")
    print(f"Groq Fast Model   : {get_model_name('fast', provider='groq')}")
    print(f"Ollama QA Model   : {get_model_name('qa', provider='ollama')}")
    print(f"Ollama Fast Model : {get_model_name('fast', provider='ollama')}")
    print(f"Qdrant Path       : {QDRANT_URL}")
    print(f"Docling OCR       : {DOCLING_DO_OCR} | Tables: {DOCLING_DO_TABLE_STRUCTURE}")