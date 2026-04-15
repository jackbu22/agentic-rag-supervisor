from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from agentic_rag_supervisor.paths import DATA_DIR as _DATA_DIR, FAISS_DB_ROOT as _FAISS_DB_ROOT, OUTPUT_DIR as _OUTPUT_DIR, REPO_ROOT

# ── load .env before any API clients ────────────────────────────────────────
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Tavily 웹 검색 (설치: pip install langchain-community tavily-python)
try:
    from langchain_community.tools.tavily_search import TavilySearchResults

    _TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
    TAVILY = TavilySearchResults(max_results=5, api_key=_TAVILY_KEY) if _TAVILY_KEY else None
    if TAVILY:
        print("[INFO] Tavily web search: enabled")
    else:
        print("[WARN] TAVILY_API_KEY not set — web agent will fall back to FAISS-only search")
except Exception as _tv_err:
    TAVILY = None
    print(f"[WARN] Tavily unavailable ({_tv_err.__class__.__name__}) — web agent will use FAISS fallback")


# ── LLM clients (Section 4) ──────────────────────────────────────────────────
_api_key = os.environ.get("OPENAI_API_KEY", "")

# GPT-4o-mini: Supervisor (SR), Web query generation (W1), Web evidence eval (W3)
LLM_MINI = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=_api_key)

# GPT-4o: Analysis (A1, A2), Draft writing (D1, D2), PDF validation (P1, P2)
LLM_FULL = ChatOpenAI(model="gpt-4o", temperature=0, api_key=_api_key)

# GPT-4o Vision: VLM for PDF chart/diagram extraction (R3)
LLM_VLM = ChatOpenAI(model="gpt-4o", temperature=0, api_key=_api_key, max_tokens=4096)


# ── Embeddings: bge-m3 (R2) → OpenAI fallback ───────────────────────────────
EMBED_MODEL_NAME = "unknown"
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    EMBED_MODEL = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    EMBED_MODEL_NAME = "bge-m3"
    print("[INFO] Embeddings: BAAI/bge-m3")
except Exception as _emb_err:
    EMBED_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", api_key=_api_key)
    EMBED_MODEL_NAME = "text-embedding-3-small"
    print(f"[INFO] bge-m3 unavailable ({_emb_err.__class__.__name__}), using text-embedding-3-small")


# ── Paths and static configuration ──────────────────────────────────────────
ROOT_DIR = REPO_ROOT
DATA_DIR = _DATA_DIR
OUTPUT_DIR = _OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES_FILE = DATA_DIR / "demo_semiconductor_sources.json"
QA_FILE = DATA_DIR / "demo_qa_set.json"

TARGET_COMPANIES = ["Samsung", "SK hynix", "Intel", "NVIDIA"]

COMPANY_SLUG: Dict[str, str] = {
    "Samsung": "samsung",
    "SK hynix": "skhynix",
    "Intel": "intel",
    "NVIDIA": "nvidia",
}

FAISS_DB_ROOT = _FAISS_DB_ROOT
FAISS_INDEX_PATH = FAISS_DB_ROOT / "merged_index"
COMPANY_FAISS_PATHS: Dict[str, Path] = {co: FAISS_DB_ROOT / f"{slug}_index" for co, slug in COMPANY_SLUG.items()}
WEB_CACHE_FAISS_PATH = FAISS_DB_ROOT / "web_cache_index"

WEB_CACHE_DIR = DATA_DIR / "web_cache"

RECENCY_WINDOW_DAYS = {"HBM4": 180, "PIM": 365, "CXL": 365, "STANDARD": 730}

DEFAULT_THRESHOLDS = {"rag": 0.72, "web": 0.55, "analysis": 0.70, "draft": 0.72, "pdf": 0.75}

DEFAULT_MAX_RETRIES = {"rag": 2, "web": 2, "analysis": 2, "draft": 2, "pdf": 1}

