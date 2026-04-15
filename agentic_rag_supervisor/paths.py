from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # `agentic_rag_supervisor/paths.py` -> repo root is one level up.
    return Path(__file__).resolve().parents[1]


REPO_ROOT = repo_root()
DATA_DIR = REPO_ROOT / "data"
FAISS_DB_ROOT = REPO_ROOT / "faiss_db"
OUTPUT_DIR = REPO_ROOT / "outputs"

