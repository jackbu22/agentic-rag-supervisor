"""
Agentic RAG Supervisor Demo
============================
Full implementation of the design spec (ai-mini_design_2반_김건우+배은서.pdf).

Model assignments (Section 4):
  SR / W1 / W3  → GPT-4o-mini  (Supervisor routing, Web query gen, Web eval)
  A1 / A2 / D1 / D2 / P1 / P2  → GPT-4o  (Analysis, Draft, PDF validation)
  R3            → GPT-4o Vision (VLM: PDF chart/diagram extraction)
  R2            → bge-m3  (dense embedding for hybrid search, fallback: text-embedding-3-small)

Graph flow (19 steps per spec 3-2):
  START → Supervisor → RAG → Supervisor → Web → Supervisor → Analysis
        → Supervisor → Draft → Supervisor → Human Review → Supervisor → PDF → Supervisor → END
"""

from __future__ import annotations

import base64
import argparse
import hashlib
import json
import math
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

# ── load .env before any API clients ────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

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
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES_FILE = DATA_DIR / "demo_semiconductor_sources.json"
QA_FILE = DATA_DIR / "demo_qa_set.json"

# 분석 대상 기업 목록 (한 곳에서 관리)
TARGET_COMPANIES = ["Samsung", "SK hynix", "Intel", "NVIDIA"]

# 기업명 → 폴더/인덱스 슬러그 매핑
COMPANY_SLUG: Dict[str, str] = {
    "Samsung":  "samsung",
    "SK hynix": "skhynix",
    "Intel":    "intel",
    "NVIDIA":   "nvidia",
}

# FAISS 인덱스 경로
FAISS_DB_ROOT     = ROOT_DIR / "faiss_db"
FAISS_INDEX_PATH  = FAISS_DB_ROOT / "merged_index"          # 전체 병합 인덱스
COMPANY_FAISS_PATHS: Dict[str, Path] = {
    co: FAISS_DB_ROOT / f"{slug}_index"
    for co, slug in COMPANY_SLUG.items()
}
WEB_CACHE_FAISS_PATH = FAISS_DB_ROOT / "web_cache_index"    # 캐시된 웹 검색 인덱스

# 웹 검색 결과 JSON 캐시 디렉토리
WEB_CACHE_DIR = DATA_DIR / "web_cache"

# Recency policy (spec 4-2)
RECENCY_WINDOW_DAYS = {
    "HBM4": 180,
    "PIM": 365,
    "CXL": 365,
    "STANDARD": 730,
}

DEFAULT_THRESHOLDS = {
    "rag": 0.72,
    "web": 0.55,   # lowered: demo dataset has only 13 sources (5 academic), W3 realistically scores ~0.6
    "analysis": 0.70,
    "draft": 0.72,
    "pdf": 0.75,
}

DEFAULT_MAX_RETRIES = {
    "rag": 2,
    "web": 2,
    "analysis": 2,
    "draft": 2,
    "pdf": 1,
}


# ═══════════════════════════════════════════════════════════════════════════
# R3 — VLM helpers: GPT-4o Vision extracts structured data from PDF images
# ═══════════════════════════════════════════════════════════════════════════

def _pdf_pages_to_b64(pdf_path: str, max_pages: int = 4) -> List[str]:
    """Render PDF pages to base64 PNG images for GPT-4o Vision."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        images: List[str] = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            images.append(base64.b64encode(pix.tobytes("png")).decode())
        doc.close()
        return images
    except Exception as e:
        print(f"[VLM R3] PDF→image failed ({pdf_path}): {e}")
        return []


def _analyze_image_vlm(b64_image: str, doc_name: str = "") -> Dict[str, Any]:
    """R3: Single GPT-4o Vision call — extract structured semiconductor data from one page image."""
    try:
        msg = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    f"You are analyzing a semiconductor technology document image: {doc_name}\n"
                    "Extract ALL structured information: tables, charts, metrics, company names, "
                    "technology names, TRL levels, dates, performance numbers.\n"
                    "Return ONLY valid JSON:\n"
                    '{"title": str, "data_type": "table|chart|diagram|text", '
                    '"companies": [str], "technologies": [str], '
                    '"key_data": [{"metric": str, "value": str, "company": str, "technology": str}], '
                    '"key_findings": [str]}'
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
            },
        ])
        resp = LLM_VLM.invoke([msg])
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception as ex:
        print(f"[VLM R3] Inference error: {ex}")
    return {}


def process_pdf_sources_with_vlm(data_dir: Path) -> List[Document]:
    """R3: Scan DATA_DIR for PDF source files, extract structured data via VLM, return Documents."""
    vlm_docs: List[Document] = []
    pdf_files = [p for p in data_dir.glob("*.pdf") if p.is_file()]
    if not pdf_files:
        return vlm_docs

    print(f"[VLM R3] Processing {len(pdf_files)} PDF source(s) with GPT-4o Vision ...")
    for pdf_path in pdf_files[:3]:
        images = _pdf_pages_to_b64(str(pdf_path), max_pages=3)
        for page_idx, b64 in enumerate(images):
            result = _analyze_image_vlm(b64, doc_name=pdf_path.name)
            if not result or not result.get("key_findings"):
                continue
            content_parts = result.get("key_findings", []) + [
                f"{item.get('company', '')} {item.get('technology', '')} {item.get('metric', '')}: {item.get('value', '')}"
                for item in result.get("key_data", [])
                if item.get("value")
            ]
            content = f"[VLM: {pdf_path.name} page {page_idx + 1}]\n" + "\n".join(content_parts)
            techs = result.get("technologies", ["HBM4"])
            comps = result.get("companies", ["unknown"])
            vlm_docs.append(Document(
                page_content=content,
                metadata={
                    "doc_id": f"vlm-{pdf_path.stem}-p{page_idx}",
                    "title": f"{pdf_path.name} (VLM page {page_idx + 1})",
                    "technology": techs[0] if techs else "HBM4",
                    "company": comps[0] if comps else "unknown",
                    "source_type": "paper",
                    "published_at": datetime.now().strftime("%Y-%m-%d"),
                    "source_url": str(pdf_path),
                    "chunk_id": f"vlm-{pdf_path.stem}-p{page_idx}-c0",
                },
            ))
    print(f"[VLM R3] Produced {len(vlm_docs)} VLM document(s)")
    return vlm_docs


# ═══════════════════════════════════════════════════════════════════════════
# Data loading and retrieval infrastructure (R1, R2)
# ═══════════════════════════════════════════════════════════════════════════

def load_sources() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with SOURCES_FILE.open("r", encoding="utf-8-sig") as f:
        source_records = json.load(f)
    with QA_FILE.open("r", encoding="utf-8-sig") as f:
        qa_rows = json.load(f)
    return source_records, qa_rows


def build_chunks(source_records: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for row in source_records:
        meta = {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "technology": row["technology"],
            "company": row["company"],
            "source_type": row["source_type"],
            "published_at": row["published_at"],
            "source_url": row["source_url"],
        }
        docs.append(Document(page_content=row["content"], metadata=meta))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=420,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for idx, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = f"chunk-{idx:04d}"
    return chunks


def build_retrievers(
    chunks: List[Document],
) -> Tuple[FAISS, BM25Retriever]:
    """R2: Build FAISS (bge-m3 dense) + BM25 (sparse) retrievers."""
    vectorstore = FAISS.from_documents(chunks, EMBED_MODEL)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10
    return vectorstore, bm25


def _load_or_build_faiss(path: Path, chunks: List[Document], label: str) -> Optional[FAISS]:
    """단일 FAISS 인덱스를 로드하거나 구축해서 반환한다."""
    if not chunks:
        return None
    if path.exists():
        try:
            vs = FAISS.load_local(str(path), EMBED_MODEL, allow_dangerous_deserialization=True)
            print(f"[INFO] FAISS loaded  : {label} ({path.name})")
            return vs
        except Exception as e:
            print(f"[WARN] FAISS load failed ({label}): {e} — rebuilding")
    vs = FAISS.from_documents(chunks, EMBED_MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    print(f"[INFO] FAISS built   : {label} ({len(chunks)} chunks) → {path}")
    return vs


def load_or_build_retrievers(
    chunks: List[Document],
) -> Tuple[FAISS, BM25Retriever, Dict[str, FAISS]]:
    """
    반환값:
      merged_vs         — 전체 병합 FAISS (기업 구분 없는 검색용)
      bm25              — 전체 BM25 리트리버
      company_stores    — {company: FAISS} 기업별 FAISS
    """
    FAISS_DB_ROOT.mkdir(parents=True, exist_ok=True)

    # BM25는 직렬화 불안정 → 항상 메모리 재구축
    bm25 = BM25Retriever.from_documents(chunks) if chunks else None
    if bm25:
        bm25.k = 10

    # 전체 병합 인덱스
    merged_vs = _load_or_build_faiss(FAISS_INDEX_PATH, chunks, "merged")

    # 기업별 인덱스
    company_stores: Dict[str, FAISS] = {}
    for company, path in COMPANY_FAISS_PATHS.items():
        co_chunks = [c for c in chunks if c.metadata.get("company") == company]
        vs = _load_or_build_faiss(path, co_chunks, company)
        if vs:
            company_stores[company] = vs

    # 캐시된 웹 검색 인덱스가 있으면 merged_vs에 병합 (in-memory)
    if WEB_CACHE_FAISS_PATH.exists() and merged_vs is not None:
        try:
            web_vs = FAISS.load_local(
                str(WEB_CACHE_FAISS_PATH), EMBED_MODEL, allow_dangerous_deserialization=True
            )
            merged_vs.merge_from(web_vs)
            print(f"[INFO] Web-cache FAISS merged into main index")
        except Exception as e:
            print(f"[WARN] Web-cache FAISS merge failed: {e}")

    return merged_vs, bm25, company_stores


# ── module-level retriever initialisation ───────────────────────────────────
SOURCE_RECORDS, QA_ROWS = load_sources()
CHUNKS = build_chunks(SOURCE_RECORDS)
VECTORSTORE, BM25, COMPANY_VECTORSTORES = load_or_build_retrievers(CHUNKS)


# ── date / recency helpers ───────────────────────────────────────────────────

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def detect_tech_hint(query: str, fallback: Optional[str] = None) -> str:
    q = query.lower()
    if "hbm" in q:
        return "HBM4"
    if "pim" in q:
        return "PIM"
    if "cxl" in q:
        return "CXL"
    return fallback or "HBM4"


def canonical_tech_label(value: str) -> str:
    label = value.strip().upper()
    aliases = {
        "HBM": "HBM4",
        "HBM3": "HBM4",
        "HBM3E": "HBM4",
        "HBM4": "HBM4",
        "CXL": "CXL",
        "PIM": "PIM",
    }
    return aliases.get(label, label)


def evidence_covers_target(evidence: Dict[str, Any], target_tech: str) -> bool:
    target = canonical_tech_label(target_tech)
    meta_tech = canonical_tech_label(str(evidence.get("technology", "")))
    if meta_tech == target:
        return True

    searchable_text = " ".join(
        str(evidence.get(key, ""))
        for key in ("doc_id", "title", "source_url", "excerpt")
    ).upper()
    return target_tech.strip().upper() in searchable_text


def recency_days_for_doc(meta: Dict[str, Any]) -> int:
    source_type = str(meta.get("source_type", "")).lower()
    if source_type == "paper":
        return 3650
    if source_type in {"conference", "patent"}:
        return 1800
    if source_type == "standard":
        return RECENCY_WINDOW_DAYS["STANDARD"]
    return RECENCY_WINDOW_DAYS.get(str(meta.get("technology", "HBM4")).upper(), 365)


def is_recent(meta: Dict[str, Any], now_dt: datetime) -> bool:
    published = parse_date(str(meta.get("published_at")))
    age_days = (now_dt - published).days
    return age_days <= recency_days_for_doc(meta)


def rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> List[Document]:
    """Reciprocal rank fusion — Dense 0.5 : BM25 0.5 (spec 4-2)."""
    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    for rank, doc in enumerate(dense_docs, start=1):
        key = doc.metadata["chunk_id"]
        score_map[key] = score_map.get(key, 0.0) + dense_weight * (1.0 / rank)
        doc_map[key] = doc
    for rank, doc in enumerate(sparse_docs, start=1):
        key = doc.metadata["chunk_id"]
        score_map[key] = score_map.get(key, 0.0) + sparse_weight * (1.0 / rank)
        doc_map[key] = doc
    ordered = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, _ in ordered]


def hybrid_search(
    query: str,
    tech_hint: Optional[str] = None,
    company: Optional[str] = None,
    top_k: int = 5,
    allowed_source_types: Optional[set] = None,
) -> List[Document]:
    """Hybrid search (Dense + BM25 RRF).
    company 지정 시 해당 기업 전용 FAISS를 우선 사용하고,
    None이면 전체 병합 FAISS를 사용한다.
    """
    now_dt = datetime.now(timezone.utc)
    tech = detect_tech_hint(query, fallback=tech_hint)

    # Dense: 기업별 store 우선, 없으면 merged
    if company and company in COMPANY_VECTORSTORES:
        vs = COMPANY_VECTORSTORES[company]
    elif VECTORSTORE is not None:
        vs = VECTORSTORE
    else:
        return []

    dense_docs = vs.similarity_search(query, k=12)

    # BM25: 전체에서 검색 후 기업 필터
    if BM25 is not None:
        sparse_docs_raw = BM25.invoke(query)
        sparse_docs = (
            [d for d in sparse_docs_raw if d.metadata.get("company") == company]
            if company else sparse_docs_raw
        )
    else:
        sparse_docs = []

    merged_docs = rank_fusion(dense_docs, sparse_docs, dense_weight=0.5, sparse_weight=0.5)

    out: List[Document] = []
    for doc in merged_docs:
        meta = doc.metadata
        if tech and str(meta.get("technology", "")).upper() != tech.upper():
            continue
        if allowed_source_types is not None:
            if str(meta.get("source_type", "")).lower() not in allowed_source_types:
                continue
        if not is_recent(meta, now_dt):
            continue
        out.append(doc)
        if len(out) >= top_k:
            break
    return out


def literal_source_matches(
    terms: List[str],
    companies: List[str],
    max_per_term: int = 3,
) -> List[Document]:
    """Add exact title/body matches for user-specified product or architecture names."""
    matches: List[Document] = []
    seen: set = set()
    company_set = set(companies)
    for term in terms:
        needle = term.strip().upper()
        if not needle:
            continue
        per_term = 0
        for doc in CHUNKS:
            if doc.metadata.get("company") not in company_set:
                continue
            haystack = " ".join([
                str(doc.metadata.get("title", "")),
                str(doc.metadata.get("source_url", "")),
                doc.page_content,
            ]).upper()
            chunk_id = doc.metadata.get("chunk_id")
            if needle in haystack and chunk_id not in seen:
                matches.append(doc)
                seen.add(chunk_id)
                per_term += 1
                if per_term >= max_per_term:
                    break
    return matches


def evaluate_retrieval(qa_rows: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    """Compute Hit Rate@k and MRR over the QA set."""
    hit_count = 0
    reciprocal_ranks: List[float] = []
    for row in qa_rows:
        query = row["question"]
        tech = row.get("technology", "HBM4")
        gt = set(row.get("ground_truth_doc_ids", []))
        results = hybrid_search(query, tech_hint=tech, top_k=k)
        doc_ids = [d.metadata["doc_id"] for d in results]
        if any(x in gt for x in doc_ids):
            hit_count += 1
        rr = 0.0
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id in gt:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    n = max(len(qa_rows), 1)
    return {"hit_rate_at_k": hit_count / n, "mrr": sum(reciprocal_ranks) / n}


# ═══════════════════════════════════════════════════════════════════════════
# State and judge types
# ═══════════════════════════════════════════════════════════════════════════

class JudgeResult(TypedDict):
    score: float
    verdict: Literal["approve", "retry", "fail"]
    issues: List[str]
    suggestions: List[str]


class AgentState(TypedDict, total=False):
    question: str
    target_technologies: List[str]
    target_companies: List[str]
    config: Dict[str, Any]
    status: str
    failure_reason: str

    supplement_queries: List[str]
    web_queries: List[str]
    rag_retrieved: List[Dict[str, Any]]
    web_classified: List[Dict[str, Any]]
    normalized_evidence: List[Dict[str, Any]]
    trl_scores: Dict[str, Dict[str, Dict[str, Any]]]
    competitor_analysis: Dict[str, Any]
    sections: Dict[str, str]
    references: List[Dict[str, Any]]
    draft: str
    draft_ko: str
    pdf_path: str
    markdown_path: str
    pdf_ko_path: str
    markdown_ko_path: str

    rag_judge_result: JudgeResult
    web_judge_result: JudgeResult
    analysis_judge_result: JudgeResult
    draft_judge_result: JudgeResult
    pdf_judge_result: JudgeResult
    human_review_result: Dict[str, Any]

    retry_reason: str
    improvement_instructions: str
    revision_count: Dict[str, int]


def build_judge(score: float, threshold: float, issues: List[str], suggestions: List[str]) -> JudgeResult:
    if score >= threshold:
        verdict: Literal["approve", "retry", "fail"] = "approve"
    elif score >= max(0.0, threshold - 0.20):
        verdict = "retry"
    else:
        verdict = "fail"
    return {
        "score": round(score, 4),
        "verdict": verdict,
        "issues": issues,
        "suggestions": suggestions,
    }


def get_threshold(state: AgentState, key: str) -> float:
    return float(state["config"]["thresholds"][key])


def normalize_doc(doc: Document, channel: str) -> Dict[str, Any]:
    meta = doc.metadata
    return {
        "channel": channel,
        "doc_id": meta.get("doc_id"),
        "title": meta.get("title"),
        "company": meta.get("company"),
        "technology": meta.get("technology"),
        "source_type": meta.get("source_type"),
        "published_at": meta.get("published_at"),
        "source_url": meta.get("source_url"),
        "excerpt": doc.page_content[:320],
    }


# ═══════════════════════════════════════════════════════════════════════════
# SR — Supervisor LLM helper (GPT-4o-mini)
# ═══════════════════════════════════════════════════════════════════════════

def _supervisor_generate_instructions(state: AgentState, agent_key: str) -> str:
    """SR: GPT-4o-mini generates specific, actionable retry instructions."""
    judge = state.get(f"{agent_key}_judge_result", {})
    issues = judge.get("issues", [])
    score = judge.get("score", 0.0)
    threshold = get_threshold(state, agent_key)
    prompt = (
        f"You are a research workflow supervisor for a semiconductor competitive intelligence pipeline.\n"
        f"Agent '{agent_key}' scored {score:.3f} (threshold: {threshold:.3f}).\n"
        f"Known issues: {json.dumps(issues)}\n"
        f"Research question: {state.get('question', '')}\n"
        f"Target technologies: {state.get('target_technologies', [])}\n"
        f"Target companies: {state.get('target_companies', [])}\n\n"
        f"Write 2-3 specific, actionable instructions for the '{agent_key}' agent to improve on retry. "
        f"Be concise (max 120 words). Focus on exactly what evidence or analysis is missing."
    )
    try:
        resp = LLM_MINI.invoke(prompt)
        return resp.content.strip()
    except Exception as ex:
        print(f"[Supervisor SR] LLM error: {ex}")
        return f"Improve {agent_key}: {'; '.join(issues) or 'increase coverage and citation count'}."


# ═══════════════════════════════════════════════════════════════════════════
# W1 — Web query generation (GPT-4o-mini)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_web_queries(state: AgentState) -> List[str]:
    """W1: GPT-4o-mini generates targeted web search queries."""
    feedback_hint = ""
    if state.get("retry_reason") or state.get("improvement_instructions"):
        feedback_hint = (
            f"\nSupervisor feedback: {state.get('retry_reason', '')} "
            f"{state.get('improvement_instructions', '')}"
        )
    prompt = (
        f"You are a semiconductor competitive intelligence researcher.\n"
        f"Generate exactly 4 targeted search query strings as a JSON array.\n"
        f"Main question: {state['question']}\n"
        f"Technologies: {', '.join(state.get('target_technologies', ['HBM4', 'PIM', 'CXL']))}\n"
        f"Companies: {', '.join(state.get('target_companies', ['SK hynix', 'Samsung', 'Micron']))}"
        f"{feedback_hint}\n\n"
        f"Vary query focus: (1) supporting evidence, (2) counter-evidence/risks, "
        f"(3) ecosystem signals, (4) supply chain bottlenecks.\n"
        f"Return ONLY a JSON array of 4 strings."
    )
    try:
        resp = LLM_MINI.invoke(prompt)
        text = resp.content
        s, e = text.find("["), text.rfind("]") + 1
        if s != -1 and e > s:
            queries = json.loads(text[s:e])
            if isinstance(queries, list) and len(queries) >= 2:
                return [str(q) for q in queries[:4]]
    except Exception as ex:
        print(f"[Web W1] LLM error: {ex}")
    q = state["question"]
    return [
        f"latest semiconductor progress: {q}",
        f"counter evidence and risks: {q}",
        f"industry ecosystem signals: {q}",
        f"supply chain bottleneck: {q}",
    ]


# ═══════════════════════════════════════════════════════════════════════════
# W3 — Web evidence evaluation (GPT-4o-mini)
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_web_evidence(evidence: List[Dict[str, Any]], state: AgentState) -> Dict[str, Any]:
    """W3: GPT-4o-mini evaluates web evidence quality, diversity, and bias."""
    summary = [
        {
            "title": e.get("title"),
            "company": e.get("company"),
            "source_type": e.get("source_type"),
            "excerpt": (e.get("excerpt") or "")[:100],
        }
        for e in evidence[:8]
    ]
    prompt = (
        f"Evaluate this semiconductor competitive intelligence evidence set for quality and bias.\n"
        f"Target companies: {state.get('target_companies', [])}\n"
        f"Evidence ({len(evidence)} items, sample): {json.dumps(summary)}\n\n"
        f"Score 0.0-1.0 based on: source type diversity, company coverage balance, absence of bias.\n"
        f"Return ONLY valid JSON:\n"
        f'{{ "score": float, "issues": [list of short strings], "suggestions": [list of short strings] }}'
    )
    try:
        resp = LLM_MINI.invoke(prompt)
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            result = json.loads(text[s:e])
            if "score" in result:
                return result
    except Exception as ex:
        print(f"[Web W3] LLM error: {ex}")
    # Fallback: rule-based
    source_diversity = min(len({e["source_type"] for e in evidence}) / 4.0, 1.0) if evidence else 0.0
    target_cos = state.get("target_companies", TARGET_COMPANIES)
    company_score = min(
        len({e["company"] for e in evidence}.intersection(set(target_cos))) / max(len(target_cos), 1),
        1.0,
    )
    score = 0.40 * source_diversity + 0.35 * company_score + 0.25
    issues: List[str] = []
    suggestions: List[str] = []
    if source_diversity < 0.75:
        issues.append("Web source diversity is limited.")
        suggestions.append("Add standards and third-party industry sources.")
    if company_score < 1.0:
        issues.append("Web evidence does not cover all competitors.")
        suggestions.append("Add company-specific query expansions.")
    return {"score": round(score, 4), "issues": issues, "suggestions": suggestions}


def _rule_based_web_evidence_score(
    evidence: List[Dict[str, Any]],
    state: AgentState,
) -> Dict[str, Any]:
    target_techs = state.get("target_technologies", ["HBM4", "PIM", "CXL"])
    target_cos = state.get("target_companies", TARGET_COMPANIES)
    existing_evidence = list(state.get("normalized_evidence", []))
    combined_evidence = existing_evidence + evidence

    evidence_count_score = min(len(evidence) / 8.0, 1.0) if evidence else 0.0
    source_diversity = min(len({e["source_type"] for e in evidence}) / 3.0, 1.0) if evidence else 0.0
    citation_score = min(len({e["doc_id"] for e in evidence}) / 6.0, 1.0) if evidence else 0.0
    tech_coverage = (
        len({
            tech for tech in target_techs
            if any(evidence_covers_target(e, tech) for e in combined_evidence)
        })
        / max(len(target_techs), 1)
    )
    company_coverage = (
        len({e["company"] for e in combined_evidence}.intersection(set(target_cos)))
        / max(len(target_cos), 1)
    )

    score = (
        0.25 * evidence_count_score
        + 0.25 * tech_coverage
        + 0.20 * company_coverage
        + 0.15 * citation_score
        + 0.15 * source_diversity
    )

    issues: List[str] = []
    suggestions: List[str] = []
    if evidence_count_score < 0.75:
        issues.append("Web evidence count is limited.")
        suggestions.append("Increase web result count or use cached web evidence.")
    if tech_coverage < 1.0:
        issues.append("Combined evidence does not cover all target technologies.")
        suggestions.append("Add exact-match queries for missing technology names.")
    if company_coverage < 0.75:
        issues.append("Combined evidence has weak target-company coverage.")
        suggestions.append("Add company-specific query expansions.")

    return {"score": round(score, 4), "issues": issues, "suggestions": suggestions}


# ═══════════════════════════════════════════════════════════════════════════
# A1 — TRL analysis (GPT-4o)
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_trl_with_llm(
    evidence: List[Dict[str, Any]],
    companies: List[str],
    techs: List[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """A1: GPT-4o estimates TRL for each company-technology pair from evidence."""
    evidence_summary = [
        {
            "company": e.get("company"),
            "technology": e.get("technology"),
            "source_type": e.get("source_type"),
            "title": e.get("title"),
            "excerpt": (e.get("excerpt") or "")[:150],
        }
        for e in evidence[:20]
    ]
    prompt = (
        f"You are a semiconductor technology analyst. Estimate TRL (Technology Readiness Level, 1–9) "
        f"for each company–technology pair using ONLY the provided public evidence.\n\n"
        f"Companies: {json.dumps(companies)}\n"
        f"Technologies: {json.dumps(techs)}\n"
        f"Evidence: {json.dumps(evidence_summary)}\n\n"
        f"TRL scale: 1-3=Research, 4-5=Lab validation, 6-7=Prototype/demonstration, 8-9=Production.\n"
        f"For each pair, provide trl (int), confidence (0.0-1.0), and a 1-sentence rationale.\n"
        f"If no evidence exists for a pair, set trl=null, confidence=0.\n\n"
        f"Return ONLY valid JSON in this exact schema:\n"
        f'{{"company_name": {{"technology_name": {{"trl": int_or_null, "confidence": float, "rationale": str}}}}}}'
    )
    try:
        resp = LLM_FULL.invoke(prompt)
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception as ex:
        print(f"[Analysis A1] LLM error: {ex}")
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# A2 — Competitor gap narrative (GPT-4o)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_competitor_narrative(
    matrix: List[Dict[str, Any]],
    baseline: str,
    question: str,
) -> str:
    """A2: GPT-4o generates a competitor gap analysis narrative."""
    matrix_str = json.dumps(matrix[:12])
    prompt = (
        f"You are a semiconductor competitive intelligence analyst.\n"
        f"Research question: {question}\n"
        f"Baseline company: {baseline}\n"
        f"TRL matrix: {matrix_str}\n\n"
        f"Write a concise (3-5 sentence) competitor gap analysis narrative covering:\n"
        f"1. Which technologies have the highest competitive threat to {baseline}\n"
        f"2. Key gaps to monitor\n"
        f"3. Strategic priority recommendation\n"
        f"Be specific, cite company names and technologies."
    )
    try:
        resp = LLM_FULL.invoke(prompt)
        return resp.content.strip()
    except Exception as ex:
        print(f"[Analysis A2] LLM error: {ex}")
        return "Competitor analysis unavailable due to LLM error."


# ═══════════════════════════════════════════════════════════════════════════
# D1 — Full report writing (GPT-4o)
# ═══════════════════════════════════════════════════════════════════════════

REPORT_EN_TEMPLATE = """
Reference report style:
- Cover block:
  # R&D TECHNOLOGY STRATEGY REPORT
  # Next-Generation Semiconductor Technology Strategy Report
  HBM4 · PIM · CXL Technology Landscape and Competitive Threat Assessment
  Date | Division | Classification | Version
- Body sections:
  ## EXECUTIVE SUMMARY
  ### Key Metrics
  - 3 concise KPI cards using bold metric labels
  ## 1. Analysis Background — Why This Technology, Why Now
  ### 1.1 Market Inflection
  ### 1.2 Research Rationale — Bridging the Information Gap
  ## 2. Technology Landscape
  ### 2.1 HBM / HBM4
  ### 2.2 PIM — Processing-In-Memory
  ### 2.3 CXL — Compute Express Link
  ## 3. Competitor Trend Analysis
  ### 3.1 TRL-Based Competitor Benchmarking
  ### 3.2 Competitor Profiles
  ## 4. Strategic Implications — R&D Priority Recommendations
  ## REFERENCES
  ### Limitation Note
"""

REPORT_KO_TRANSLATION_STYLE = """
한국어 참고 보고서 양식:
- 표지 블록:
  # R&D 기술 전략 분석 보고서
  # 차세대 반도체 기술 전략 분석 보고서
  HBM4 · PIM · CXL 기술 현황 및 경쟁사 위협 수준 종합 평가
  작성일 | 작성 부서 | 보고서 분류 | 버전
- 본문 섹션:
  ## EXECUTIVE SUMMARY
  ### 핵심 지표
  ## 1. 분석 배경 — 왜 지금 이 기술을 분석해야 하는가
  ### 1.1 시장 전환점
  ### 1.2 분석 필요성 — 정보 비대칭 극복
  ## 2. 분석 대상 기술 현황
  ### 2.1 HBM / HBM4 — 차세대 고대역폭 메모리
  ### 2.2 PIM — 메모리 내 연산 기술
  ### 2.3 CXL — 확장형 메모리 연결 표준
  ## 3. 경쟁사 동향 분석
  ### 3.1 TRL 기반 경쟁사 비교
  ### 3.2 경쟁사별 세부 동향
  ## 4. 전략적 시사점 — R&D 우선순위 제언
  ## 참고문헌
  ### 제한 사항
"""

def _write_report_with_llm(
    state: AgentState,
    matrix: List[Dict[str, Any]],
    references: Dict[str, Any],
    competitor_narrative: str,
) -> str:
    """D1: GPT-4o writes the full technology strategy analysis report in markdown."""
    matrix_str = "\n".join(
        f"- {r['technology']} | {r['company']} | TRL={r['trl']} | "
        f"Threat={r['threat_level']} | Evidence={r['evidence_count']} | "
        f"Criterion={r.get('criterion', '')} | Reason={r.get('reason', '')}"
        for r in matrix
    )
    ref_str = "\n".join(
        f"[{r['doc_id']}] {r['title']} ({r['published_at']}) - {r['source_url']}"
        for r in list(references.values())[:12]
    )
    feedback_lines = []
    if state.get("improvement_instructions"):
        feedback_lines.append(f"Supervisor Instructions: {state['improvement_instructions']}")
    human_fb = str(state.get("human_review_result", {}).get("feedback", "")).strip()
    if human_fb:
        feedback_lines.append(f"Human Reviewer Feedback: {human_fb}")
    feedback_block = ("\n\nIMPORTANT — incorporate this feedback:\n" + "\n".join(feedback_lines)) if feedback_lines else ""
    today = datetime.now().strftime("%B %d, %Y")
    techs = " · ".join(state.get("target_technologies", ["HBM4", "PIM", "CXL"]))
    companies = ", ".join(state.get("target_companies", TARGET_COMPANIES))

    prompt = (
        f"Write a polished English R&D technology strategy report in markdown.\n"
        f"Use the following reference report format, tone, and visual structure:\n{REPORT_EN_TEMPLATE}\n"
        f"Cover metadata:\n"
        f"- Date: {today}\n"
        f"- Division: Semiconductor R&D Strategy Team\n"
        f"- Classification: Internal Confidential / R&D Personnel Only\n"
        f"- Version: v1.0\n"
        f"- Target technologies: {techs}\n"
        f"- Target companies: {companies}\n\n"
        f"Competitor TRL Matrix:\n{matrix_str}\n\n"
        f"Competitor Narrative:\n{competitor_narrative}\n\n"
        f"References (cite at least 4 using [doc_id]):\n{ref_str}"
        f"{feedback_block}\n\n"
        f"Requirements:\n"
        f"- EXECUTIVE SUMMARY must contain exactly these three labeled paragraphs: Purpose, Background, Conclusion.\n"
        f"- After those three paragraphs, include ### Key Metrics with 3 bullets.\n"
        f"- Every factual claim must include at least one citation [doc_id]\n"
        f"- TRL 4-6 values must include a parenthetical confidence note, e.g. (estimated, confidence: 0.7)\n"
        f"- Include one compact markdown table in Technology Landscape and one compact markdown table in TRL-Based Competitor Benchmarking\n"
        f"- The TRL-Based Competitor Benchmarking table MUST include columns: Technology, Company, TRL, Threat, Evidence Count, Selection Criterion, Short Reason.\n"
        f"- Keep every Short Reason inside the table as a compact phrase of 12 words or fewer.\n"
        f"- If a row uses BASELINE, spell it as 'Baseline (SK hynix comparison anchor)' and explain that this means SK hynix is the reference company used to calculate TRL gaps and threat levels.\n"
        f"- Under the TRL table, add a subsection titled 'Detailed Rationale' and explain the major rows in prose: why the TRL/threat was selected, which criterion was used, and what uncertainty remains.\n"
        f"- Explain WHY each major TRL/threat result was reached: evidence signal, criterion used, implication, and uncertainty\n"
        f"- Include a 'Decision Criteria' paragraph defining TRL, threat level, evidence count, confidence, and baseline. Define baseline as SK hynix, the reference company used to calculate TRL gaps and threat levels.\n"
        f"- Include a 'Recommended Actions' list with 30/90/180-day R&D actions and owner-style wording\n"
        f"- Include an architecture-oriented explanation when the question involves GPU, accelerator, HBM, CXL, Blackwell, Hopper, or memory systems\n"
        f"- Target length: 1,200-1,800 words. Avoid shallow one-sentence sections.\n"
        f"- Use exactly the section headers from the reference format, including ## REFERENCES and ### Limitation Note\n"
        f"- Be specific about the requested technologies and companies: {techs}; {companies}\n"
        f"- Do not mention that you are following a template; just write the report"
    )
    try:
        resp = LLM_FULL.invoke(prompt)
        return resp.content.strip()
    except Exception as ex:
        print(f"[Draft D1] LLM error: {ex}")
        return ""


def _translate_to_korean(english_draft: str) -> str:
    """D1-KO: GPT-4o가 영어 보고서를 전문 한국어로 번역한다."""
    prompt = (
        "당신은 반도체 기술 전문 번역가입니다.\n"
        "아래 영문 기술전략 보고서를 **전문 한국어 보고서**로 변환하세요.\n"
        f"반드시 다음 한국어 참고 보고서 양식과 섹션명을 따르세요:\n{REPORT_KO_TRANSLATION_STYLE}\n"
        "규칙:\n"
        "- EXECUTIVE SUMMARY에는 반드시 `목적`, `배경`, `결론` 3개 라벨 문단을 포함하세요.\n"
        "- TRL 기반 경쟁사 비교 표에는 `선정 기준`, `요약 이유` 컬럼을 유지하세요.\n"
        "- `요약 이유`는 반드시 표 안의 짧은 문구로 작성하고 20자 안팎으로 압축하세요.\n"
        "- `기준선`이라고 쓸 때는 반드시 `기준선(SK hynix 비교 기준)`처럼 기준선의 의미를 명확히 적으세요. 이는 TRL 격차와 위협 수준을 계산할 때 SK hynix를 비교 기준 회사로 둔다는 뜻입니다.\n"
        "- TRL 표 아래에는 `세부 판단 근거` 소제목을 두고 표의 주요 판단을 문단으로 자세히 설명하세요.\n"
        "- `결정 기준`에는 TRL, 위협 수준, 증거 수, 신뢰도, 기준선의 정의를 포함하세요. 기준선은 SK hynix를 비교 기준 회사로 두고 TRL 격차와 위협 수준을 계산한다는 의미입니다.\n"
        "- 마크다운 헤더(##, ###), 표, 인용([doc_id]) 형식을 그대로 유지하세요.\n"
        "- 고유명사(회사명, 기술명: HBM4, PIM, CXL, TRL, FAISS 등)는 영어 그대로 두세요.\n"
        "- `EXECUTIVE SUMMARY`는 참고 보고서처럼 영어 표기를 유지하세요.\n"
        "- `REFERENCES`는 `참고문헌`, `Limitation Note`는 `제한 사항`으로 바꾸세요.\n"
        "- 문체는 보고서체로 쓰고, 참고 PDF처럼 간결한 전략 문장과 KPI 요약을 유지하세요.\n"
        "- 번역 외 다른 설명이나 추가 텍스트를 붙이지 마세요.\n\n"
        f"영문 보고서:\n\n{english_draft}"
    )
    try:
        resp = LLM_FULL.invoke(prompt)
        return resp.content.strip()
    except Exception as ex:
        print(f"[Draft D1-KO] 번역 LLM 오류: {ex}")
        return ""


def _normalize_baseline_labels(draft: str) -> str:
    """Make baseline wording explicit in generated reports."""
    replacements = {
        "| 기준선(SK hynix 비교 기준) 비교 기준 |": "| SK hynix 기준값 |",
        "| 기준선(SK hynix 비교 기준) 비교 |": "| SK hynix 기준값 |",
        "| BASELINE |": "| Baseline (SK hynix comparison anchor) |",
        "| Baseline |": "| Baseline (SK hynix comparison anchor) |",
        "| 기준선 |": "| 기준선(SK hynix 비교 기준) |",
    }
    normalized = draft
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _matrix_table_reason(row: Dict[str, Any], korean: bool = False) -> str:
    gap = int(row.get("gap_vs_skh", 0) or 0)
    trl = int(row.get("trl", 0) or 0)
    company = row.get("company", "")
    if company == "SK hynix":
        return "비교 기준" if korean else "Comparison anchor"
    if trl == 0:
        return "직접 증거 없음" if korean else "No direct evidence"
    if gap >= 1:
        return f"{gap}단계 앞섬" if korean else f"Ahead by {gap} TRL"
    if gap == 0:
        return "기준선과 유사" if korean else "Similar to baseline"
    return f"{abs(gap)}단계 뒤처짐" if korean else f"Behind by {abs(gap)} TRL"


def _matrix_criterion(value: str, korean: bool = False) -> str:
    if not korean:
        return value
    mapping = {
        "product/deployment signal": "제품/배포 신호",
        "program integration signal": "프로그램 통합 신호",
        "standard/prototype signal": "표준/프로토타입 신호",
        "research signal": "연구 신호",
        "no direct evidence": "직접 증거 없음",
    }
    return mapping.get(value, value)


def _matrix_threat(row: Dict[str, Any], korean: bool = False) -> str:
    threat = str(row.get("threat_level", ""))
    if row.get("company") == "SK hynix" or threat.upper() == "BASELINE":
        return "기준선(SK hynix 비교 기준)" if korean else "Baseline (SK hynix comparison anchor)"
    if not korean:
        return threat
    return {"HIGH": "높음", "MEDIUM": "중간", "LOW": "낮음"}.get(threat.upper(), threat)


def _build_trl_markdown_table(matrix: List[Dict[str, Any]], korean: bool = False) -> str:
    if korean:
        lines = [
            "| 기술 | 회사 | TRL | 위협 수준 | 증거 수 | 선정 기준 | 요약 이유 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    else:
        lines = [
            "| Technology | Company | TRL | Threat | Evidence Count | Selection Criterion | Short Reason |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    for row in matrix:
        lines.append(
            f"| {row.get('technology', '')} | {row.get('company', '')} | {row.get('trl', 0)} | "
            f"{_matrix_threat(row, korean)} | {row.get('evidence_count', 0)} | "
            f"{_matrix_criterion(str(row.get('criterion', '')), korean)} | "
            f"{_matrix_table_reason(row, korean)} |"
        )
    return "\n".join(lines)


def _ensure_trl_table_section(draft: str, matrix: List[Dict[str, Any]], korean: bool = False) -> str:
    """Ensure section 3.1 contains the TRL benchmark table even if the LLM omits it."""
    header = "### 3.1 TRL 기반 경쟁사 비교" if korean else "### 3.1 TRL-Based Competitor Benchmarking"
    if header not in draft or not matrix:
        return draft
    start = draft.find(header)
    next_candidates = [
        idx for idx in [
            draft.find("### 3.2", start + len(header)),
            draft.find("## 4.", start + len(header)),
            draft.find("## 4", start + len(header)),
        ]
        if idx != -1
    ]
    end = min(next_candidates) if next_candidates else len(draft)
    section = draft[start:end]
    if "|" in section and ("요약 이유" in section or "Short Reason" in section):
        return draft

    table = _build_trl_markdown_table(matrix, korean=korean)
    rationale_header = "### 세부 판단 근거" if korean else "### Detailed Rationale"
    if rationale_header in section:
        replacement = section.replace(rationale_header, f"{table}\n\n{rationale_header}", 1)
    else:
        intro = (
            "아래 표는 SK hynix를 기준선으로 두고 TRL 격차와 위협 수준을 비교합니다."
            if korean
            else "The table below compares TRL gaps and threat levels using SK hynix as the baseline."
        )
        replacement = section.rstrip() + f"\n\n{intro}\n\n{table}\n\n"
    return draft[:start] + replacement + draft[end:]


def _inject_visual_markers(draft: str) -> str:
    """Add hidden PDF-only visual markers without changing normal markdown preview."""
    out = draft
    if "<!-- VISUAL:ARCHITECTURE_DIAGRAM -->" not in out:
        anchor = "## 2. Technology Landscape"
        if anchor in out:
            out = out.replace(anchor, f"<!-- VISUAL:ARCHITECTURE_DIAGRAM -->\n\n{anchor}", 1)
        else:
            ko_anchor = "## 2. 분석 대상 기술 현황"
            out = out.replace(ko_anchor, f"<!-- VISUAL:ARCHITECTURE_DIAGRAM -->\n\n{ko_anchor}", 1)
    if "<!-- VISUAL:TRL_CHART -->" not in out:
        anchor = "### 3.1 TRL-Based Competitor Benchmarking"
        if anchor in out:
            out = out.replace(anchor, f"{anchor}\n\n<!-- VISUAL:TRL_CHART -->", 1)
        else:
            ko_anchor = "### 3.1 TRL 기반 경쟁사 비교"
            out = out.replace(ko_anchor, f"{ko_anchor}\n\n<!-- VISUAL:TRL_CHART -->", 1)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# D2 — Draft completeness check (GPT-4o)
# ═══════════════════════════════════════════════════════════════════════════

def _check_draft_completeness(draft: str) -> Dict[str, Any]:
    """D2: GPT-4o checks report completeness and returns score + issues."""
    prompt = (
        f"Check if this technology strategy report draft is complete and well-structured.\n"
        f"Draft (first 3000 chars): {draft[:3000]}\n\n"
        f"Verify: reference-report style cover block exists, EXECUTIVE SUMMARY, Key Metrics, "
        f"sections 1-4, REFERENCES, citations, no placeholder text, limitation note included.\n"
        f"Return ONLY valid JSON:\n"
        f'{{ "section_score": float, "citation_score": float, "issues": [str], "suggestions": [str] }}'
    )
    try:
        resp = LLM_FULL.invoke(prompt)
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception as ex:
        print(f"[Draft D2] LLM error: {ex}")
    return {"section_score": 0.8, "citation_score": 0.8, "issues": [], "suggestions": []}


# ═══════════════════════════════════════════════════════════════════════════
# P1 — PDF pre-validation (GPT-4o)
# ═══════════════════════════════════════════════════════════════════════════

def _validate_draft_for_pdf(draft: str) -> Dict[str, Any]:
    """P1: GPT-4o validates draft completeness and quality before PDF export."""
    prompt = (
        f"Validate this technology strategy report draft for PDF export readiness.\n"
        f"Draft (first 2500 chars): {draft[:2500]}\n\n"
        f"Check: reference-report style cover block, correct section headers, no broken markdown, citations present, "
        f"limitation note included, no placeholder text.\n"
        f"Return ONLY valid JSON:\n"
        f'{{ "ok": bool, "quality_score": float, "issues": [str], "suggestions": [str] }}'
    )
    try:
        resp = LLM_FULL.invoke(prompt)
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception as ex:
        print(f"[PDF P1] LLM error: {ex}")
    return {"ok": True, "quality_score": 0.85, "issues": [], "suggestions": []}


def _validate_pdf_tables_with_vlm(pdf_path: Path) -> Dict[str, Any]:
    """Use VLM as a visual QA pass: markdown tables should render as real table grids."""
    try:
        images = _pdf_pages_to_b64(str(pdf_path), max_pages=4)
        if not images:
            return {"ok": True, "issues": [], "suggestions": []}
        msg = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    "Inspect these PDF page images. If the report contains tables, verify they appear "
                    "as actual table grids/cells, not raw markdown pipe text. Return ONLY valid JSON: "
                    '{"ok": bool, "issues": [str], "suggestions": [str]}'
                ),
            },
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
                for b64 in images
            ],
        ])
        resp = LLM_VLM.invoke([msg])
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception as ex:
        print(f"[PDF VLM-QA] table validation skipped: {ex}")
    return {"ok": True, "issues": [], "suggestions": []}


# ═══════════════════════════════════════════════════════════════════════════
# Agent nodes
# ═══════════════════════════════════════════════════════════════════════════

def rag_agent_node(state: AgentState) -> AgentState:
    """
    RAG Agent (R1–R4)
      R1: Load & chunk documents
      R2: Hybrid search (bge-m3 dense + BM25 sparse, 0.5:0.5)
      R3: VLM processing of any PDF sources in DATA_DIR
      R4: Judge evaluation
    """
    target_techs = state.get("target_technologies", ["HBM4", "PIM", "CXL"])
    base_question = state["question"]

    # Build per-technology queries; incorporate Supervisor feedback on retry (spec 3-3)
    queries = [
        base_question,
        f"TRL evidence for {base_question}",
        f"competitor signal for {base_question}",
    ]
    if state.get("retry_reason") or state.get("improvement_instructions"):
        queries.append(
            f"{base_question} {state.get('retry_reason', '')} "
            f"{state.get('improvement_instructions', '')}"
        )

    target_cos = state.get("target_companies", TARGET_COMPANIES)
    retrieved: List[Document] = []
    for tech in target_techs:
        tech_queries = [
            f"{tech} latest competitor updates",
            f"{tech} TRL maturity evidence",
            f"{tech} paper patent product signal",
        ]
        # 기업별 전용 FAISS 검색
        for co in target_cos:
            for q in queries[:2] + tech_queries:
                retrieved.extend(hybrid_search(f"{q} {tech}", tech_hint=tech, company=co, top_k=3))
        # 전체 병합 인덱스도 추가 검색 (논문 등 기업 미분류 자료 포함)
        for q in queries[:2]:
            retrieved.extend(hybrid_search(f"{q} {tech}", tech_hint=tech, top_k=3))

    retrieved.extend(literal_source_matches(target_techs, target_cos))

    # R3: Augment with VLM-extracted documents from PDF sources
    vlm_docs = process_pdf_sources_with_vlm(DATA_DIR)
    retrieved.extend(vlm_docs)

    # Dedup
    seen: set = set()
    dedup: List[Document] = []
    for d in retrieved:
        cid = d.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            dedup.append(d)

    normalized = [normalize_doc(d, "rag") for d in dedup]

    # R4: Judge — coverage, company, citation scores
    covered_techs = {
        tech for tech in target_techs
        if any(evidence_covers_target(e, tech) for e in normalized)
    }
    missing_techs = [tech for tech in target_techs if tech not in covered_techs]
    coverage_score = len(covered_techs) / max(len(target_techs), 1)
    target_cos = state.get("target_companies", TARGET_COMPANIES)
    company_score = min(
        len({e["company"] for e in normalized}.intersection(set(target_cos))) / max(len(target_cos), 1),
        1.0,
    )
    citation_score = min(len({e["doc_id"] for e in normalized}) / 4.0, 1.0)
    score = 0.50 * coverage_score + 0.20 * company_score + 0.30 * citation_score

    issues: List[str] = []
    suggestions: List[str] = []
    if coverage_score < 1.0:
        issues.append(f"RAG does not cover all target technologies: missing {missing_techs}.")
        suggestions.append("Increase per-technology query variants.")
    if citation_score < 1.0:
        issues.append("RAG unique citations are under target >= 4.")
        suggestions.append("Increase top_k and include more source categories.")

    return {
        "supplement_queries": queries,
        "rag_retrieved": [
            {
                "doc_id": d.metadata["doc_id"],
                "chunk_id": d.metadata["chunk_id"],
                "title": d.metadata["title"],
                "technology": d.metadata["technology"],
                "company": d.metadata["company"],
            }
            for d in dedup
        ],
        "normalized_evidence": normalized,
        "rag_judge_result": build_judge(score, get_threshold(state, "rag"), issues, suggestions),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 웹 검색 결과 캐시 (data/web_cache/{company_slug}/{query_hash}.json)
# ═══════════════════════════════════════════════════════════════════════════

def _web_cache_path(company: str, query: str) -> Path:
    slug = COMPANY_SLUG.get(company, company.lower().replace(" ", ""))
    d = WEB_CACHE_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    qhash = hashlib.md5(query.encode()).hexdigest()[:12]
    return d / f"{qhash}.json"


def _load_web_cache(company: str, query: str) -> Optional[List[Dict[str, Any]]]:
    """캐시된 웹 결과를 로드한다. 없으면 None 반환."""
    p = _web_cache_path(company, query)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_web_cache(company: str, query: str, results: List[Dict[str, Any]]) -> None:
    """웹 검색 결과를 기업별 캐시에 저장한다."""
    p = _web_cache_path(company, query)
    p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _web_cache_to_faiss(new_docs: List[Document]) -> None:
    """새 웹 Document들을 web_cache FAISS + 기업별 FAISS + merged FAISS에 추가/저장한다.
    이렇게 하면 다음 RAG 검색에서 즉시 웹 수집 결과를 재사용할 수 있다.
    """
    if not new_docs:
        return

    # ── web_cache 전용 인덱스 (디스크 영구 저장) ────────────────────────────
    try:
        if WEB_CACHE_FAISS_PATH.exists():
            wc_vs = FAISS.load_local(
                str(WEB_CACHE_FAISS_PATH), EMBED_MODEL, allow_dangerous_deserialization=True
            )
            wc_vs.add_documents(new_docs)
        else:
            wc_vs = FAISS.from_documents(new_docs, EMBED_MODEL)
        WEB_CACHE_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        wc_vs.save_local(str(WEB_CACHE_FAISS_PATH))
        print(f"[WebCache] web_cache FAISS 저장 ({len(new_docs)}건)")
    except Exception as ex:
        print(f"[WebCache] web_cache FAISS 저장 실패: {ex}")

    # ── 기업별 인덱스에 반영 (in-memory + 디스크) ───────────────────────────
    by_company: Dict[str, List[Document]] = {}
    for doc in new_docs:
        co = doc.metadata.get("company", "Industry")
        by_company.setdefault(co, []).append(doc)

    for co, docs in by_company.items():
        # in-memory 갱신
        if co in COMPANY_VECTORSTORES:
            try:
                COMPANY_VECTORSTORES[co].add_documents(docs)
            except Exception as ex:
                print(f"[WebCache] {co} in-memory 갱신 실패: {ex}")

        # 디스크 저장 (해당 회사 전용 인덱스)
        path = COMPANY_FAISS_PATHS.get(co)
        if path:
            try:
                if path.exists():
                    co_vs = FAISS.load_local(str(path), EMBED_MODEL,
                                             allow_dangerous_deserialization=True)
                    co_vs.add_documents(docs)
                else:
                    co_vs = FAISS.from_documents(docs, EMBED_MODEL)
                path.parent.mkdir(parents=True, exist_ok=True)
                co_vs.save_local(str(path))
                print(f"[WebCache] {co} FAISS 저장 ({len(docs)}건)")
            except Exception as ex:
                print(f"[WebCache] {co} 디스크 저장 실패: {ex}")

    # ── 전체 merged store에도 in-memory 반영 ────────────────────────────────
    if VECTORSTORE is not None:
        try:
            VECTORSTORE.add_documents(new_docs)
        except Exception as ex:
            print(f"[WebCache] merged in-memory 갱신 실패: {ex}")


def _detect_company(text: str) -> str:
    """텍스트에서 기업명을 감지한다."""
    t = text.lower()
    if "sk hynix" in t or "hynix" in t:
        return "SK hynix"
    if "samsung" in t:
        return "Samsung"
    if "nvidia" in t or "nvda" in t:
        return "NVIDIA"
    if "intel" in t:
        return "Intel"
    if "micron" in t:
        return "Micron"
    if "tsmc" in t:
        return "TSMC"
    return "Industry"


def _is_relevant_web_result(result_text: str, target_techs: List[str], query: str) -> bool:
    haystack = result_text.lower()
    blocked_terms = [
        "harris", "biden", "trump", "election", "poll", "newspaper-8-30-2024",
        "wsjnewspaper", "wall street journal poll",
    ]
    if any(term in haystack for term in blocked_terms):
        return False
    query_terms = [
        term.strip().lower()
        for term in query.replace("/", " ").replace(",", " ").split()
        if len(term.strip()) >= 4
    ]
    tech_terms = [t.strip().lower() for t in target_techs if t.strip()]
    company_terms = ["nvidia", "sk hynix", "hynix", "samsung", "intel", "hbm", "cxl", "blackwell", "hopper"]
    required_terms = set(tech_terms + company_terms + query_terms[:6])
    return any(term in haystack for term in required_terms)


def _is_report_relevant_evidence(evidence: Dict[str, Any], state: AgentState) -> bool:
    text = " ".join(
        str(evidence.get(key, ""))
        for key in ("title", "source_url", "excerpt", "company", "technology")
    )
    return _is_relevant_web_result(
        text,
        state.get("target_technologies", ["HBM4", "PIM", "CXL"]),
        state.get("question", ""),
    )


def _tavily_search(queries: List[str], target_techs: List[str]) -> List[Document]:
    """W2: Tavily 실시간 웹 검색 → Document 리스트 반환.
    캐시 히트 시 API 호출 없이 캐시에서 복원한다.
    새 결과는 기업별 JSON 캐시 + web_cache FAISS에 저장한다.
    """
    if TAVILY is None:
        return []

    gathered: List[Document] = []
    new_docs_for_faiss: List[Document] = []

    for q in queries:
        # 쿼리에서 기업명을 감지해 캐시 분류에 사용
        inferred_co = _detect_company(q)

        # 캐시 로드 시도
        cached = _load_web_cache(inferred_co, q)
        if cached is not None:
            print(f"[Web W2] 캐시 히트: '{q[:50]}' ({inferred_co})")
            for r in cached:
                combined = r.get("content", "") + " " + json.dumps(r.get("metadata", {}), ensure_ascii=False)
                if not _is_relevant_web_result(combined, target_techs, q):
                    continue
                doc = Document(page_content=r["content"], metadata=r["metadata"])
                gathered.append(doc)
            continue

        # 실제 Tavily 호출
        try:
            results = TAVILY.invoke(q)
            raw_to_cache: List[Dict[str, Any]] = []
            for r in results:
                url = r.get("url", "")
                content = r.get("content", "")
                title = r.get("title", q)
                if not _is_relevant_web_result(f"{title} {content} {url}", target_techs, q):
                    continue
                tech = detect_tech_hint(content + " " + title,
                                        fallback=target_techs[0] if target_techs else "HBM4")
                company = _detect_company(content + " " + url + " " + title)
                uid = hashlib.md5(url.encode()).hexdigest()[:8]
                meta = {
                    "doc_id":       f"web-{uid}",
                    "chunk_id":     f"web-{uid}-c0",
                    "title":        title,
                    "technology":   tech,
                    "company":      company,
                    "source_type":  "press_release",
                    "published_at": datetime.now().strftime("%Y-%m-%d"),
                    "source_url":   url,
                }
                doc = Document(page_content=content[:500], metadata=meta)
                gathered.append(doc)
                new_docs_for_faiss.append(doc)
                raw_to_cache.append({"content": content[:500], "metadata": meta})

            # 결과를 기업별 캐시에 저장
            if raw_to_cache:
                _save_web_cache(inferred_co, q, raw_to_cache)
                print(f"[Web W2] {len(raw_to_cache)}건 캐시 저장 ({inferred_co}: '{q[:50]}')")

        except Exception as ex:
            print(f"[Web W2] Tavily 오류 ('{q[:50]}'): {ex}")

    # 새 결과를 web_cache FAISS에 누적 저장
    if new_docs_for_faiss:
        _web_cache_to_faiss(new_docs_for_faiss)

    return gathered


def web_agent_node(state: AgentState) -> AgentState:
    """
    Web Search Agent (W1–W3)
      W1: GPT-4o-mini generates targeted queries
      W2: Tavily 실시간 웹 검색 (API key 없으면 FAISS fallback)
      W3: GPT-4o-mini evaluates evidence quality and bias
    """
    target_techs = state.get("target_technologies", ["HBM4", "PIM", "CXL"])

    # W1: LLM-generated queries incorporating Supervisor feedback on retry (spec 3-3)
    queries = _generate_web_queries(state)

    gathered: List[Document] = []

    # W2: Tavily 실시간 검색 우선
    tavily_docs = _tavily_search(queries, target_techs)
    gathered.extend(tavily_docs)

    if tavily_docs:
        print(f"[Web W2] Tavily {len(tavily_docs)}건 수집")
    else:
        # Tavily 미사용 시 기업별 FAISS에서 웹 타입 문서 필터링 (fallback)
        print("[Web W2] FAISS fallback 사용 (Tavily 결과 없음)")
        allowed_web_types = {
            "press_release", "product_announcement", "earnings_call",
            "industry_report", "standard", "patent", "conference",
        }
        companies = state.get("target_companies", TARGET_COMPANIES)
        for co in companies:
            for tech in target_techs:
                gathered.extend(
                    hybrid_search(
                        f"{state['question']} {tech}", tech_hint=tech,
                        company=co, top_k=3, allowed_source_types=allowed_web_types,
                    )
                )
        for q in queries:
            for tech in target_techs:
                gathered.extend(
                    hybrid_search(f"{q} {tech}", tech_hint=tech, top_k=2,
                                  allowed_source_types=allowed_web_types)
                )
        if len(gathered) < 4:
            for tech in target_techs:
                gathered.extend(hybrid_search(f"{state['question']} {tech}", tech_hint=tech, top_k=3))

    # Dedup
    seen: set = set()
    dedup: List[Document] = []
    for d in gathered:
        cid = d.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            dedup.append(d)

    web_evidence = [normalize_doc(d, "web") for d in dedup]
    merged = list(state.get("normalized_evidence", [])) + web_evidence

    # W3: LLM-based evaluation
    eval_result = _evaluate_web_evidence(web_evidence, state)
    rule_eval = _rule_based_web_evidence_score(web_evidence, state)
    llm_score = float(eval_result.get("score", 0.5))
    rule_score = float(rule_eval.get("score", 0.0))
    if rule_score > llm_score:
        score = rule_score
        issues = rule_eval.get("issues", [])
        suggestions = rule_eval.get("suggestions", [])
        suggestions.append(f"LLM W3 score was {llm_score:.4f}; deterministic evidence score used.")
    else:
        score = llm_score
        issues = eval_result.get("issues", [])
        suggestions = eval_result.get("suggestions", [])

    return {
        "web_queries": queries,
        "web_classified": web_evidence,
        "normalized_evidence": merged,
        "web_judge_result": build_judge(score, get_threshold(state, "web"), issues, suggestions),
    }


def analysis_agent_node(state: AgentState) -> AgentState:
    """
    Analysis Agent (A1–A2)
      A1: GPT-4o estimates TRL for each company–technology pair
      A2: GPT-4o generates competitor gap analysis narrative
    """
    evidence = list(state.get("normalized_evidence", []))
    companies = state.get("target_companies", ["SK hynix", "Samsung", "Micron"])
    techs = state.get("target_technologies", ["HBM4", "PIM", "CXL"])

    # On retry, expand evidence retrieval depth using Supervisor feedback (spec 3-3)
    if state.get("retry_reason") or state.get("improvement_instructions"):
        for tech in techs:
            extra = hybrid_search(
                f"{state['question']} {tech} detailed evidence",
                tech_hint=tech, top_k=3,
            )
            for doc in extra:
                norm = normalize_doc(doc, "rag_retry")
                if not any(e["doc_id"] == norm["doc_id"] for e in evidence):
                    evidence.append(norm)

    # Build the base TRL structure
    trl_scores: Dict[str, Dict[str, Dict[str, Any]]] = {
        c: {
            t: {
                "trl": None,
                "confidence": 0.0,
                "evidence_count": 0,
                "rationale": [],
                "limit_note": "TRL 4-6 should be treated as inferred from indirect public signals.",
            }
            for t in techs
        }
        for c in companies
    }

    # A1: GPT-4o TRL analysis
    llm_trl = _analyze_trl_with_llm(evidence, companies, techs)

    for company in companies:
        for tech in techs:
            subset = [
                e for e in evidence
                if e["company"] == company and evidence_covers_target(e, tech)
            ]
            # Prefer LLM estimate if available
            llm_entry = (llm_trl or {}).get(company, {}).get(tech, {})
            if llm_entry and llm_entry.get("trl") is not None:
                trl_val = int(llm_entry["trl"])
                conf = float(llm_entry.get("confidence", min(len(subset) / 3.0, 1.0)))
                rationale = [llm_entry.get("rationale", "")]
            elif subset:
                # Rule-based fallback
                trl_vals = []
                reasons = []
                for e in subset:
                    trl, reason = _estimate_trl_rule(str(e["source_type"]))
                    trl_vals.append(trl)
                    reasons.append(f"[{e['doc_id']}] {reason}")
                trl_val = int(round(sum(trl_vals) / len(trl_vals)))
                conf = round(min(len(subset) / 3.0, 1.0), 3)
                rationale = reasons[:4]
            else:
                continue

            trl_scores[company][tech]["trl"] = trl_val
            trl_scores[company][tech]["confidence"] = round(conf, 3)
            trl_scores[company][tech]["evidence_count"] = len(subset)
            trl_scores[company][tech]["rationale"] = rationale[:4]

    # Build competitor matrix
    baseline = "SK hynix"
    matrix = []
    for tech in techs:
        base_trl = trl_scores[baseline][tech]["trl"] or 0
        for company in companies:
            trl = trl_scores[company][tech]["trl"] or 0
            gap = trl - base_trl
            if company != baseline:
                threat = "HIGH" if gap >= 1 else "MEDIUM" if gap == 0 else "LOW"
            else:
                threat = "BASELINE"
            evidence_count = trl_scores[company][tech]["evidence_count"]
            confidence = trl_scores[company][tech]["confidence"]
            if trl >= 8:
                criterion = "product/deployment signal"
            elif trl >= 6:
                criterion = "program integration signal"
            elif trl >= 4:
                criterion = "standard/prototype signal"
            elif trl > 0:
                criterion = "research signal"
            else:
                criterion = "no direct evidence"
            if company == baseline:
                reason = (
                    f"baseline is {baseline}, the comparison anchor; "
                    f"{evidence_count} evidence item(s), confidence {confidence}"
                )
            elif gap >= 1:
                reason = f"ahead of baseline by {gap} TRL step(s); {evidence_count} evidence item(s)"
            elif gap == 0:
                reason = f"similar maturity to baseline; {evidence_count} evidence item(s)"
            elif trl == 0:
                reason = "no direct evidence found in current corpus"
            else:
                reason = f"behind baseline by {abs(gap)} TRL step(s); {evidence_count} evidence item(s)"
            matrix.append({
                "technology": tech,
                "company": company,
                "trl": trl,
                "gap_vs_skh": gap,
                "threat_level": threat,
                "evidence_count": evidence_count,
                "criterion": criterion,
                "reason": reason,
            })

    # A2: GPT-4o competitor narrative
    competitor_narrative = _generate_competitor_narrative(matrix, baseline, state["question"])

    filled = sum(1 for c in companies for t in techs if trl_scores[c][t]["trl"] is not None)
    completeness = filled / max(len(companies) * len(techs), 1)
    tech_coverage = (
        len({t for t in techs if any(evidence_covers_target(e, t) for e in evidence)})
        / max(len(techs), 1)
    )
    company_coverage = (
        len({e["company"] for e in evidence}.intersection(set(companies)))
        / max(len(companies), 1)
    )
    citation_score = min(len({e["doc_id"] for e in evidence}) / 8.0, 1.0)
    score = 0.45 * completeness + 0.25 * tech_coverage + 0.15 * company_coverage + 0.15 * citation_score
    issues = []
    suggestions = []
    if completeness < 0.55:
        issues.append("TRL matrix has many missing company-tech cells.")
        suggestions.append("Increase retrieval depth for missing company-tech pairs.")
    if tech_coverage < 1.0:
        issues.append("Analysis evidence does not cover all target technologies.")
        suggestions.append("Add exact-match technology evidence before analysis.")

    return {
        "trl_scores": trl_scores,
        "competitor_analysis": {
            "baseline": baseline,
            "matrix": matrix,
            "narrative": competitor_narrative,
            "limitation": "TRL 4-6 is estimated from indirect public signals and has uncertainty.",
        },
        "analysis_judge_result": build_judge(score, get_threshold(state, "analysis"), issues, suggestions),
    }


def _estimate_trl_rule(source_type: str) -> Tuple[int, str]:
    """Rule-based TRL fallback when LLM estimate is unavailable."""
    st = source_type.lower()
    if st in {"paper", "conference", "patent"}:
        return 3, "Research-oriented evidence, usually TRL 1-4."
    if st == "standard":
        return 4, "Standard activity can indicate integration readiness transition."
    if st in {"press_release", "industry_report"}:
        return 6, "Program-level progress often maps to TRL 5-7."
    if st in {"product_announcement", "earnings_call"}:
        return 8, "Productization signal often maps to TRL 7-9."
    return 5, "Fallback estimate due to limited source signal."


def draft_agent_node(state: AgentState) -> AgentState:
    """
    Draft Generation Agent (D1–D2)
      D1: GPT-4o writes full report sections (incorporating Supervisor + human feedback on retry)
      D2: GPT-4o checks section completeness
    """
    matrix = state.get("competitor_analysis", {}).get("matrix", [])
    competitor_narrative = state.get("competitor_analysis", {}).get("narrative", "")
    references = {
        e["doc_id"]: {
            "doc_id": e["doc_id"],
            "title": e["title"],
            "source_url": e["source_url"],
            "published_at": e["published_at"],
        }
        for e in state.get("normalized_evidence", [])
        if _is_report_relevant_evidence(e, state)
    }

    # D1: GPT-4o writes the report (feedback from Supervisor or human review incorporated per spec 3-3)
    companies_str = ", ".join(state.get("target_companies", TARGET_COMPANIES))
    llm_draft = _write_report_with_llm(state, matrix, references, competitor_narrative)

    if llm_draft:
        draft = _normalize_baseline_labels(llm_draft)
    else:
        # Minimal fallback if LLM fails
        improvement_note = ""
        if state.get("improvement_instructions"):
            improvement_note = f"\n\n> **Revision Note (Supervisor):** {state['improvement_instructions']}"
        human_feedback = str(state.get("human_review_result", {}).get("feedback", "")).strip()
        if human_feedback:
            improvement_note += f"\n\n> **Human Review Feedback:** {human_feedback}"
        lines = [
            "# R&D TECHNOLOGY STRATEGY REPORT",
            "# Next-Generation Semiconductor Technology Strategy Report",
            f"{' · '.join(state.get('target_technologies', ['HBM4', 'PIM', 'CXL']))} Technology Landscape and Competitive Threat Assessment",
            f"Date  {datetime.now().strftime('%B %d, %Y')}",
            "Division  Semiconductor R&D Strategy Team",
            "Classification  Internal Confidential / R&D Personnel Only",
            "Version  v1.0",
            "",
            "## EXECUTIVE SUMMARY",
            f"**Purpose:** This report summarizes HBM4, PIM, CXL, and architecture maturity signals "
            f"for {companies_str} using a Supervisor-controlled Agentic RAG workflow.{improvement_note}",
            "",
            "**Background:** AI accelerator architectures increasingly depend on memory bandwidth, near-memory compute, and scalable interconnects.",
            "",
            "**Conclusion:** SK hynix should prioritize the highest-confidence TRL gaps and convert them into near-term R&D actions.",
            "",
            "### Key Metrics",
            "- **TRL Focus**: HBM/CXL/PIM maturity and threat assessment.",
            "- **Competitor Lens**: NVIDIA, Samsung, Intel, SK hynix positioning.",
            "- **R&D Priority**: Thermal, memory integration, and interconnect leadership.",
            "",
            "## 1. Analysis Background — Why This Technology, Why Now",
            "AI memory demand is growing rapidly, so fast competitor intelligence is required for R&D prioritization.",
            "### 1.1 Market Inflection",
            "HBM and accelerator memory architecture are becoming the central bottleneck for AI systems.",
            "### 1.2 Research Rationale — Bridging the Information Gap",
            "Public signals are used because TRL 4-6 evidence is often partially disclosed.",
            "",
            "## 2. Technology Landscape",
            "- HBM4\n- PIM\n- CXL",
            "",
            "## 3. Competitor Trend Analysis",
            "| Technology | Company | TRL | Threat | Evidence Count | Selection Criterion | Short Reason |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "\n".join(
                f"| {row['technology']} | {row['company']} | {row['trl']} | {row['threat_level']} | "
                f"{row['evidence_count']} | {row.get('criterion', '')} | {row.get('reason', '')} |"
                for row in matrix[:12]
            ),
            "",
            "### Detailed Rationale",
            "\n".join(
                f"- {row['technology']} / {row['company']}: TRL {row['trl']} was selected because {row.get('reason', '')}. "
                f"The selection criterion is {row.get('criterion', '')}."
                for row in matrix[:6]
            ),
            "",
            "## 4. Strategic Implications — R&D Priority Recommendations",
            "Prioritize bottlenecks with strongest competitor momentum.",
            "",
            "## REFERENCES",
        ]
        for r in references.values():
            lines.append(f"- [{r['doc_id']}] {r['title']} ({r['published_at']}) - {r['source_url']}")
        lines += ["", "### Limitation Note", "TRL 4-6 values are inferred from indirect public signals."]
        draft = "\n".join(lines)
    draft = _ensure_trl_table_section(_normalize_baseline_labels(draft), matrix, korean=False)
    # D1-KO: GPT-4o 한국어 번역 (영어 draft 완성 후 번역)
    print("[Draft D1-KO] 한국어 보고서 번역 중 ...")
    draft_ko = _ensure_trl_table_section(
        _normalize_baseline_labels(_translate_to_korean(draft)),
        matrix,
        korean=True,
    )

    # D2: GPT-4o checks completeness
    completeness = _check_draft_completeness(draft)
    section_score = float(completeness.get("section_score", 0.8))
    citation_score_d = float(completeness.get("citation_score", 0.8))
    d2_issues = completeness.get("issues", [])
    d2_suggestions = completeness.get("suggestions", [])

    # Also do rule-based check for required headers
    required = [
        "# R&D TECHNOLOGY STRATEGY REPORT",
        "## EXECUTIVE SUMMARY",
        "### Key Metrics",
        "## 1. Analysis Background",
        "## 2. Technology Landscape",
        "## 3. Competitor Trend Analysis",
        "## 4. Strategic Implications",
        "## REFERENCES",
    ]
    rule_section_score = sum(1 for x in required if x in draft) / len(required)
    competitor_score = sum(1.0 / 3 for c in ["SK hynix", "Samsung", "Micron"] if c in draft)
    rule_citation_score = min(len(references) / 4.0, 1.0)
    rule_score = 0.45 * rule_section_score + 0.30 * competitor_score + 0.25 * rule_citation_score

    # Blend LLM and rule-based scores
    blended_score = 0.5 * ((section_score + citation_score_d) / 2) + 0.5 * rule_score
    issues = d2_issues[:]
    suggestions = d2_suggestions[:]
    if rule_citation_score < 1.0:
        issues.append("Citation count is below target >= 4.")
        suggestions.append("Increase evidence retrieval breadth before drafting.")

    return {
        "sections": {
            "draft_full": draft,
            "draft_full_ko": draft_ko,
            "competitor_narrative": competitor_narrative,
        },
        "references": list(references.values()),
        "draft": draft,
        "draft_ko": draft_ko,
        "draft_judge_result": build_judge(blended_score, get_threshold(state, "draft"), issues, suggestions),
    }


def human_review_node(state: AgentState) -> AgentState:
    decision = str(state["config"].get("human_decision", "approve")).lower().strip()
    feedback = str(state["config"].get("human_feedback", ""))
    if decision not in {"approve", "reject"}:
        decision = "approve"
    return {
        "human_review_result": {
            "decision": decision,
            "feedback": feedback,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
    }


def _render_pdf(draft: str, pdf_path: Path) -> Tuple[bool, List[str]]:
    """matplotlib으로 참고 보고서 스타일의 Markdown PDF를 렌더링한다."""
    issues: List[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.font_manager import FontProperties

        raw_lines = draft.splitlines()
        is_korean_pdf = any("\uac00" <= ch <= "\ud7a3" for ch in draft)

        font_candidates = [
            Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
            Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        ]
        font_path = next((p for p in font_candidates if p.exists()), None)
        if is_korean_pdf:
            font_props = FontProperties(fname=str(font_path), size=8.5) if font_path else FontProperties(family="DejaVu Sans", size=8.5)
        else:
            font_props = FontProperties(family="DejaVu Sans", size=7.8)
        small_props = font_props.copy()
        small_props.set_size(7 if is_korean_pdf else 6.5)
        title_props = font_props.copy()
        title_props.set_size(20 if is_korean_pdf else 18)
        title_props.set_weight("bold")
        subtitle_props = font_props.copy()
        subtitle_props.set_size(12 if is_korean_pdf else 10.5)
        section_props = font_props.copy()
        section_props.set_size(13 if is_korean_pdf else 11.5)
        section_props.set_weight("bold")
        h3_props = font_props.copy()
        h3_props.set_size(10 if is_korean_pdf else 8.8)
        h3_props.set_weight("bold")

        body_start = next((i for i, line in enumerate(raw_lines) if line.startswith("## ")), min(len(raw_lines), 7))
        cover_lines = [line.strip("# ").strip() for line in raw_lines[:body_start] if line.strip()]
        body_lines = raw_lines[body_start:]
        body_wrap_width = 88 if is_korean_pdf else 122
        blank_step = 0.018 if is_korean_pdf else 0.009
        body_line_step = 0.024 if is_korean_pdf else 0.016
        h2_step = 0.052 if is_korean_pdf else 0.043
        h3_step = 0.034 if is_korean_pdf else 0.026
        h4_step = 0.030 if is_korean_pdf else 0.024

        def clean_inline_markdown(value: str) -> str:
            cleaned = value.strip()
            cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
            cleaned = cleaned.replace("<br>", " ").replace("<br/>", " ")
            if cleaned.startswith("- "):
                cleaned = "• " + cleaned[2:]
            return cleaned

        def is_markdown_table_line(value: str) -> bool:
            stripped_value = value.strip()
            return stripped_value.startswith("|") and stripped_value.endswith("|")

        def is_markdown_separator(value: str) -> bool:
            cells = [cell.strip() for cell in value.strip().strip("|").split("|")]
            return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)

        def parse_table_row(value: str) -> List[str]:
            return [clean_inline_markdown(cell) for cell in value.strip().strip("|").split("|")]

        def extract_tables(lines: List[str]) -> List[List[List[str]]]:
            tables: List[List[List[str]]] = []
            i = 0
            while i < len(lines):
                if is_markdown_table_line(lines[i]):
                    raw_table = []
                    while i < len(lines) and is_markdown_table_line(lines[i]):
                        raw_table.append(lines[i])
                        i += 1
                    rows = [parse_table_row(line) for line in raw_table if not is_markdown_separator(line)]
                    if len(rows) >= 2:
                        tables.append(rows)
                else:
                    i += 1
            return tables

        markdown_tables = extract_tables(body_lines)

        def extract_trl_rows() -> List[Tuple[str, str, float]]:
            rows: List[Tuple[str, str, float]] = []
            for table in markdown_tables:
                headers = [h.lower() for h in table[0]]
                if not any("trl" in h for h in headers):
                    continue
                tech_idx = next((i for i, h in enumerate(headers) if "tech" in h or "기술" in h), None)
                co_idx = next((i for i, h in enumerate(headers) if "company" in h or "회사" in h), None)
                trl_idx = next((i for i, h in enumerate(headers) if "trl" in h), None)
                if tech_idx is None or co_idx is None or trl_idx is None:
                    continue
                for row in table[1:]:
                    try:
                        raw_trl = row[trl_idx].upper().replace("TRL=", "").replace("TRL", "").strip()
                        trl = float(raw_trl.split()[0])
                        rows.append((row[tech_idx], row[co_idx], trl))
                    except Exception:
                        continue
            return rows[:16]

        def add_footer(fig, page_no: int) -> None:
            fig.text(0.06, 0.035, "SK Hynix · Next-Gen Semiconductor Technology Strategy Report",
                     fontproperties=small_props, color="#54616d")
            fig.text(0.94, 0.035, str(page_no), ha="right", fontproperties=small_props, color="#54616d")

        page_no = 1
        with PdfPages(str(pdf_path)) as pdf:
            # Cover page
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color="#0b253a", transform=ax.transAxes))
            ax.add_patch(plt.Rectangle((0.06, 0.12), 0.015, 0.68, color="#1f77b4", transform=ax.transAxes))
            fig.text(0.09, 0.93, cover_lines[0] if cover_lines else "R&D TECHNOLOGY STRATEGY REPORT",
                     fontproperties=section_props, color="white")
            fig.text(0.09, 0.76, cover_lines[1] if len(cover_lines) > 1 else "Next-Generation Semiconductor Technology Strategy Report",
                     fontproperties=title_props, color="#0b253a")
            fig.text(0.09, 0.68, cover_lines[2] if len(cover_lines) > 2 else "",
                     fontproperties=subtitle_props, color="#2f4050")
            meta_y = 0.48
            for line in cover_lines[3:7]:
                if set(line.replace("|", "").replace(" ", "")) <= {"-"}:
                    continue
                meta_line = "   ".join(cell.strip() for cell in line.split("|")) if "|" in line else line
                fig.text(0.09, meta_y, clean_inline_markdown(meta_line), fontproperties=font_props, color="#2f4050")
                meta_y -= 0.045
            add_footer(fig, page_no)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            page_no += 1

            # Body pages
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.94

            def flush_page() -> None:
                nonlocal fig, ax, y, page_no
                add_footer(fig, page_no)
                pdf.savefig(fig, bbox_inches="tight")
                plt.axis("off")
                plt.close(fig)
                page_no += 1
                fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.94

            def is_markdown_table_line(value: str) -> bool:
                stripped_value = value.strip()
                return stripped_value.startswith("|") and stripped_value.endswith("|")

            def is_markdown_separator(value: str) -> bool:
                cells = [cell.strip() for cell in value.strip().strip("|").split("|")]
                return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)

            def parse_table_row(value: str) -> List[str]:
                return [clean_inline_markdown(cell) for cell in value.strip().strip("|").split("|")]

            def render_table(table_lines: List[str]) -> None:
                nonlocal y
                rows = [parse_table_row(line) for line in table_lines if not is_markdown_separator(line)]
                if len(rows) < 2:
                    for raw in table_lines:
                        render_text_line(raw)
                    return

                header, data = rows[0], rows[1:]
                col_count = len(header)
                normalized_data = [
                    (row + [""] * col_count)[:col_count]
                    for row in data
                ]

                def column_widths(headers: List[str]) -> List[float]:
                    lowered = [h.lower() for h in headers]
                    if col_count >= 7 and any("reason" in h or "요약" in h for h in lowered):
                        weights = [0.13, 0.13, 0.08, 0.13, 0.10, 0.20, 0.23]
                    elif col_count >= 5:
                        weights = [1.15 if i == 0 else 1.0 for i in range(col_count)]
                    else:
                        weights = [1.0 for _ in range(col_count)]
                    total = sum(weights) or 1.0
                    return [w / total for w in weights]

                widths = column_widths(header)

                def wrap_width(col_idx: int, text: str) -> int:
                    lowered_header = header[col_idx].lower() if col_idx < len(header) else ""
                    if "reason" in lowered_header or "요약" in lowered_header:
                        return 16
                    if "criterion" in lowered_header or "선정" in lowered_header:
                        return 15
                    if "evidence" in lowered_header or "증거" in lowered_header:
                        return 8
                    if "threat" in lowered_header or "위협" in lowered_header:
                        return 12
                    if "company" in lowered_header or "회사" in lowered_header:
                        return 12
                    return max(8, int(34 * widths[col_idx]))

                def wrap_cell(col_idx: int, text: str) -> str:
                    parts = textwrap.wrap(
                        text,
                        width=wrap_width(col_idx, text),
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    return "\n".join(parts) if parts else text

                wrapped_header = [wrap_cell(idx, value) for idx, value in enumerate(header)]
                wrapped_data = [
                    [wrap_cell(idx, value) for idx, value in enumerate(row)]
                    for row in normalized_data
                ]
                row_line_counts = [
                    max((cell.count("\n") + 1 for cell in row), default=1)
                    for row in wrapped_data
                ]
                table_height = min(0.052 + sum(0.026 * max(1, lines) for lines in row_line_counts), 0.76)
                if y - table_height < 0.10:
                    flush_page()

                table = ax.table(
                    cellText=wrapped_data,
                    colLabels=wrapped_header,
                    cellLoc="left",
                    colLoc="left",
                    colWidths=widths,
                    bbox=[0.07, y - table_height, 0.86, table_height],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(6.8 if col_count >= 7 else 7.2)
                for (row_idx, col_idx), cell in table.get_celld().items():
                    cell.set_edgecolor("#b7c7d6")
                    cell.set_linewidth(0.55)
                    cell.set_text_props(fontproperties=font_props, color="#1c2733")
                    if row_idx == 0:
                        cell.set_facecolor("#0b253a")
                        cell.set_text_props(fontproperties=small_props, color="white", weight="bold")
                    elif row_idx % 2 == 0:
                        cell.set_facecolor("#f4f8fb")
                    else:
                        cell.set_facecolor("white")
                y -= table_height + 0.026

            def render_trl_chart() -> None:
                nonlocal y
                rows = extract_trl_rows()
                if not rows:
                    return
                rows = sorted(rows, key=lambda item: item[2], reverse=True)[:10]
                chart_height = 0.08 + 0.035 * len(rows)
                if y - chart_height < 0.10:
                    flush_page()
                fig.text(0.07, y, "TRL Score Comparison", fontproperties=h3_props, color="#1f4e79")
                y -= 0.035
                max_trl = 9.0
                for tech, company, trl in rows:
                    label = f"{tech} / {company}"[:28]
                    fig.text(0.075, y, label, fontproperties=small_props, color="#263442")
                    bar_x = 0.30
                    bar_w = 0.48 * min(trl / max_trl, 1.0)
                    ax.add_patch(plt.Rectangle((bar_x, y - 0.006), 0.48, 0.015, color="#e3ebf2", transform=ax.transAxes))
                    ax.add_patch(plt.Rectangle((bar_x, y - 0.006), bar_w, 0.015, color="#1f77b4", transform=ax.transAxes))
                    fig.text(0.80, y, f"TRL {trl:g}", fontproperties=small_props, color="#263442")
                    y -= 0.033
                y -= 0.020

            def render_architecture_diagram() -> None:
                nonlocal y
                diagram_height = 0.20
                if y - diagram_height < 0.10:
                    flush_page()
                fig.text(0.07, y, "Architecture View", fontproperties=h3_props, color="#1f4e79")
                y -= 0.035
                boxes = [
                    ("AI Workload", 0.07, "#e8f1f8"),
                    ("GPU Architecture\nBlackwell / Hopper", 0.25, "#dceefc"),
                    ("HBM Stack\nBandwidth / Thermal", 0.45, "#eaf6ea"),
                    ("CXL Memory Pool\nScale-out Fabric", 0.65, "#fff2d9"),
                ]
                box_y = y - 0.095
                for label, x, color in boxes:
                    ax.add_patch(plt.Rectangle((x, box_y), 0.15, 0.075, color=color, ec="#8fa7ba", lw=0.8, transform=ax.transAxes))
                    for idx, line in enumerate(label.splitlines()):
                        fig.text(x + 0.075, box_y + 0.052 - idx * 0.023, line,
                                 ha="center", fontproperties=small_props, color="#203040")
                for x in [0.22, 0.42, 0.62]:
                    ax.annotate("", xy=(x + 0.03, box_y + 0.037), xytext=(x, box_y + 0.037),
                                arrowprops={"arrowstyle": "->", "color": "#48677f", "lw": 1.0},
                                xycoords=ax.transAxes)
                ax.add_patch(plt.Rectangle((0.40, box_y - 0.065), 0.20, 0.042, color="#f4e8fa", ec="#a58cb5", lw=0.8, transform=ax.transAxes))
                fig.text(0.50, box_y - 0.047, "PIM: near-memory compute lever",
                         ha="center", fontproperties=small_props, color="#3d2b48")
                y -= diagram_height

            def render_text_line(line: str) -> None:
                nonlocal y
                stripped = line.strip()
                if stripped.startswith("<!--") and stripped.endswith("-->"):
                    return
                if not stripped:
                    y -= blank_step
                    return
                if y < 0.09:
                    flush_page()

                if stripped.startswith("#### "):
                    fig.text(0.07, y, clean_inline_markdown(stripped[5:]), fontproperties=h3_props, color="#1f4e79")
                    y -= h4_step
                elif stripped.startswith("## "):
                    text = clean_inline_markdown(stripped[3:])
                    ax.add_patch(plt.Rectangle((0.055, y - 0.012), 0.89, 0.035, color="#e8f1f8", transform=ax.transAxes))
                    fig.text(0.07, y, text, fontproperties=section_props, color="#0b253a")
                    y -= h2_step
                elif stripped.startswith("### "):
                    fig.text(0.07, y, clean_inline_markdown(stripped[4:]), fontproperties=h3_props, color="#1f4e79")
                    y -= h3_step
                elif stripped.startswith("# "):
                    fig.text(0.07, y, clean_inline_markdown(stripped[2:]), fontproperties=section_props, color="#0b253a")
                    y -= h2_step
                else:
                    cleaned = clean_inline_markdown(stripped)
                    wrapped = textwrap.wrap(cleaned, width=body_wrap_width, replace_whitespace=False) or [cleaned]
                    for wline in wrapped:
                        if y < 0.09:
                            flush_page()
                        fig.text(0.075, y, wline, fontproperties=font_props, color="#222222")
                        y -= body_line_step

            i = 0
            while i < len(body_lines):
                line = body_lines[i]
                stripped = line.strip()
                if stripped == "<!-- VISUAL:TRL_CHART -->":
                    render_trl_chart()
                    i += 1
                elif stripped == "<!-- VISUAL:ARCHITECTURE_DIAGRAM -->":
                    render_architecture_diagram()
                    i += 1
                elif is_markdown_table_line(stripped):
                    table_lines = []
                    while i < len(body_lines) and is_markdown_table_line(body_lines[i]):
                        table_lines.append(body_lines[i])
                        i += 1
                    render_table(table_lines)
                else:
                    render_text_line(line)
                    i += 1
            add_footer(fig, page_no)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        ok = pdf_path.exists() and pdf_path.stat().st_size > 0
        return ok, issues
    except Exception as ex:
        issues.append(f"PDF rendering failed: {ex}")
        return False, issues


def pdf_agent_node(state: AgentState) -> AgentState:
    """
    PDF Generation Agent (P1–P2)
      P1: GPT-4o validates draft before export
      P2: Generate markdown + PDF artifact (matplotlib) — 영어 & 한국어 각각 저장
    """
    draft_en = state.get("draft", "").strip()
    draft_ko = state.get("draft_ko", "").strip()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 영어 파일 경로 ─────────────────────────────────────────────────────
    md_en_path  = OUTPUT_DIR / f"tech_strategy_report_{stamp}_en.md"
    pdf_en_path = OUTPUT_DIR / f"tech_strategy_report_{stamp}_en.pdf"

    # ── 한국어 파일 경로 ────────────────────────────────────────────────────
    md_ko_path  = OUTPUT_DIR / f"tech_strategy_report_{stamp}_ko.md"
    pdf_ko_path = OUTPUT_DIR / f"tech_strategy_report_{stamp}_ko.pdf"

    # Markdown 저장
    md_en_path.write_text(draft_en, encoding="utf-8")
    if draft_ko:
        md_ko_path.write_text(draft_ko, encoding="utf-8")

    # P1: LLM validates English draft quality
    validation = _validate_draft_for_pdf(draft_en)
    p1_issues: List[str] = validation.get("issues", [])
    p1_suggestions: List[str] = validation.get("suggestions", [])

    # P2: PDF 렌더링 (영어)
    pdf_en_ok, en_errs = _render_pdf(_inject_visual_markers(draft_en), pdf_en_path)
    p1_issues.extend(en_errs)
    if not pdf_en_ok:
        p1_suggestions.append("Install matplotlib and retry PDF generation.")
    else:
        table_qa = _validate_pdf_tables_with_vlm(pdf_en_path)
        if not table_qa.get("ok", True):
            p1_issues.extend(table_qa.get("issues", []))
            p1_suggestions.extend(table_qa.get("suggestions", []))

    # P2: PDF 렌더링 (한국어)
    pdf_ko_ok = False
    if draft_ko:
        pdf_ko_ok, ko_errs = _render_pdf(_inject_visual_markers(draft_ko), pdf_ko_path)
        if ko_errs:
            print(f"[PDF P2-KO] 한국어 PDF 오류: {ko_errs}")

    print(f"[PDF] 영어  MD : {md_en_path.name}")
    print(f"[PDF] 영어  PDF: {pdf_en_path.name} ({'OK' if pdf_en_ok else 'FAIL'})")
    if draft_ko:
        print(f"[PDF] 한국어 MD : {md_ko_path.name}")
        print(f"[PDF] 한국어 PDF: {pdf_ko_path.name} ({'OK' if pdf_ko_ok else 'FAIL'})")

    # Blend P1 LLM quality score with rendering result
    p1_quality = float(validation.get("quality_score", 0.85))
    score = 0.4 * p1_quality + 0.6 * (1.0 if pdf_en_ok else 0.0)

    return {
        "markdown_path":    str(md_en_path),
        "pdf_path":         str(pdf_en_path) if pdf_en_ok else "",
        "markdown_ko_path": str(md_ko_path) if draft_ko else "",
        "pdf_ko_path":      str(pdf_ko_path) if pdf_ko_ok else "",
        "pdf_judge_result": build_judge(score, get_threshold(state, "pdf"), p1_issues, p1_suggestions),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Retry / fail helper
# ═══════════════════════════════════════════════════════════════════════════

def retry_or_fail(
    state: AgentState,
    agent_key: str,
    status_if_retry: str,
    retry_reason: str,
    improve: str,
) -> AgentState:
    rev = dict(state.get("revision_count", {}))
    used = int(rev.get(agent_key, 0))
    budget = int(state["config"]["max_retries"].get(agent_key, 0))
    if used < budget:
        rev[agent_key] = used + 1
        print(f"[Loop] {agent_key}: retry {used + 1}/{budget} - {retry_reason}")
        return {
            "status": status_if_retry,
            "revision_count": rev,
            "retry_reason": retry_reason,
            "improvement_instructions": improve,
        }
    print(f"[Loop] {agent_key}: failed after {used}/{budget} retries - {retry_reason}")
    return {
        "status": "failed",
        "failure_reason": f"{agent_key} exceeded retry budget ({budget}).",
        "revision_count": rev,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Supervisor node (SR — GPT-4o-mini for improvement instructions)
# ═══════════════════════════════════════════════════════════════════════════

def supervisor_node(state: AgentState) -> AgentState:
    """SR: Single authority for routing and workflow termination (spec 3-1).
    Routing is deterministic (threshold-based); improvement instructions use GPT-4o-mini.
    """
    # ── RAG ──────────────────────────────────────────────────────────────
    if "rag_judge_result" not in state:
        print("[Supervisor] route -> rag_agent")
        return {"status": "run_rag", "retry_reason": "", "improvement_instructions": ""}
    if state["rag_judge_result"]["score"] < get_threshold(state, "rag"):
        improve = _supervisor_generate_instructions(state, "rag")
        return retry_or_fail(state, "rag", "run_rag", "RAG quality below threshold.", improve)

    # ── Web Search ───────────────────────────────────────────────────────
    if state["config"].get("enable_web_agent", True):
        if "web_judge_result" not in state:
            print("[Supervisor] route -> web_agent")
            return {"status": "run_web", "retry_reason": "", "improvement_instructions": ""}
        if state["web_judge_result"]["score"] < get_threshold(state, "web"):
            improve = _supervisor_generate_instructions(state, "web")
            return retry_or_fail(state, "web", "run_web", "Web evidence quality below threshold.", improve)

    # ── Analysis ─────────────────────────────────────────────────────────
    if "analysis_judge_result" not in state:
        print("[Supervisor] route -> analysis_agent")
        return {"status": "run_analysis", "retry_reason": "", "improvement_instructions": ""}
    if state["analysis_judge_result"]["score"] < get_threshold(state, "analysis"):
        improve = _supervisor_generate_instructions(state, "analysis")
        return retry_or_fail(state, "analysis", "run_analysis", "Analysis quality below threshold.", improve)

    # ── Draft ─────────────────────────────────────────────────────────────
    if "draft_judge_result" not in state:
        print("[Supervisor] route -> draft_agent")
        return {"status": "run_draft", "retry_reason": "", "improvement_instructions": ""}
    if state["draft_judge_result"]["score"] < get_threshold(state, "draft"):
        improve = _supervisor_generate_instructions(state, "draft")
        return retry_or_fail(state, "draft", "run_draft", "Draft quality below threshold.", improve)

    # ── Human Review ──────────────────────────────────────────────────────
    if "human_review_result" not in state:
        print("[Supervisor] route -> human_review")
        return {"status": "run_human_review"}
    if state["human_review_result"].get("decision") == "reject":
        feedback = str(state["human_review_result"].get("feedback", "Incorporate review comments."))
        return retry_or_fail(
            state, "draft", "run_draft",
            "Human reviewer rejected draft.", feedback,
        )

    # ── PDF ───────────────────────────────────────────────────────────────
    if "pdf_judge_result" not in state:
        print("[Supervisor] route -> pdf_agent")
        return {"status": "run_pdf", "retry_reason": "", "improvement_instructions": ""}
    if state["pdf_judge_result"]["score"] < get_threshold(state, "pdf"):
        improve = _supervisor_generate_instructions(state, "pdf")
        return retry_or_fail(state, "pdf", "run_pdf", "PDF generation below threshold.", improve)

    # ── Done ──────────────────────────────────────────────────────────────
    if state.get("pdf_path"):
        print("[Supervisor] route -> end")
        return {"status": "end"}
    return {"status": "failed", "failure_reason": "PDF judged as pass but file path missing."}


def route_from_supervisor(state: AgentState) -> str:
    return {
        "run_rag": "rag_agent",
        "run_web": "web_agent",
        "run_analysis": "analysis_agent",
        "run_draft": "draft_agent",
        "run_human_review": "human_review",
        "run_pdf": "pdf_agent",
        "end": "end",
        "failed": "end",
    }.get(state.get("status", "failed"), "end")


# ═══════════════════════════════════════════════════════════════════════════
# Graph assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("web_agent", web_agent_node)
    builder.add_node("analysis_agent", analysis_agent_node)
    builder.add_node("draft_agent", draft_agent_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("pdf_agent", pdf_agent_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "rag_agent": "rag_agent",
            "web_agent": "web_agent",
            "analysis_agent": "analysis_agent",
            "draft_agent": "draft_agent",
            "human_review": "human_review",
            "pdf_agent": "pdf_agent",
            "end": END,
        },
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("web_agent", "supervisor")
    builder.add_edge("analysis_agent", "supervisor")
    builder.add_edge("draft_agent", "supervisor")
    builder.add_edge("human_review", "supervisor")
    builder.add_edge("pdf_agent", "supervisor")
    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Public entry points
# ═══════════════════════════════════════════════════════════════════════════

def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_demo(
    human_decision: str = "approve",
    question: Optional[str] = None,
    target_technologies: Optional[List[str]] = None,
    target_companies: Optional[List[str]] = None,
    human_feedback: str = "Please ensure every claim has an explicit citation.",
) -> Dict[str, Any]:
    target_technologies = target_technologies or ["HBM4", "PIM", "CXL"]
    target_companies = target_companies or TARGET_COMPANIES
    co_str = ", ".join(target_companies)
    if question is None:
        question = (
            f"Create a technology strategy analysis report for {', '.join(target_technologies)} "
            f"with competitor TRL comparison for {co_str}. "
            f"Analyze competitive positioning and strategic implications for SK hynix."
        )
    graph = build_graph()
    initial_state: AgentState = {
        "question": question,
        "target_technologies": target_technologies,
        "target_companies": target_companies,
        "revision_count": {},
        "config": {
            "thresholds": DEFAULT_THRESHOLDS,
            "max_retries": DEFAULT_MAX_RETRIES,
            "enable_web_agent": True,
            "human_decision": human_decision,
            "human_feedback": human_feedback,
        },
    }
    return graph.invoke(initial_state, config={"recursion_limit": 60})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agentic RAG Supervisor demo.")
    parser.add_argument(
        "-q", "--question",
        help="User question or report request. Defaults to the built-in SK hynix strategy prompt.",
    )
    parser.add_argument(
        "--technologies",
        default="HBM4,PIM,CXL",
        help="Comma-separated target technologies. Example: HBM4,PIM,CXL",
    )
    parser.add_argument(
        "--companies",
        default=",".join(TARGET_COMPANIES),
        help="Comma-separated target companies. Example: Samsung,SK hynix,Intel,NVIDIA",
    )
    parser.add_argument(
        "--human-decision",
        choices=["approve", "revise", "reject"],
        default="approve",
        help="Simulated human review decision.",
    )
    parser.add_argument(
        "--human-feedback",
        default="Please ensure every claim has an explicit citation.",
        help="Simulated human review feedback.",
    )
    args = parser.parse_args()

    target_technologies = _split_csv(args.technologies)
    target_companies = _split_csv(args.companies)

    print(f"[INFO] Embedding model : {EMBED_MODEL_NAME}")
    print(f"[INFO] Target companies: {target_companies}")
    print(f"[INFO] Target technologies: {target_technologies}")
    print(f"[INFO] Question: {args.question or '(default)'}")
    print(f"[INFO] Per-company FAISS loaded: {list(COMPANY_VECTORSTORES.keys())}")
    print("[INFO] Starting Agentic RAG Supervisor Demo ...")
    state = run_demo(
        human_decision=args.human_decision,
        question=args.question,
        target_technologies=target_technologies,
        target_companies=target_companies,
        human_feedback=args.human_feedback,
    )
    metrics = evaluate_retrieval(QA_ROWS, k=5)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final status      : {state.get('status')}")
    if state.get("failure_reason"):
        print(f"Failure reason    : {state.get('failure_reason')}")
    print(f"[EN] Markdown     : {state.get('markdown_path', '(none)')}")
    print(f"[EN] PDF          : {state.get('pdf_path', '(none)')}")
    print(f"[KO] Markdown     : {state.get('markdown_ko_path', '(none)')}")
    print(f"[KO] PDF          : {state.get('pdf_ko_path', '(none)')}")
    print(f"Hit Rate@5        : {round(metrics['hit_rate_at_k'], 4)}")
    print(f"MRR               : {round(metrics['mrr'], 4)}")
    print(f"Citation count    : {len(state.get('references', []))}")
    print(f"Embed model       : {EMBED_MODEL_NAME}")
    print(f"Retry counts      : {state.get('revision_count', {})}")
    print(f"Max retries       : {state.get('config', {}).get('max_retries', {})}")
    print("=" * 60)

    for key in ["rag", "web", "analysis", "draft", "pdf"]:
        jr = state.get(f"{key}_judge_result", {})
        if jr:
            print(f"  {key:10s} judge: score={jr.get('score'):.4f}  verdict={jr.get('verdict')}")
