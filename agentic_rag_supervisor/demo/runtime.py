from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag_supervisor.demo import settings


# ── module-level retriever state (initialized by initialize()) ──────────────
SOURCE_RECORDS: List[Dict[str, Any]] = []
QA_ROWS: List[Dict[str, Any]] = []
CHUNKS: List[Document] = []
VECTORSTORE: Optional[FAISS] = None
BM25: Optional[BM25Retriever] = None
COMPANY_VECTORSTORES: Dict[str, FAISS] = {}
_INITIALIZED = False

# ── optional lexical retrievers (BM25 / TF-IDF) ─────────────────────────────
_BM25 = None
_TFIDF_WORD = None
_TFIDF_WORD_X = None


def load_sources() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with settings.SOURCES_FILE.open("r", encoding="utf-8-sig") as f:
        source_records = json.load(f)
    with settings.QA_FILE.open("r", encoding="utf-8-sig") as f:
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


def _load_or_build_faiss(path: Path, chunks: List[Document], label: str) -> Optional[FAISS]:
    if not chunks:
        return None
    if path.exists():
        try:
            vs = FAISS.load_local(str(path), settings.EMBED_MODEL, allow_dangerous_deserialization=True)
            print(f"[INFO] FAISS loaded  : {label} ({path.name})")
            return vs
        except Exception as e:
            print(f"[WARN] FAISS load failed ({label}): {e} — rebuilding")
    vs = FAISS.from_documents(chunks, settings.EMBED_MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    print(f"[INFO] FAISS built   : {label} ({len(chunks)} chunks) → {path}")
    return vs


def load_or_build_retrievers(chunks: List[Document]) -> Tuple[Optional[FAISS], Optional[BM25Retriever], Dict[str, FAISS]]:
    settings.FAISS_DB_ROOT.mkdir(parents=True, exist_ok=True)

    bm25 = BM25Retriever.from_documents(chunks) if chunks else None
    if bm25:
        bm25.k = 10

    merged_vs = _load_or_build_faiss(settings.FAISS_INDEX_PATH, chunks, "merged")

    company_stores: Dict[str, FAISS] = {}
    for company, path in settings.COMPANY_FAISS_PATHS.items():
        co_chunks = [c for c in chunks if c.metadata.get("company") == company]
        vs = _load_or_build_faiss(path, co_chunks, company)
        if vs:
            company_stores[company] = vs

    if settings.WEB_CACHE_FAISS_PATH.exists() and merged_vs is not None:
        try:
            web_vs = FAISS.load_local(
                str(settings.WEB_CACHE_FAISS_PATH),
                settings.EMBED_MODEL,
                allow_dangerous_deserialization=True,
            )
            merged_vs.merge_from(web_vs)
            print("[INFO] Web-cache FAISS merged into main index")
        except Exception as e:
            print(f"[WARN] Web-cache FAISS merge failed: {e}")

    return merged_vs, bm25, company_stores


def initialize(force: bool = False) -> None:
    global _INITIALIZED, SOURCE_RECORDS, QA_ROWS, CHUNKS, VECTORSTORE, BM25, COMPANY_VECTORSTORES
    global _BM25, _TFIDF_WORD, _TFIDF_WORD_X
    if _INITIALIZED and not force:
        return
    SOURCE_RECORDS, QA_ROWS = load_sources()
    CHUNKS = build_chunks(SOURCE_RECORDS)
    VECTORSTORE, BM25, COMPANY_VECTORSTORES = load_or_build_retrievers(CHUNKS)

    # Lexical indexes (doc-level). Useful for benchmarking and as a fallback when
    # dense/FAISS isn't available.
    try:
        import re

        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer

        token_re = re.compile(r"[A-Za-z0-9_]+")

        def tokenize(text: str) -> list[str]:
            return token_re.findall(text.lower())

        doc_texts = [(str(r.get("title", "")) + "\n" + str(r.get("content", ""))).strip() for r in SOURCE_RECORDS]
        _BM25 = BM25Okapi([tokenize(t) for t in doc_texts])
        _TFIDF_WORD = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        _TFIDF_WORD_X = _TFIDF_WORD.fit_transform(doc_texts)
    except Exception:
        _BM25 = None
        _TFIDF_WORD = None
        _TFIDF_WORD_X = None

    _INITIALIZED = True


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
    aliases = {"HBM": "HBM4", "HBM3": "HBM4", "HBM3E": "HBM4", "HBM4": "HBM4", "CXL": "CXL", "PIM": "PIM"}
    return aliases.get(label, label)


def evidence_covers_target(evidence: Dict[str, Any], target_tech: str) -> bool:
    target = canonical_tech_label(target_tech)
    meta_tech = canonical_tech_label(str(evidence.get("technology", "")))
    if target in meta_tech:
        return True
    text = (evidence.get("excerpt") or "").upper()
    return target in text


def recency_days_for_doc(doc: Dict[str, Any], tech_hint: str) -> int:
    tech = canonical_tech_label(tech_hint)
    if "standard" in str(doc.get("source_type", "")).lower():
        tech = "STANDARD"
    return int(settings.RECENCY_WINDOW_DAYS.get(tech, 365))


def is_recent(doc: Dict[str, Any], tech_hint: str) -> bool:
    pub = doc.get("published_at")
    if not pub:
        return False
    try:
        dt = parse_date(str(pub))
    except Exception:
        return False
    window = recency_days_for_doc(doc, tech_hint)
    delta = datetime.now(tz=timezone.utc) - dt
    return delta.days <= window


def rank_fusion(dense: List[Document], sparse: List[Document], alpha: float = 0.5) -> List[Document]:
    """Reciprocal rank fusion — Dense 0.5 : BM25 0.5 (spec 4-2)."""
    scores: Dict[str, float] = {}
    for rank, doc in enumerate(dense, start=1):
        cid = doc.metadata.get("chunk_id", "")
        scores[cid] = scores.get(cid, 0.0) + alpha * (1.0 / (rank + 50))
    for rank, doc in enumerate(sparse, start=1):
        cid = doc.metadata.get("chunk_id", "")
        scores[cid] = scores.get(cid, 0.0) + (1.0 - alpha) * (1.0 / (rank + 50))
    merged = {d.metadata.get("chunk_id", ""): d for d in dense + sparse}
    return [merged[cid] for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True) if cid in merged]


def hybrid_search(
    query: str,
    tech_hint: Optional[str] = None,
    company: Optional[str] = None,
    top_k: int = 5,
    require_recent: bool = True,
) -> List[Document]:
    initialize()
    mode = os.environ.get("ARS_RETRIEVER_MODE", "hybrid").strip().lower()
    if mode in {"bm25", "tfidf_word"} and _BM25 is not None and _TFIDF_WORD is not None and _TFIDF_WORD_X is not None:
        return _lexical_search(query, mode=mode, top_k=top_k, company=company)

    tech = tech_hint or detect_tech_hint(query)
    dense_docs: List[Document] = []
    sparse_docs: List[Document] = []

    if company and company in COMPANY_VECTORSTORES:
        vs = COMPANY_VECTORSTORES[company]
        dense_docs = vs.similarity_search(query, k=max(top_k, 6))
    elif VECTORSTORE is not None:
        dense_docs = VECTORSTORE.similarity_search(query, k=max(top_k, 6))

    if BM25 is not None:
        sparse_docs_raw = BM25.invoke(query)
        if company:
            sparse_docs_raw = [d for d in sparse_docs_raw if d.metadata.get("company") == company]
        sparse_docs = sparse_docs_raw[: max(top_k, 6)]

    fused = rank_fusion(dense_docs, sparse_docs, alpha=0.5)
    if require_recent:
        out: List[Document] = []
        for d in fused:
            if is_recent(d.metadata, tech):
                out.append(d)
        if len(out) >= min(top_k, 3):
            fused = out
    return fused[:top_k]


def _lexical_search(query: str, mode: str, top_k: int, company: Optional[str]) -> List[Document]:
    """Doc-level lexical retrieval over SOURCE_RECORDS."""
    assert mode in {"bm25", "tfidf_word"}

    if not SOURCE_RECORDS:
        return []

    rows = SOURCE_RECORDS
    if company:
        rows = [r for r in rows if r.get("company") == company]
        if not rows:
            rows = SOURCE_RECORDS

    # Map subset back to original indices for shared matrices
    id_to_idx: dict[str, int] = {str(r["doc_id"]): i for i, r in enumerate(SOURCE_RECORDS)}
    candidate_ids = [str(r["doc_id"]) for r in rows]
    candidate_indices = [id_to_idx[d] for d in candidate_ids if d in id_to_idx]

    if mode == "bm25":
        import re

        token_re = re.compile(r"[A-Za-z0-9_]+")
        tokens = token_re.findall(query.lower())
        scores = _BM25.get_scores(tokens)
        ranked = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]
    else:
        qv = _TFIDF_WORD.transform([query])
        sims = (_TFIDF_WORD_X @ qv.T).toarray().ravel()
        ranked = sorted(candidate_indices, key=lambda i: sims[i], reverse=True)[:top_k]

    out: List[Document] = []
    for i in ranked:
        r = SOURCE_RECORDS[i]
        meta = {
            "doc_id": r.get("doc_id"),
            "chunk_id": f"doc-{r.get('doc_id')}",
            "title": r.get("title"),
            "technology": r.get("technology"),
            "company": r.get("company"),
            "source_type": r.get("source_type"),
            "published_at": r.get("published_at"),
            "source_url": r.get("source_url"),
        }
        out.append(Document(page_content=str(r.get("content", "")), metadata=meta))
    return out


def literal_source_matches(target_techs: List[str], target_companies: List[str]) -> List[Document]:
    initialize()
    matches: List[Document] = []
    techs = {canonical_tech_label(t) for t in target_techs}
    companies = set(target_companies)
    for ch in CHUNKS:
        t = canonical_tech_label(str(ch.metadata.get("technology", "")))
        co = str(ch.metadata.get("company", ""))
        if t in techs and (not companies or co in companies):
            matches.append(ch)
    return matches[:12]


def evaluate_retrieval(qa_rows: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    initialize()
    hit_count = 0
    reciprocal_ranks: List[float] = []
    for row in qa_rows[: min(len(qa_rows), 20)]:
        query = row.get("question", "")
        tech = detect_tech_hint(query, fallback=row.get("technology", "HBM4"))
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
