from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from agentic_rag_supervisor.paths import DATA_DIR, FAISS_DB_ROOT

PDF_ROOT_DIR = DATA_DIR / "raw_pdfs"
SOURCES_FILE = DATA_DIR / "demo_semiconductor_sources.json"
SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".md", ".mdx", ".txt"}

SLUG_TO_COMPANY: dict[str, str] = {
    "samsung": "Samsung",
    "skhynix": "SK hynix",
    "intel": "Intel",
    "nvidia": "NVIDIA",
}

COMPANY_FAISS_PATHS: dict[str, Path] = {
    company: FAISS_DB_ROOT / f"{slug}_index" for slug, company in SLUG_TO_COMPANY.items()
}
MERGED_FAISS_PATH = FAISS_DB_ROOT / "merged_index"

SOURCE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "paper": ["paper", "arxiv", "isscc", "vlsi", "iedm", "hotchips", "hot_chips", "symposium", "journal", "letters"],
    "patent": ["patent"],
    "earnings_call": ["earnings", "analyst", "investor", "ir_", "annual", "quarterly", "q1", "q2", "q3", "q4"],
    "press_release": ["press", "news", "release", "newsroom"],
    "product_announcement": ["product", "launch", "announce", "sample", "roadmap"],
    "conference": ["conference", "conf", "workshop", "presentation", "keynote"],
    "standard": ["standard", "jedec", "spec"],
}

TECH_MAP: dict[str, str] = {
    "HBM": "HBM4",
    "HBM3": "HBM4",
    "HBM3E": "HBM4",
    "HBM4": "HBM4",
    "PIM": "PIM",
    "AXDIMM": "PIM",
    "NMP": "PIM",
    "CXL": "CXL",
    "DRAM": "HBM4",
    "MEMORY": "HBM4",
}


def infer_source_type(desc: str) -> str:
    d = desc.lower()
    for stype, keywords in SOURCE_TYPE_KEYWORDS.items():
        if any(k in d for k in keywords):
            return stype
    return "industry_report"


def normalize_tech(raw: str) -> str:
    return TECH_MAP.get(raw.upper(), raw.upper())


def detect_tech_from_text(text: str) -> str:
    t = text.upper()
    if "HBM4" in t or "HBM3E" in t or "HBM3" in t or "HBM" in t:
        return "HBM4"
    if "PIM" in t or "AXDIMM" in t or "NMP" in t or "PROCESSING.IN.MEMORY" in t:
        return "PIM"
    if "CXL" in t or "COMPUTE EXPRESS LINK" in t:
        return "CXL"
    if "BLACKWELL" in t or "HOPPER" in t or "GPU" in t or "NVIDIA" in t:
        return "HBM4"
    if "XEON" in t or "GAUDI" in t or "FOUNDRY" in t:
        return "CXL"
    return "HBM4"


def detect_year_from_text(text: str) -> str:
    import re

    m = re.search(r"(20\\d{2})", text)
    if m:
        return m.group(1)
    return str(datetime.now().year)


def parse_filename(source_path: Path, company: str) -> dict:
    name = source_path.stem
    parts = name.split("_")
    tech_raw = parts[0] if parts else "HBM4"
    year_raw = parts[1] if len(parts) > 1 else str(datetime.now().year)
    desc = "_".join(parts[2:]) if len(parts) > 2 else name

    tech = normalize_tech(tech_raw)
    year = year_raw if year_raw.isdigit() else str(datetime.now().year)
    published_at = f"{year}-01-01"
    source_type = infer_source_type(desc)

    return {
        "doc_id": "",
        "title": f"{tech} {year} {desc.replace('_', ' ')}",
        "technology": tech,
        "company": company,
        "source_type": source_type,
        "published_at": published_at,
        "source_url": str(source_path),
        "content": "",
    }


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 30) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    texts: list[str] = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        texts.append(page.get_text("text") or "")
    doc.close()
    return "\n".join(texts).strip()


def extract_text_from_source(source_path: Path) -> str:
    if source_path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(source_path)
    return source_path.read_text(encoding="utf-8", errors="ignore").strip()


def make_content_summary(full_text: str, max_chars: int = 600) -> str:
    s = " ".join(full_text.split())
    return s[:max_chars]


def rebuild_all_indexes(all_records: list[dict]) -> None:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    merged_docs: list[Document] = [Document(page_content=r["content"], metadata=r) for r in all_records]
    if merged_docs:
        merged_vs = FAISS.from_documents(merged_docs, emb)
        FAISS_DB_ROOT.mkdir(parents=True, exist_ok=True)
        merged_vs.save_local(str(MERGED_FAISS_PATH))

    for company, path in COMPANY_FAISS_PATHS.items():
        co_docs = [Document(page_content=r["content"], metadata=r) for r in all_records if r["company"] == company]
        if not co_docs:
            continue
        vs = FAISS.from_documents(co_docs, emb)
        path.parent.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(path))


def ingest_all(force: bool = False, rebuild_only: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    existing_records: list[dict] = []
    if SOURCES_FILE.exists():
        with SOURCES_FILE.open("r", encoding="utf-8-sig") as f:
            existing_records = json.load(f)

    existing_ids = {r.get("doc_id") for r in existing_records if r.get("doc_id")}

    if rebuild_only:
        if not existing_records:
            print(f"[WARN] {SOURCES_FILE} 에 기존 기록이 없습니다. 먼저 수집을 수행하세요.")
            return
        rebuild_all_indexes(existing_records)
        print("[INFO] 인덱스 재구축 완료")
        return

    new_records: list[dict] = []
    total_sources = 0

    for slug, company in SLUG_TO_COMPANY.items():
        company_dir = PDF_ROOT_DIR / slug
        company_dir.mkdir(parents=True, exist_ok=True)

        sources = [p for p in company_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS]
        if not sources:
            continue

        print(f"\n[INFO] {company} 자료 {len(sources)}개 스캔: {company_dir}")
        for source_path in sources:
            total_sources += 1
            meta = parse_filename(source_path, company=company)
            raw_id = f"{company}|{meta['technology']}|{meta['published_at']}|{source_path.name}"
            doc_id = hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:12]
            meta["doc_id"] = doc_id

            if (doc_id in existing_ids) and not force:
                continue

            print(f"  - {source_path.name}")
            try:
                full_text = extract_text_from_source(source_path)
            except Exception as e:
                print(f"         [WARN] 파싱 실패: {e}")
                continue

            if not full_text.strip():
                print("         [WARN] 텍스트 없음, 건너뜀")
                continue

            if meta["technology"] == "HBM4" and detect_tech_from_text(meta["title"]) == "HBM4":
                meta["technology"] = detect_tech_from_text(full_text[:2000])

            content_year = detect_year_from_text(full_text[:3000])
            if content_year != str(datetime.now().year):
                meta["published_at"] = f"{content_year}-01-01"

            meta["content"] = make_content_summary(full_text)
            new_records.append(meta)
            existing_ids.add(doc_id)
            print(f"         content: {meta['content'][:80]}...")

    if total_sources == 0:
        print("\n[INFO] 처리할 자료 없음. 아래 경로에 PDF/MDX/MD/TXT를 넣으세요:")
        for slug in SLUG_TO_COMPANY:
            print(f"  {(PDF_ROOT_DIR / slug).resolve()}")
        return

    if not new_records:
        print("\n[INFO] 새 문서 없음 (기존 인덱스 유지).")
        print("[TIP]  강제 재처리: python ingest_papers.py --force")
        return

    all_records = existing_records + new_records
    with SOURCES_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] {len(new_records)}개 문서 추가 → {SOURCES_FILE}")
    print(f"[INFO] 전체 문서 수: {len(all_records)}")

    rebuild_all_indexes(all_records)

    print("\n" + "=" * 55)
    print("수집 완료!")
    print("=" * 55)
    co_counts: dict[str, int] = {}
    for r in new_records:
        co_counts[r["company"]] = co_counts.get(r["company"], 0) + 1
    for co, cnt in co_counts.items():
        print(f"  {co:<12}: {cnt}개 문서")
    print("\n다음 명령으로 분석을 실행하세요:")
    print("  python agentic_rag_supervisor_demo.py")


def show_status() -> None:
    print("\n[ 수집 현황 ]")
    print("-" * 50)

    if SOURCES_FILE.exists():
        with SOURCES_FILE.open("r", encoding="utf-8-sig") as f:
            records = json.load(f)
        print(f"sources.json: {len(records)}개 문서")
        from collections import Counter

        co_counts = Counter(r["company"] for r in records)
        for co, cnt in sorted(co_counts.items()):
            print(f"  {co:<15}: {cnt}개")
    else:
        print("sources.json: 없음")

    print()

    print("FAISS 인덱스:")
    for slug, company in SLUG_TO_COMPANY.items():
        path = FAISS_DB_ROOT / f"{slug}_index"
        status = "OK" if path.exists() else "없음"
        print(f"  {company:<15}: {status}")
    merged_status = "OK" if MERGED_FAISS_PATH.exists() else "없음"
    print(f"  {'merged':<15}: {merged_status}")

    print()

    print("raw_pdfs 폴더:")
    for slug, company in SLUG_TO_COMPANY.items():
        d = PDF_ROOT_DIR / slug
        source_files = (
            [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS]
            if d.exists()
            else []
        )
        print(f"  {company:<15}: {len(source_files)}개 자료  ({d})")

    print("-" * 50)
