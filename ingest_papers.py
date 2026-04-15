"""
ingest_papers.py
================
논문 PDF / Markdown / IR 자료를 오프라인으로 수집해서
  1. demo_semiconductor_sources.json 에 병합
  2. 기업별 FAISS 인덱스 구축 (faiss_db/{company}_index/)
  3. 전체 병합 FAISS 인덱스 구축 (faiss_db/merged_index/)

폴더 구조:
  data/raw_pdfs/
  ├── samsung/       ← 삼성 관련 PDF
  ├── skhynix/       ← SK하이닉스 관련 PDF
  ├── intel/         ← 인텔 관련 PDF
  └── nvidia/        ← 엔비디아 관련 PDF

파일명 규칙 (자동 메타데이터 추출):
  {technology}_{연도}_{설명}.pdf
  예: HBM4_2025_hot_chips_keynote.pdf
      CXL_2025_product_announcement.pdf
      PIM_2025_isscc_paper.pdf

※ 폴더명에서 기업명이 자동으로 추출되므로 파일명에 회사명 불필요

사용법:
  python ingest_papers.py                # 새 자료만 추가
  python ingest_papers.py --force        # 전체 재처리
  python ingest_papers.py --rebuild-only # PDF 파싱 없이 인덱스만 재구축
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from datetime import datetime
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent
DATA_DIR      = ROOT_DIR / "data"
PDF_ROOT_DIR  = DATA_DIR / "raw_pdfs"
SOURCES_FILE  = DATA_DIR / "demo_semiconductor_sources.json"
FAISS_DB_ROOT = ROOT_DIR / "faiss_db"
SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".md", ".mdx", ".txt"}

# ── 기업 슬러그 → 표준 기업명 매핑 ─────────────────────────────────────────
SLUG_TO_COMPANY: dict[str, str] = {
    "samsung":  "Samsung",
    "skhynix":  "SK hynix",
    "intel":    "Intel",
    "nvidia":   "NVIDIA",
}

# ── 기업별 FAISS 경로 ────────────────────────────────────────────────────────
COMPANY_FAISS_PATHS: dict[str, Path] = {
    company: FAISS_DB_ROOT / f"{slug}_index"
    for slug, company in SLUG_TO_COMPANY.items()
}
MERGED_FAISS_PATH = FAISS_DB_ROOT / "merged_index"

# ── source_type 자동 추론 키워드 ─────────────────────────────────────────────
SOURCE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "paper":                ["paper", "arxiv", "isscc", "vlsi", "iedm", "hotchips",
                             "hot_chips", "symposium", "journal", "letters"],
    "patent":               ["patent"],
    "earnings_call":        ["earnings", "analyst", "investor", "ir_", "annual",
                             "quarterly", "q1", "q2", "q3", "q4"],
    "press_release":        ["press", "news", "release", "newsroom"],
    "product_announcement": ["product", "launch", "announce", "sample", "roadmap"],
    "conference":           ["conference", "conf", "workshop", "presentation", "keynote"],
    "standard":             ["standard", "jedec", "spec"],
}

# ── 기술명 정규화 ────────────────────────────────────────────────────────────
TECH_MAP: dict[str, str] = {
    "HBM":  "HBM4", "HBM3": "HBM4", "HBM3E": "HBM4", "HBM4": "HBM4",
    "PIM":  "PIM",  "AXDIMM": "PIM", "NMP": "PIM",
    "CXL":  "CXL",
    "DRAM": "HBM4", "MEMORY": "HBM4",
}


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════════════

def infer_source_type(desc: str) -> str:
    d = desc.lower()
    for stype, keywords in SOURCE_TYPE_KEYWORDS.items():
        if any(k in d for k in keywords):
            return stype
    return "industry_report"


def normalize_tech(raw: str) -> str:
    return TECH_MAP.get(raw.upper(), raw.upper())


def detect_tech_from_text(text: str) -> str:
    """파일명 또는 내용에서 기술 분류를 감지한다."""
    t = text.upper()
    if "HBM4" in t or "HBM3E" in t or "HBM3" in t or "HBM" in t:
        return "HBM4"
    if "PIM" in t or "AXDIMM" in t or "NMP" in t or "PROCESSING.IN.MEMORY" in t:
        return "PIM"
    if "CXL" in t or "COMPUTE EXPRESS LINK" in t:
        return "CXL"
    # NVIDIA 관련 → GPU 메모리 = HBM4
    if "BLACKWELL" in t or "HOPPER" in t or "GPU" in t or "NVIDIA" in t:
        return "HBM4"
    # Intel Foundry / Xeon → CXL
    if "XEON" in t or "GAUDI" in t or "FOUNDRY" in t:
        return "CXL"
    return "HBM4"  # 기본값


def detect_year_from_text(text: str) -> str:
    """텍스트에서 연도를 감지한다."""
    import re
    # 2020~2029 범위 연도 탐색
    matches = re.findall(r"20[2-9][0-9]", text)
    if matches:
        # 가장 최근 연도 반환
        return sorted(set(matches))[-1]
    return str(datetime.now().year)


def parse_filename(pdf_path: Path, company: str) -> dict:
    """파일명에서 메타데이터를 추출한다.
    권장 형식: {technology}_{연도}_{설명}.pdf
    임의 파일명(한글, 공백 등)도 내용 기반으로 자동 처리한다.
    """
    stem = pdf_path.stem
    slug_co = next((s for s, c in SLUG_TO_COMPANY.items() if c == company), "unknown")
    uid     = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8].upper()

    # ── 기술명 감지: 파일명에서 우선 탐지 ───────────────────────────────────
    technology = detect_tech_from_text(stem)

    # ── 연도 감지: 파일명 → 없으면 현재 연도 ────────────────────────────────
    year = detect_year_from_text(stem)

    # ── source_type 감지 ────────────────────────────────────────────────────
    source_type = infer_source_type(stem)

    doc_id = f"{slug_co.upper()}-{year}-{uid}"

    return {
        "doc_id":       doc_id,
        "title":        stem,          # 원래 파일명을 title로 사용
        "technology":   technology,
        "company":      company,
        "source_type":  source_type,
        "published_at": f"{year}-01-01",
        "source_url":   str(pdf_path.resolve()),
    }


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 30) -> str:
    try:
        import fitz
        doc   = fitz.open(str(pdf_path))
        pages = [page.get_text() for i, page in enumerate(doc) if i < max_pages]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        print("[ERROR] PyMuPDF 미설치: pip install pymupdf")
        sys.exit(1)
    except Exception as e:
        print(f"[WARN] 텍스트 추출 실패 ({pdf_path.name}): {e}")
        return ""


def extract_text_from_source(source_path: Path) -> str:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(source_path)
    if suffix in {".md", ".mdx", ".txt"}:
        try:
            return source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return source_path.read_text(encoding="utf-8-sig")
        except Exception as e:
            print(f"[WARN] 텍스트 추출 실패 ({source_path.name}): {e}")
            return ""
    return ""


def make_content_summary(full_text: str, max_chars: int = 600) -> str:
    lines  = [line.strip() for line in full_text.splitlines() if line.strip()]
    joined = " ".join(lines)
    return joined[:max_chars]


# ═══════════════════════════════════════════════════════════════════════════
# FAISS 인덱스 구축
# ═══════════════════════════════════════════════════════════════════════════

def rebuild_all_indexes(all_records: list) -> None:
    """전체 레코드로 기업별 + 병합 FAISS 인덱스를 재구축한다."""
    print("\n[INFO] FAISS 인덱스 재구축 시작 ...")

    try:
        from agentic_rag_supervisor_demo import EMBED_MODEL, build_chunks
        from langchain_community.vectorstores import FAISS
    except ImportError as e:
        print(f"[ERROR] 임포트 실패: {e}")
        print("[INFO] FAISS 재구축 건너뜀. 수동으로 실행하세요.")
        return

    all_chunks = build_chunks(all_records)
    FAISS_DB_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 기업별 인덱스 ──────────────────────────────────────────────────────
    for company, path in COMPANY_FAISS_PATHS.items():
        co_chunks = [c for c in all_chunks if c.metadata.get("company") == company]
        if not co_chunks:
            print(f"  [SKIP] {company}: 청크 없음")
            continue
        vs = FAISS.from_documents(co_chunks, EMBED_MODEL)
        vs.save_local(str(path))
        print(f"  [OK] {company}: {len(co_chunks)}청크 → {path.name}")

    # ── 병합 인덱스 ───────────────────────────────────────────────────────
    if all_chunks:
        vs_merged = FAISS.from_documents(all_chunks, EMBED_MODEL)
        vs_merged.save_local(str(MERGED_FAISS_PATH))
        print(f"  [OK] merged : {len(all_chunks)}청크 → {MERGED_FAISS_PATH.name}")

    print(f"[INFO] FAISS 인덱스 재구축 완료")


# ═══════════════════════════════════════════════════════════════════════════
# 수집 메인 로직
# ═══════════════════════════════════════════════════════════════════════════

def ingest_all(force: bool = False, rebuild_only: bool = False) -> None:
    """
    raw_pdfs/{company}/ 폴더의 모든 PDF/Markdown/TXT 자료를 처리해서
    sources.json에 추가하고 FAISS 인덱스를 재구축한다.
    """
    # 기존 sources 로드
    if SOURCES_FILE.exists():
        with SOURCES_FILE.open("r", encoding="utf-8-sig") as f:
            existing_records: list = json.load(f)
    else:
        existing_records = []

    existing_ids = {r["doc_id"] for r in existing_records}

    # rebuild_only 모드: 파싱 없이 인덱스만 재구축
    if rebuild_only:
        print("[INFO] --rebuild-only 모드: FAISS 인덱스만 재구축합니다.")
        rebuild_all_indexes(existing_records)
        return

    # 회사 폴더 초기화
    for slug in SLUG_TO_COMPANY:
        (PDF_ROOT_DIR / slug).mkdir(parents=True, exist_ok=True)

    new_records: list = []
    total_sources = 0

    for slug, company in SLUG_TO_COMPANY.items():
        company_dir = PDF_ROOT_DIR / slug
        source_files = sorted(
            p for p in company_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
        )

        if not source_files:
            print(f"[{company}] 처리할 자료 없음 ({company_dir})")
            continue

        print(f"\n[{company}] {len(source_files)}개 자료 발견")
        total_sources += len(source_files)

        for source_path in source_files:
            meta   = parse_filename(source_path, company)
            doc_id = meta["doc_id"]

            if doc_id in existing_ids and not force:
                print(f"  [SKIP] {doc_id} ({source_path.name})")
                continue

            print(f"  [처리] {source_path.name}")
            print(f"         doc_id={doc_id}  tech={meta['technology']}"
                  f"  type={meta['source_type']}")

            full_text = extract_text_from_source(source_path)
            if not full_text.strip():
                print(f"         [WARN] 텍스트 없음, 건너뜀")
                continue

            # 내용으로 기술명/연도 보완 감지
            if meta["technology"] == "HBM4" and detect_tech_from_text(meta["title"]) == "HBM4":
                # 파일명에서 기술 감지 못한 경우 내용 앞부분으로 재시도
                content_tech = detect_tech_from_text(full_text[:2000])
                meta["technology"] = content_tech

            content_year = detect_year_from_text(full_text[:3000])
            if content_year != str(datetime.now().year):
                meta["published_at"] = f"{content_year}-01-01"

            meta["content"] = make_content_summary(full_text)
            new_records.append(meta)
            existing_ids.add(doc_id)
            print(f"         content: {meta['content'][:80]}...")

    if total_sources == 0:
        print(f"\n[INFO] 처리할 자료 없음. 아래 경로에 PDF/MDX/MD/TXT를 넣으세요:")
        for slug in SLUG_TO_COMPANY:
            print(f"  {(PDF_ROOT_DIR / slug).resolve()}")
        return

    if not new_records:
        print("\n[INFO] 새 문서 없음 (기존 인덱스 유지).")
        print("[TIP]  강제 재처리: python ingest_papers.py --force")
        return

    # sources.json 저장
    all_records = existing_records + new_records
    with SOURCES_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] {len(new_records)}개 문서 추가 → {SOURCES_FILE}")
    print(f"[INFO] 전체 문서 수: {len(all_records)}")

    # FAISS 재구축
    rebuild_all_indexes(all_records)

    # 완료 안내
    print("\n" + "=" * 55)
    print("수집 완료!")
    print("=" * 55)
    co_counts = {}
    for r in new_records:
        co_counts[r["company"]] = co_counts.get(r["company"], 0) + 1
    for co, cnt in co_counts.items():
        print(f"  {co:<12}: {cnt}개 문서")
    print("\n다음 명령으로 분석을 실행하세요:")
    print("  python agentic_rag_supervisor_demo.py")


# ═══════════════════════════════════════════════════════════════════════════
# 현황 확인 커맨드
# ═══════════════════════════════════════════════════════════════════════════

def show_status() -> None:
    """현재 수집 현황을 출력한다."""
    print("\n[ 수집 현황 ]")
    print("-" * 50)

    # sources.json
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

    # FAISS 인덱스
    print("FAISS 인덱스:")
    for slug, company in SLUG_TO_COMPANY.items():
        path = FAISS_DB_ROOT / f"{slug}_index"
        status = "OK" if path.exists() else "없음"
        print(f"  {company:<15}: {status}")
    merged_status = "OK" if MERGED_FAISS_PATH.exists() else "없음"
    print(f"  {'merged':<15}: {merged_status}")

    print()

    # raw_pdfs 폴더
    print("raw_pdfs 폴더:")
    for slug, company in SLUG_TO_COMPANY.items():
        d = PDF_ROOT_DIR / slug
        source_files = [
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
        ] if d.exists() else []
        print(f"  {company:<15}: {len(source_files)}개 자료  ({d})")

    print("-" * 50)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="반도체 논문/IR 자료 수집 및 FAISS 인덱스 구축"
    )
    parser.add_argument("--force",        action="store_true",
                        help="이미 인덱싱된 문서도 강제로 다시 처리")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="자료 파싱 없이 FAISS 인덱스만 재구축")
    parser.add_argument("--status",       action="store_true",
                        help="현재 수집 현황 확인")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        ingest_all(force=args.force, rebuild_only=args.rebuild_only)
