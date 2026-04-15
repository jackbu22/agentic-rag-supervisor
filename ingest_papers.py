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


def main(argv: list[str] | None = None) -> int:
    from agentic_rag_supervisor.ingest.cli import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

