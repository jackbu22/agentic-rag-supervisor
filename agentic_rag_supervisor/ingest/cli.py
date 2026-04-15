from __future__ import annotations

import argparse

from agentic_rag_supervisor.ingest.core import ingest_all, show_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="반도체 논문/IR 자료 수집 및 FAISS 인덱스 구축")
    parser.add_argument("--force", action="store_true", help="이미 인덱싱된 문서도 강제로 다시 처리")
    parser.add_argument("--rebuild-only", action="store_true", help="자료 파싱 없이 FAISS 인덱스만 재구축")
    parser.add_argument("--status", action="store_true", help="현재 수집 현황 확인")
    args = parser.parse_args(argv)

    if args.status:
        show_status()
        return 0

    ingest_all(force=args.force, rebuild_only=args.rebuild_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

