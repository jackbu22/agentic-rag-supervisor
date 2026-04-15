"""
Backwards-compatible entrypoint for the Agentic RAG Supervisor demo.

The implementation was split into a package under `agentic_rag_supervisor/demo/`
to make the codebase easier to maintain.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from agentic_rag_supervisor.demo.cli import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

