from __future__ import annotations

import argparse

def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    # NOTE: keep this module import-light so `--help` works even when optional
    # dependencies (LangChain stack) are not installed or version-mismatched.
    parser = argparse.ArgumentParser(description="Run the Agentic RAG Supervisor demo.")
    parser.add_argument(
        "-q",
        "--question",
        help="User question or report request. Defaults to the built-in SK hynix strategy prompt.",
    )
    parser.add_argument(
        "--technologies",
        default="HBM4,PIM,CXL",
        help="Comma-separated target technologies. Example: HBM4,PIM,CXL",
    )
    parser.add_argument(
        "--companies",
        default="Samsung,SK hynix,Intel,NVIDIA",
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
    args = parser.parse_args(argv)

    from agentic_rag_supervisor.demo import runtime, settings
    from agentic_rag_supervisor.demo.graph import run_demo

    target_technologies = _split_csv(args.technologies)
    target_companies = _split_csv(args.companies)

    runtime.initialize()

    print(f"[INFO] Embedding model : {settings.EMBED_MODEL_NAME}")
    print(f"[INFO] Target companies: {target_companies}")
    print(f"[INFO] Target technologies: {target_technologies}")
    print(f"[INFO] Question: {args.question or '(default)'}")
    print(f"[INFO] Per-company FAISS loaded: {list(runtime.COMPANY_VECTORSTORES.keys())}")
    print("[INFO] Starting Agentic RAG Supervisor Demo ...")

    state = run_demo(
        human_decision=args.human_decision,
        question=args.question,
        target_technologies=target_technologies,
        target_companies=target_companies,
        human_feedback=args.human_feedback,
    )
    metrics = runtime.evaluate_retrieval(runtime.QA_ROWS, k=5) if runtime.QA_ROWS else {"hit_rate_at_k": 0.0, "mrr": 0.0}

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
    print(f"Embed model       : {settings.EMBED_MODEL_NAME}")
    print(f"Retry counts      : {state.get('revision_count', {})}")
    print(f"Max retries       : {state.get('config', {}).get('max_retries', {})}")
    print("=" * 60)

    for key in ["rag", "web", "analysis", "draft", "pdf"]:
        jr = state.get(f"{key}_judge_result", {})
        if jr:
            print(f"  {key:10s} judge: score={jr.get('score'):.4f}  verdict={jr.get('verdict')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
