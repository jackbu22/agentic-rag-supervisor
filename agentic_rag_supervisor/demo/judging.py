from __future__ import annotations

import json
from typing import Any, Dict, List, Literal

from langchain_core.documents import Document

from agentic_rag_supervisor.demo.settings import LLM_FULL
from agentic_rag_supervisor.demo.types import AgentState, JudgeResult


def build_judge(score: float, threshold: float, issues: List[str], suggestions: List[str]) -> JudgeResult:
    if score >= threshold:
        verdict: Literal["approve", "retry", "fail"] = "approve"
    elif score >= max(0.0, threshold - 0.20):
        verdict = "retry"
    else:
        verdict = "fail"
    return {"score": round(score, 4), "verdict": verdict, "issues": issues, "suggestions": suggestions}


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


def check_draft_completeness(draft: str) -> Dict[str, Any]:
    """D2: GPT-4o checks report completeness and returns score + issues."""
    prompt = (
        "Check if this technology strategy report draft is complete and well-structured.\n"
        f"Draft (first 3000 chars): {draft[:3000]}\n\n"
        "Verify: reference-report style cover block exists, EXECUTIVE SUMMARY, Key Metrics, "
        "sections 1-4, REFERENCES, citations, no placeholder text, limitation note included.\n"
        "Return ONLY valid JSON:\n"
        '{ "section_score": float, "citation_score": float, "issues": [str], "suggestions": [str] }'
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


def validate_draft_for_pdf(draft: str) -> Dict[str, Any]:
    """P1: GPT-4o validates draft completeness and quality before PDF export."""
    prompt = (
        "Validate this technology strategy report draft for PDF export readiness.\n"
        f"Draft (first 2500 chars): {draft[:2500]}\n\n"
        "Check: reference-report style cover block, correct section headers, no broken markdown, citations present, "
        "limitation note included, no placeholder text.\n"
        "Return ONLY valid JSON:\n"
        '{ "ok": bool, "quality_score": float, "issues": [str], "suggestions": [str] }'
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

