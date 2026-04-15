from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict


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

