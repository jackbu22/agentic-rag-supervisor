from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from agentic_rag_supervisor.demo import runtime, settings
from agentic_rag_supervisor.demo.judging import (
    build_judge,
    check_draft_completeness,
    get_threshold,
    normalize_doc,
    validate_draft_for_pdf,
)
from agentic_rag_supervisor.demo.pdf import (
    inject_visual_markers,
    process_pdf_sources_with_vlm,
    render_pdf,
    validate_pdf_tables_with_vlm,
)
from agentic_rag_supervisor.demo.types import AgentState


def _supervisor_generate_instructions(state: AgentState, agent_key: str) -> str:
    """SR: GPT-4o-mini generates specific, actionable retry instructions."""
    judge = state.get(f"{agent_key}_judge_result", {})
    issues = judge.get("issues", [])
    score = judge.get("score", 0.0)
    threshold = get_threshold(state, agent_key)
    prompt = (
        "You are a research workflow supervisor for a semiconductor competitive intelligence pipeline.\n"
        f"Agent '{agent_key}' scored {score:.3f} (threshold: {threshold:.3f}).\n"
        f"Known issues: {json.dumps(issues)}\n"
        f"Research question: {state.get('question', '')}\n"
        f"Target technologies: {state.get('target_technologies', [])}\n"
        f"Target companies: {state.get('target_companies', [])}\n\n"
        f"Write 2-3 specific, actionable instructions for the '{agent_key}' agent to improve on retry. "
        "Be concise (max 120 words). Focus on exactly what evidence or analysis is missing."
    )
    try:
        resp = settings.LLM_MINI.invoke(prompt)
        return str(resp.content).strip()
    except Exception as ex:
        print(f"[Supervisor SR] LLM error: {ex}")
        return f"Improve {agent_key}: {'; '.join(issues) or 'increase coverage and citation count'}."


def _generate_web_queries(state: AgentState) -> List[str]:
    """W1: GPT-4o-mini generates targeted web search queries."""
    feedback_hint = ""
    if state.get("retry_reason") or state.get("improvement_instructions"):
        feedback_hint = f"\nSupervisor feedback: {state.get('retry_reason', '')} {state.get('improvement_instructions', '')}"
    prompt = (
        "You are a semiconductor competitive intelligence researcher.\n"
        "Generate exactly 4 targeted search query strings as a JSON array.\n"
        f"Main question: {state['question']}\n"
        f"Technologies: {', '.join(state.get('target_technologies', ['HBM4', 'PIM', 'CXL']))}\n"
        f"Companies: {', '.join(state.get('target_companies', ['SK hynix', 'Samsung', 'Micron']))}"
        f"{feedback_hint}\n\n"
        "Vary query focus: (1) supporting evidence, (2) counter-evidence/risks, "
        "(3) ecosystem signals, (4) supply chain bottlenecks.\n"
        "Return ONLY a JSON array of 4 strings."
    )
    try:
        resp = settings.LLM_MINI.invoke(prompt)
        text = str(resp.content)
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


def _evaluate_web_evidence(evidence: List[Dict[str, Any]], state: AgentState) -> Dict[str, Any]:
    """W3: GPT-4o-mini evaluates web evidence quality, diversity, and bias."""
    summary = [
        {"title": e.get("title"), "company": e.get("company"), "source_type": e.get("source_type"), "excerpt": (e.get("excerpt") or "")[:100]}
        for e in evidence[:8]
    ]
    prompt = (
        "Evaluate this semiconductor competitive intelligence evidence set for quality and bias.\n"
        f"Target companies: {state.get('target_companies', [])}\n"
        f"Evidence ({len(evidence)} items, sample): {json.dumps(summary)}\n\n"
        "Score 0.0-1.0 based on: source type diversity, company coverage balance, absence of bias.\n"
        "Return ONLY valid JSON:\n"
        '{ "score": float, "issues": [list of short strings], "suggestions": [list of short strings] }'
    )
    try:
        resp = settings.LLM_MINI.invoke(prompt)
        text = str(resp.content)
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            result = json.loads(text[s:e])
            if "score" in result:
                return result
    except Exception as ex:
        print(f"[Web W3] LLM error: {ex}")
    return _rule_based_web_evidence_score(evidence, state)


def _rule_based_web_evidence_score(evidence: List[Dict[str, Any]], state: AgentState) -> Dict[str, Any]:
    source_diversity = min(len({e["source_type"] for e in evidence}) / 4.0, 1.0) if evidence else 0.0
    target_cos = state.get("target_companies", settings.TARGET_COMPANIES)
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


def _analyze_trl_with_llm(evidence: List[Dict[str, Any]], state: AgentState) -> Dict[str, Any]:
    """A1: GPT-4o analyzes TRL per company/technology based on evidence."""
    prompt = (
        "You are an expert semiconductor technology analyst.\n"
        "Given evidence snippets, estimate TRL (1-9) per technology per company and a short reason.\n"
        "Use only evidence given; if uncertain, lower TRL.\n"
        f"Question: {state.get('question', '')}\n"
        f"Target technologies: {state.get('target_technologies', [])}\n"
        f"Target companies: {state.get('target_companies', [])}\n"
        f"Evidence sample: {json.dumps(evidence[:12])}\n\n"
        "Return ONLY valid JSON of form:\n"
        '{ "trl_scores": { "HBM4": { "Samsung": {"trl": 0-9, "reason": str}, ... }, ... } }'
    )
    try:
        resp = settings.LLM_FULL.invoke(prompt)
        text = str(resp.content)
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception as ex:
        print(f"[Analysis A1] LLM error: {ex}")
    return {"trl_scores": {}}


def _generate_competitor_narrative(trl_scores: Dict[str, Any], state: AgentState) -> str:
    """A2: GPT-4o generates competitor narrative based on TRL matrix."""
    prompt = (
        "Write a concise competitor narrative based on this TRL matrix.\n"
        f"Question: {state.get('question', '')}\n"
        f"TRL matrix: {json.dumps(trl_scores)[:3000]}\n\n"
        "Output: 6-10 bullet points highlighting competitive positioning and implications for SK hynix."
    )
    try:
        resp = settings.LLM_FULL.invoke(prompt)
        return str(resp.content).strip()
    except Exception as ex:
        print(f"[Analysis A2] LLM error: {ex}")
        return "Narrative unavailable."


def _write_report_with_llm(state: AgentState, evidence: List[Dict[str, Any]], competitor_narrative: str) -> str:
    """D1: GPT-4o writes the final reference-style strategy report with citations."""
    prompt = (
        "You are writing a reference-report style technology strategy analysis for semiconductor competitive intelligence.\n"
        "You MUST include explicit citations to provided evidence in the format [doc_id].\n"
        "Structure:\n"
        "Cover block (Title, Date, Target companies/technologies)\n"
        "EXECUTIVE SUMMARY\n"
        "Key Metrics\n"
        "## 1. Problem framing\n"
        "## 2. Technology Landscape\n"
        "## 3. Competitor analysis\n"
        "### 3.1 TRL-Based Competitor Benchmarking\n"
        "### 3.2 Threat/Opportunity\n"
        "## 4. Recommendations\n"
        "REFERENCES (bulleted list, each with doc_id)\n"
        "Also include a limitations note.\n\n"
        f"Question: {state.get('question', '')}\n"
        f"Target technologies: {state.get('target_technologies', [])}\n"
        f"Target companies: {state.get('target_companies', [])}\n"
        f"Competitor narrative: {competitor_narrative}\n"
        f"Evidence (up to 18 items): {json.dumps(evidence[:18])}\n"
    )
    try:
        resp = settings.LLM_FULL.invoke(prompt)
        return str(resp.content).strip()
    except Exception as ex:
        print(f"[Draft D1] LLM error: {ex}")
        return "Draft unavailable due to LLM error."


def _translate_to_korean(draft: str) -> str:
    prompt = (
        "Translate the following technology strategy report to Korean while preserving structure, headings, and citations.\n"
        "Keep citations like [doc_id] unchanged.\n\n"
        f"{draft}"
    )
    try:
        resp = settings.LLM_FULL.invoke(prompt)
        return str(resp.content).strip()
    except Exception as ex:
        print(f"[Draft KO] LLM error: {ex}")
        return ""


def _normalize_baseline_labels(matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure the baseline row is labeled consistently for table generation."""
    out: List[Dict[str, Any]] = []
    for row in matrix:
        r = dict(row)
        if r.get("company") == "SK hynix":
            r["threat_level"] = "BASELINE"
        out.append(r)
    return out


def _matrix_table_reason(row: Dict[str, Any], korean: bool = False) -> str:
    reason = str(row.get("short_reason", ""))
    if not korean:
        return reason
    mapping = {
        "Closer to deployment than baseline": "기준선 대비 배포 단계에 더 근접",
        "Evidence shows active integration": "증거가 적극적 통합을 시사",
        "Research-only signals": "연구 단계 신호 위주",
    }
    return mapping.get(reason, reason)


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
    header = "### 3.1 TRL 기반 경쟁사 비교" if korean else "### 3.1 TRL-Based Competitor Benchmarking"
    if header not in draft or not matrix:
        return draft
    start = draft.find(header)
    next_candidates = [
        idx
        for idx in [
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


def rag_agent_node(state: AgentState) -> AgentState:
    target_techs = state.get("target_technologies", ["HBM4", "PIM", "CXL"])
    base_question = state["question"]

    queries = [base_question, f"TRL evidence for {base_question}", f"competitor signal for {base_question}"]
    if state.get("retry_reason") or state.get("improvement_instructions"):
        queries.append(f"{base_question} {state.get('retry_reason', '')} {state.get('improvement_instructions', '')}")

    target_cos = state.get("target_companies", settings.TARGET_COMPANIES)
    retrieved: List[Document] = []
    for tech in target_techs:
        tech_queries = [f"{tech} latest competitor updates", f"{tech} TRL maturity evidence", f"{tech} paper patent product signal"]
        for co in target_cos:
            for q in queries[:2] + tech_queries:
                retrieved.extend(runtime.hybrid_search(f"{q} {tech}", tech_hint=tech, company=co, top_k=3))
        for q in queries[:2]:
            retrieved.extend(runtime.hybrid_search(f"{q} {tech}", tech_hint=tech, top_k=3))

    retrieved.extend(runtime.literal_source_matches(target_techs, target_cos))
    vlm_docs = process_pdf_sources_with_vlm(settings.DATA_DIR)
    retrieved.extend(vlm_docs)

    seen: set[str] = set()
    dedup: List[Document] = []
    for d in retrieved:
        cid = str(d.metadata["chunk_id"])
        if cid not in seen:
            seen.add(cid)
            dedup.append(d)

    normalized = [normalize_doc(d, "rag") for d in dedup]

    covered_techs = {tech for tech in target_techs if any(runtime.evidence_covers_target(e, tech) for e in normalized)}
    missing_techs = [tech for tech in target_techs if tech not in covered_techs]
    coverage_score = len(covered_techs) / max(len(target_techs), 1)
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
            {"doc_id": d.metadata["doc_id"], "chunk_id": d.metadata["chunk_id"], "title": d.metadata["title"], "technology": d.metadata["technology"], "company": d.metadata["company"]}
            for d in dedup
        ],
        "normalized_evidence": normalized,
        "rag_judge_result": build_judge(score, get_threshold(state, "rag"), issues, suggestions),
    }


def _web_cache_path(company: str, query: str) -> Path:
    slug = settings.COMPANY_SLUG.get(company, company.lower().replace(" ", ""))
    d = settings.WEB_CACHE_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    qhash = hashlib.md5(query.encode()).hexdigest()[:12]
    return d / f"{qhash}.json"


def _load_web_cache(company: str, query: str) -> Optional[List[Dict[str, Any]]]:
    p = _web_cache_path(company, query)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_web_cache(company: str, query: str, results: List[Dict[str, Any]]) -> None:
    p = _web_cache_path(company, query)
    p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _web_cache_to_faiss(new_docs: List[Document]) -> None:
    if not new_docs:
        return

    try:
        if settings.WEB_CACHE_FAISS_PATH.exists():
            wc_vs = FAISS.load_local(
                str(settings.WEB_CACHE_FAISS_PATH),
                settings.EMBED_MODEL,
                allow_dangerous_deserialization=True,
            )
            wc_vs.add_documents(new_docs)
        else:
            wc_vs = FAISS.from_documents(new_docs, settings.EMBED_MODEL)
        settings.WEB_CACHE_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        wc_vs.save_local(str(settings.WEB_CACHE_FAISS_PATH))
    except Exception as ex:
        print(f"[Web cache] FAISS save failed: {ex}")

    try:
        runtime.initialize()
        if runtime.VECTORSTORE is not None:
            runtime.VECTORSTORE.add_documents(new_docs)
        for d in new_docs:
            co = d.metadata.get("company")
            if co and co in runtime.COMPANY_VECTORSTORES:
                runtime.COMPANY_VECTORSTORES[co].add_documents([d])
    except Exception as ex:
        print(f"[Web cache] in-memory merge failed: {ex}")


def _detect_company(text: str) -> Optional[str]:
    t = text.lower()
    if "samsung" in t:
        return "Samsung"
    if "sk hynix" in t or "skhynix" in t:
        return "SK hynix"
    if "intel" in t:
        return "Intel"
    if "nvidia" in t:
        return "NVIDIA"
    return None


def _is_relevant_web_result(result: Dict[str, Any], state: AgentState) -> bool:
    title = (result.get("title") or "").lower()
    content = (result.get("content") or "").lower()
    q = (state.get("question") or "").lower()
    if any(k in title or k in content for k in ["hbm", "pim", "cxl", "jedec", "memory"]):
        return True
    return any(w in title or w in content for w in q.split()[:4])


def _is_report_relevant_evidence(e: Dict[str, Any]) -> bool:
    src = str(e.get("source_url") or "")
    return src.startswith("http") or src.endswith(".pdf") or "arxiv" in src.lower()


def _tavily_search(query: str) -> List[Dict[str, Any]]:
    if settings.TAVILY is None:
        return []
    try:
        return list(settings.TAVILY.invoke({"query": query}) or [])
    except Exception as ex:
        print(f"[Tavily] error: {ex}")
        return []


def web_agent_node(state: AgentState) -> AgentState:
    """Web Agent (W1–W4): query gen, web search, classify, judge."""
    queries = _generate_web_queries(state)
    target_cos = state.get("target_companies", settings.TARGET_COMPANIES)
    evidence: List[Dict[str, Any]] = []
    new_docs: List[Document] = []

    for co in target_cos:
        for q in queries:
            cq = f"{co} {q}"
            cached = _load_web_cache(co, cq)
            if cached is None:
                results = _tavily_search(cq)
                if results:
                    _save_web_cache(co, cq, results)
                    cached = results
            if not cached:
                continue
            for r in cached[:4]:
                if not _is_relevant_web_result(r, state):
                    continue
                doc = Document(
                    page_content=(r.get("content") or r.get("snippet") or "")[:1200],
                    metadata={
                        "doc_id": f"web-{hashlib.md5((r.get('url','')+r.get('title','')).encode()).hexdigest()[:10]}",
                        "chunk_id": f"web-{hashlib.md5((r.get('url','')+r.get('title','')).encode()).hexdigest()[:10]}",
                        "title": r.get("title") or "",
                        "company": _detect_company((r.get("title") or "") + " " + (r.get("content") or "")) or co,
                        "technology": runtime.detect_tech_hint(cq, fallback="HBM4"),
                        "source_type": "web",
                        "published_at": datetime.now().strftime("%Y-%m-%d"),
                        "source_url": r.get("url") or "",
                    },
                )
                new_docs.append(doc)
                evidence.append(normalize_doc(doc, "web"))

    _web_cache_to_faiss(new_docs)

    judged = _evaluate_web_evidence(evidence, state)
    score = float(judged.get("score", 0.0))
    issues = list(judged.get("issues", []))
    suggestions = list(judged.get("suggestions", []))

    # W4: filter obvious junk
    evidence = [e for e in evidence if _is_report_relevant_evidence(e)]

    return {
        "web_queries": queries,
        "web_classified": evidence,
        "web_judge_result": build_judge(score, get_threshold(state, "web"), issues, suggestions),
    }


def analysis_agent_node(state: AgentState) -> AgentState:
    """Analysis Agent (A1–A2)"""
    evidence = list(state.get("normalized_evidence", [])) + list(state.get("web_classified", []))
    analysis = _analyze_trl_with_llm(evidence, state)
    trl_scores = analysis.get("trl_scores", {}) if isinstance(analysis, dict) else {}
    competitor_narrative = _generate_competitor_narrative(trl_scores, state)

    # Rule-based quality
    target_techs = state.get("target_technologies", ["HBM4", "PIM", "CXL"])
    target_cos = state.get("target_companies", settings.TARGET_COMPANIES)
    filled = 0
    for tech in target_techs:
        for co in target_cos:
            if str(trl_scores.get(tech, {}).get(co, {}).get("trl", "")).strip():
                filled += 1
    matrix_score = filled / max(len(target_techs) * len(target_cos), 1)
    score = 0.55 * matrix_score + 0.45
    issues: List[str] = []
    suggestions: List[str] = []
    if matrix_score < 0.7:
        issues.append("TRL matrix coverage is incomplete.")
        suggestions.append("Add more evidence or infer conservative TRL with clear reasons.")

    return {
        "trl_scores": trl_scores,
        "competitor_analysis": {"competitor_narrative": competitor_narrative},
        "analysis_judge_result": build_judge(score, get_threshold(state, "analysis"), issues, suggestions),
    }


def _estimate_trl_rule(evidence: List[Dict[str, Any]], tech: str, company: str) -> Tuple[int, str, str]:
    """Rule-based TRL estimate as fallback when LLM output is missing."""
    snippets = " ".join([(e.get("excerpt") or "") for e in evidence if e.get("technology") == tech and e.get("company") == company])[:2000]
    s = snippets.lower()
    if any(k in s for k in ["shipping", "mass production", "volume", "deployment"]):
        return 8, "HIGH", "product/deployment signal"
    if any(k in s for k in ["prototype", "demo", "pilot", "evaluation"]):
        return 6, "MEDIUM", "standard/prototype signal"
    if any(k in s for k in ["paper", "research", "arxiv", "isscc", "iedm"]):
        return 4, "LOW", "research signal"
    return 3, "LOW", "no direct evidence"


def draft_agent_node(state: AgentState) -> AgentState:
    """Draft Agent (D1–D2)"""
    evidence = list(state.get("normalized_evidence", [])) + list(state.get("web_classified", []))
    competitor_narrative = str(state.get("competitor_analysis", {}).get("competitor_narrative", ""))
    draft = _write_report_with_llm(state, evidence, competitor_narrative)
    draft_ko = _translate_to_korean(draft) if draft else ""

    # Build TRL benchmark matrix for table injection
    matrix: List[Dict[str, Any]] = []
    target_techs = state.get("target_technologies", ["HBM4", "PIM", "CXL"])
    target_cos = state.get("target_companies", settings.TARGET_COMPANIES)
    trl_scores = state.get("trl_scores", {}) or {}

    for tech in target_techs:
        for co in target_cos:
            entry = (trl_scores.get(tech, {}) or {}).get(co, {}) if isinstance(trl_scores, dict) else {}
            trl = entry.get("trl")
            reason = entry.get("reason", "")
            if trl is None or trl == "":
                trl, threat, criterion = _estimate_trl_rule(evidence, tech, co)
                short_reason = "Closer to deployment than baseline" if threat == "HIGH" else "Research-only signals"
            else:
                try:
                    trl = int(float(trl))
                except Exception:
                    trl = 3
                threat = "HIGH" if trl >= 7 else ("MEDIUM" if trl >= 5 else "LOW")
                criterion = "program integration signal" if trl >= 6 else "research signal"
                short_reason = str(reason)[:60] if reason else "Evidence shows active integration"
            matrix.append(
                {
                    "technology": tech,
                    "company": co,
                    "trl": trl,
                    "threat_level": threat,
                    "criterion": criterion,
                    "short_reason": short_reason,
                    "evidence_count": sum(1 for e in evidence if e.get("technology") == tech and e.get("company") == co),
                }
            )
    matrix = _normalize_baseline_labels(matrix)

    draft = _ensure_trl_table_section(draft, matrix, korean=False)
    if draft_ko:
        draft_ko = _ensure_trl_table_section(draft_ko, matrix, korean=True)

    # References map
    references: Dict[str, Dict[str, Any]] = {}
    for e in evidence:
        doc_id = str(e.get("doc_id") or "")
        if not doc_id:
            continue
        references[doc_id] = {
            "doc_id": doc_id,
            "title": e.get("title"),
            "source_url": e.get("source_url"),
            "published_at": e.get("published_at"),
            "company": e.get("company"),
            "technology": e.get("technology"),
            "source_type": e.get("source_type"),
        }

    d2 = check_draft_completeness(draft)
    section_score = float(d2.get("section_score", 0.8))
    citation_score_d = float(d2.get("citation_score", 0.8))
    d2_issues = list(d2.get("issues", []))
    d2_suggestions = list(d2.get("suggestions", []))

    # Simple rule scores
    rule_section_score = min((1.0 if "EXECUTIVE SUMMARY" in draft.upper() else 0.6) + 0.2, 1.0)
    competitor_score = 1.0 if "### 3.1" in draft else 0.6
    rule_citation_score = min(len(references) / 4.0, 1.0)
    rule_score = 0.45 * rule_section_score + 0.30 * competitor_score + 0.25 * rule_citation_score

    blended_score = 0.5 * ((section_score + citation_score_d) / 2) + 0.5 * rule_score
    issues = d2_issues[:]
    suggestions = d2_suggestions[:]
    if rule_citation_score < 1.0:
        issues.append("Citation count is below target >= 4.")
        suggestions.append("Increase evidence retrieval breadth before drafting.")

    return {
        "sections": {"draft_full": draft, "draft_full_ko": draft_ko, "competitor_narrative": competitor_narrative},
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
    return {"human_review_result": {"decision": decision, "feedback": feedback, "reviewed_at": datetime.now(timezone.utc).isoformat()}}


def pdf_agent_node(state: AgentState) -> AgentState:
    draft_en = str(state.get("draft", "")).strip()
    draft_ko = str(state.get("draft_ko", "")).strip()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_en_path = settings.OUTPUT_DIR / f"tech_strategy_report_{stamp}_en.md"
    pdf_en_path = settings.OUTPUT_DIR / f"tech_strategy_report_{stamp}_en.pdf"
    md_ko_path = settings.OUTPUT_DIR / f"tech_strategy_report_{stamp}_ko.md"
    pdf_ko_path = settings.OUTPUT_DIR / f"tech_strategy_report_{stamp}_ko.pdf"

    md_en_path.write_text(draft_en, encoding="utf-8")
    if draft_ko:
        md_ko_path.write_text(draft_ko, encoding="utf-8")

    validation = validate_draft_for_pdf(draft_en)
    p1_issues: List[str] = list(validation.get("issues", []))
    p1_suggestions: List[str] = list(validation.get("suggestions", []))

    pdf_en_ok, en_errs = render_pdf(inject_visual_markers(draft_en), pdf_en_path)
    p1_issues.extend(en_errs)
    if not pdf_en_ok:
        p1_suggestions.append("Install matplotlib and retry PDF generation.")
    else:
        table_qa = validate_pdf_tables_with_vlm(pdf_en_path)
        if not table_qa.get("ok", True):
            p1_issues.extend(table_qa.get("issues", []))
            p1_suggestions.extend(table_qa.get("suggestions", []))

    pdf_ko_ok = False
    if draft_ko:
        pdf_ko_ok, ko_errs = render_pdf(inject_visual_markers(draft_ko), pdf_ko_path)
        if ko_errs:
            print(f"[PDF P2-KO] 한국어 PDF 오류: {ko_errs}")

    print(f"[PDF] 영어  MD : {md_en_path.name}")
    print(f"[PDF] 영어  PDF: {pdf_en_path.name} ({'OK' if pdf_en_ok else 'FAIL'})")
    if draft_ko:
        print(f"[PDF] 한국어 MD : {md_ko_path.name}")
        print(f"[PDF] 한국어 PDF: {pdf_ko_path.name} ({'OK' if pdf_ko_ok else 'FAIL'})")

    p1_quality = float(validation.get("quality_score", 0.85))
    score = 0.4 * p1_quality + 0.6 * (1.0 if pdf_en_ok else 0.0)

    return {
        "markdown_path": str(md_en_path),
        "pdf_path": str(pdf_en_path) if pdf_en_ok else "",
        "markdown_ko_path": str(md_ko_path) if draft_ko else "",
        "pdf_ko_path": str(pdf_ko_path) if pdf_ko_ok else "",
        "pdf_judge_result": build_judge(score, get_threshold(state, "pdf"), p1_issues, p1_suggestions),
    }


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
        return {"status": status_if_retry, "revision_count": rev, "retry_reason": retry_reason, "improvement_instructions": improve}
    print(f"[Loop] {agent_key}: failed after {used}/{budget} retries - {retry_reason}")
    return {"status": "failed", "failure_reason": f"{agent_key} exceeded retry budget ({budget}).", "revision_count": rev}


def supervisor_node(state: AgentState) -> AgentState:
    if "rag_judge_result" not in state:
        print("[Supervisor] route -> rag_agent")
        return {"status": "run_rag", "retry_reason": "", "improvement_instructions": ""}
    if state["rag_judge_result"]["score"] < get_threshold(state, "rag"):
        improve = _supervisor_generate_instructions(state, "rag")
        return retry_or_fail(state, "rag", "run_rag", "RAG quality below threshold.", improve)

    if state["config"].get("enable_web_agent", True):
        if "web_judge_result" not in state:
            print("[Supervisor] route -> web_agent")
            return {"status": "run_web", "retry_reason": "", "improvement_instructions": ""}
        if state["web_judge_result"]["score"] < get_threshold(state, "web"):
            improve = _supervisor_generate_instructions(state, "web")
            return retry_or_fail(state, "web", "run_web", "Web evidence quality below threshold.", improve)

    if "analysis_judge_result" not in state:
        print("[Supervisor] route -> analysis_agent")
        return {"status": "run_analysis", "retry_reason": "", "improvement_instructions": ""}
    if state["analysis_judge_result"]["score"] < get_threshold(state, "analysis"):
        improve = _supervisor_generate_instructions(state, "analysis")
        return retry_or_fail(state, "analysis", "run_analysis", "Analysis quality below threshold.", improve)

    if "draft_judge_result" not in state:
        print("[Supervisor] route -> draft_agent")
        return {"status": "run_draft", "retry_reason": "", "improvement_instructions": ""}
    if state["draft_judge_result"]["score"] < get_threshold(state, "draft"):
        improve = _supervisor_generate_instructions(state, "draft")
        return retry_or_fail(state, "draft", "run_draft", "Draft quality below threshold.", improve)

    if "human_review_result" not in state:
        print("[Supervisor] route -> human_review")
        return {"status": "run_human_review"}
    if state["human_review_result"].get("decision") == "reject":
        feedback = str(state["human_review_result"].get("feedback", "Incorporate review comments."))
        return retry_or_fail(state, "draft", "run_draft", "Human reviewer rejected draft.", feedback)

    if "pdf_judge_result" not in state:
        print("[Supervisor] route -> pdf_agent")
        return {"status": "run_pdf", "retry_reason": "", "improvement_instructions": ""}
    if state["pdf_judge_result"]["score"] < get_threshold(state, "pdf"):
        improve = _supervisor_generate_instructions(state, "pdf")
        return retry_or_fail(state, "pdf", "run_pdf", "PDF generation below threshold.", improve)

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

