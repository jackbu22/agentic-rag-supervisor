from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from agentic_rag_supervisor.demo import runtime, settings
from agentic_rag_supervisor.demo.agents import (
    analysis_agent_node,
    draft_agent_node,
    human_review_node,
    pdf_agent_node,
    rag_agent_node,
    route_from_supervisor,
    supervisor_node,
    web_agent_node,
)
from agentic_rag_supervisor.demo.types import AgentState


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("web_agent", web_agent_node)
    builder.add_node("analysis_agent", analysis_agent_node)
    builder.add_node("draft_agent", draft_agent_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("pdf_agent", pdf_agent_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "rag_agent": "rag_agent",
            "web_agent": "web_agent",
            "analysis_agent": "analysis_agent",
            "draft_agent": "draft_agent",
            "human_review": "human_review",
            "pdf_agent": "pdf_agent",
            "end": END,
        },
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("web_agent", "supervisor")
    builder.add_edge("analysis_agent", "supervisor")
    builder.add_edge("draft_agent", "supervisor")
    builder.add_edge("human_review", "supervisor")
    builder.add_edge("pdf_agent", "supervisor")
    return builder.compile()


def run_demo(
    human_decision: str = "approve",
    question: Optional[str] = None,
    target_technologies: Optional[List[str]] = None,
    target_companies: Optional[List[str]] = None,
    human_feedback: str = "Please ensure every claim has an explicit citation.",
) -> Dict[str, Any]:
    runtime.initialize()
    target_technologies = target_technologies or ["HBM4", "PIM", "CXL"]
    target_companies = target_companies or settings.TARGET_COMPANIES
    co_str = ", ".join(target_companies)
    if question is None:
        question = (
            f"Create a technology strategy analysis report for {', '.join(target_technologies)} "
            f"with competitor TRL comparison for {co_str}. "
            "Analyze competitive positioning and strategic implications for SK hynix."
        )
    graph = build_graph()
    initial_state: AgentState = {
        "question": question,
        "target_technologies": target_technologies,
        "target_companies": target_companies,
        "revision_count": {},
        "config": {
            "thresholds": settings.DEFAULT_THRESHOLDS,
            "max_retries": settings.DEFAULT_MAX_RETRIES,
            "enable_web_agent": True,
            "human_decision": human_decision,
            "human_feedback": human_feedback,
        },
    }
    return graph.invoke(initial_state, config={"recursion_limit": 60})

