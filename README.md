# Agentic RAG Supervisor

반도체 기술/시장 리서치를 위해 **PDF/문서 기반 RAG + 멀티 에이전트 + 품질 심사(Judge) + 재시도 루프**를 결합한 데모 프로젝트입니다.

## Abstract

## Overview
- Objective: PDF/오프라인 자료 및 웹 근거를 결합해, 경쟁사(예: Samsung/SK hynix/Intel/NVIDIA) 기준 **기술 성숙도(TRL) 비교 및 전략 리포트**를 자동 생성합니다.
- Method: LangGraph 워크플로우로 `RAG → Web → Analysis → Draft → Human Review → PDF` 파이프라인을 구성하고, 단계별 Judge 점수(Threshold)로 **재시도/중단**을 제어합니다.
- Tools: OpenAI API, Tavily Web Search, FAISS, BM25, PyMuPDF, matplotlib

## Features
- PDF/오프라인 자료 기반 정보 추출 및 요약(수집 스크립트 포함)
- 하이브리드 검색: Dense(FAISS) + Sparse(BM25) RRF 결합
- 리포트 자동 생성: 참고 보고서 스타일 Markdown + PDF 렌더링(영/한)
- 평가 지표 출력: Hit Rate@K, MRR(데모 QA 셋 기반)
- 확증 편향 방지(실무 지향)
  - Web evidence set에 대해 **소스 다양성/커버리지/편향**을 점수화(W3)하고, 부족 시 재시도
  - Supervisor가 각 단계 실패 원인을 바탕으로 **구체적 개선 지시**를 생성(SR)
  - Human Review 단계로 “승인/반려” 흐름을 포함

## Tech Stack

| Category   | Details |
|------------|---------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | GPT-4o-mini / GPT-4o (OpenAI API) |
| Retrieval  | FAISS + BM25 (RRF), Hit Rate@K, MRR |
| Embedding  | bge-m3 (기본) / OpenAI `text-embedding-3-small` (fallback) |


## Agents
- RAG Agent: 오프라인 근거 검색 및 정규화(필요 시 PDF VLM 보강)
- Web Agent: 웹 질의 생성/수집/분류 및 근거 품질 평가
- Analysis Agent: 근거 기반 TRL 추정 및 경쟁 내러티브 생성
- Draft Agent: 근거 인용(`[doc_id]`)을 포함한 리포트 초안 작성 및 완성도 점검
- PDF Agent: Markdown/PDF 산출 및 시각 QA(테이블 렌더링) 점검
- Supervisor: Judge 점수 기반 라우팅 및 재시도 예산 관리

## Architecture
LangGraph로 아래 노드 흐름을 구성합니다.

`START → Supervisor → RAG → Supervisor → Web → Supervisor → Analysis → Supervisor → Draft → Supervisor → Human Review → Supervisor → PDF → Supervisor → END`

![Agent pipeline](./Agent%20pipeline.png)

## Directory Structure
현재 레포 구조(핵심만):
```
.
├── agentic_rag_supervisor/
│   ├── demo/                    # 데모 파이프라인(그래프/에이전트/설정/CLI)
│   └── ingest/                  # 수집/인덱싱(오프라인 문서 → sources.json/FAISS)
├── data/                        # 소스/QA 데이터
│   └── raw_pdfs/                # 회사별 원문 자료(하위 폴더: samsung, skhynix, intel, nvidia)
├── agentic_rag_supervisor_demo.py  # (호환용) 데모 실행 wrapper
├── ingest_papers.py             # (호환용) 수집 실행 wrapper
└── requirements.txt
```

실행 중 생성될 수 있는 디렉토리:
```
outputs/     # 리포트 산출물(MD/PDF)
faiss_db/    # FAISS 인덱스(merged/per-company/web_cache)
data/web_cache/  # 웹 검색 캐시(JSON)
```

## Getting Started

### 1) 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 환경변수
```bash
export OPENAI_API_KEY="..."
# 선택: 웹 에이전트 활성화(검색 품질 향상)
export TAVILY_API_KEY="..."
```

### 3) 자료 수집/인덱싱
회사별 폴더에 PDF/MD/TXT를 넣고 실행합니다.
```bash
python ingest_papers.py
# 강제 재처리
python ingest_papers.py --force
# 상태 확인
python ingest_papers.py --status
```

### 4) 데모 실행
```bash
python agentic_rag_supervisor_demo.py -q "SK hynix 관점에서 HBM4/PIM/CXL 경쟁사 전략 리포트를 작성해줘"
```

## Contributors
- 김건우: Prompt Engineering, PDF Parsing, RAG Agent, 엔드포인트 설계
- 배은서: Agent 아키텍쳐 Design, RAG Agent
