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
- Retriever/Embedding 후보 평가: Dense-only, BM25-only, Hybrid-RRF 및 bge/OpenAI/Jina/Voyage 후보를 동일 QA 셋으로 비교
- 확증 편향 방지(실무 지향)
  - Web evidence set에 대해 **소스 다양성/커버리지/편향**을 점수화(W3)하고, 부족 시 재시도
  - Supervisor가 각 단계 실패 원인을 바탕으로 **구체적 개선 지시**를 생성(SR)
  - Human Review 단계로 “승인/반려” 흐름을 포함

## Tech Stack

| Category   | Details |
|------------|---------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | GPT-4o-mini / GPT-4o (OpenAI API) |
| Retrieval  | FAISS Dense, BM25 Sparse, Hybrid RRF |
| Retrieval Evaluation | Hit Rate@K, Recall@K, MRR, NDCG@K |
| Embedding  | bge-m3 (기본) / OpenAI `text-embedding-3-small` (fallback) / Jina / Voyage AI 후보 평가 |


## Agents
- RAG Agent: 오프라인 근거 검색 및 정규화(필요 시 PDF VLM 보강)
- Web Agent: 웹 질의 생성/수집/분류 및 근거 품질 평가
- Analysis Agent: 근거 기반 TRL 추정 및 경쟁 내러티브 생성
- Draft Agent: 근거 인용(`[doc_id]`)을 포함한 리포트 초안 작성 및 완성도 점검
- PDF Agent: Markdown/PDF 산출 및 시각 QA(테이블 렌더링) 점검
- Supervisor: Judge 점수 기반 라우팅 및 재시도 예산 관리

### TRL Evaluation Design
현재 구현에서는 TRL 평가가 `Analysis Agent` 내부의 A1 단계로 포함되어 있습니다.
즉, RAG/Web Agent가 수집한 근거를 입력으로 받아 `company × technology` TRL matrix를 생성하고,
Draft Agent가 이를 표와 세부 판단 근거로 보고서에 반영합니다.

설계 확장 관점에서는 TRL 평가를 별도 node로 분리할 수 있습니다.

`RAG Agent → Web Agent → TRL Evaluation Agent → Competitive Analysis Agent → Draft Agent → PDF Agent`

별도 `TRL Evaluation Agent`로 분리할 경우 다음 지표로 자체 Judge를 둘 수 있습니다.

| Metric | Meaning |
|--------|---------|
| Evidence Coverage | TRL 산정에 필요한 기술/기업별 근거 확보율 |
| Matrix Coverage | `target_technologies × target_companies` 조합 중 TRL이 채워진 비율 |
| Source Quality | paper, standard, product announcement, press release 등 근거 출처 품질 |
| Rationale Completeness | TRL, 위협 수준, 선정 기준, 불확실성이 설명되었는지 |

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
├── tools/
│   └── evaluate_retrieval.py    # Retriever/Embedding 후보 지표 평가
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

### 5) Retriever / Embedding 평가
리뷰 피드백 대응을 위해 `tools/evaluate_retrieval.py`로 보유 QA 셋 기준 검색 성능을 비교합니다.
이 스크립트는 동일한 `demo_qa_set.json` ground truth를 사용해 Dense-only, BM25-only, Hybrid-RRF를 비교합니다.

기본 bge-m3 평가:
```bash
python tools/evaluate_retrieval.py \
  --providers bge \
  --k 1,3,5 \
  --output outputs/retrieval_eval_bge.json
```

Jina/Voyage 후보까지 비교:
```bash
export JINA_API_KEY="..."
export VOYAGE_API_KEY="..."
pip install langchain-voyageai voyageai

python tools/evaluate_retrieval.py \
  --providers bge,openai,jina,voyage \
  --k 1,3,5,10 \
  --output outputs/retrieval_eval_all.json
```

평가 지표:

| Metric | Meaning |
|--------|---------|
| Hit Rate@K | top-k 안에 정답 문서가 하나라도 포함되는 비율 |
| Recall@K | 정답 문서 중 top-k에서 회수한 비율 |
| MRR | 첫 번째 정답 문서가 얼마나 높은 순위에 있는지 |
| NDCG@K | 정답 문서가 상위 랭킹에 잘 배치되었는지 |

현재 bge-m3 기준 샘플 결과에서는 Hybrid-RRF가 Recall@5와 NDCG@5에서 가장 높아,
보고서 생성용 기본 Retriever로 유지하는 것이 타당합니다.

```text
Dense  hit_rate@5=0.8, recall@5=0.7, mrr=0.6667, ndcg@5=0.6613
BM25   hit_rate@5=0.8, recall@5=0.7, mrr=0.7000, ndcg@5=0.6528
Hybrid hit_rate@5=0.8, recall@5=0.8, mrr=0.6667, ndcg@5=0.6754
```

따라서 현재 데이터에서는 다음처럼 정리할 수 있습니다.

- 보고서 생성용 검색: Hybrid-RRF 기본 유지
- 정확한 키워드 근거 확인: BM25 보조 활용
- 도메인 임베딩 최종 선정: bge-m3, Jina Embeddings v3, Voyage AI `voyage-3-large`를 동일 지표로 추가 비교

## Contributors
- 김건우: Prompt Engineering, PDF Parsing, RAG Agent, 엔드포인트 설계
- 배은서: Agent 아키텍쳐 Design, RAG Agent
