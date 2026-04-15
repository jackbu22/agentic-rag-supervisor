from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_chunks(source_rows: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Document]:
    docs = [
        Document(
            page_content=row["content"],
            metadata={
                "doc_id": row["doc_id"],
                "title": row["title"],
                "technology": row["technology"],
                "company": row["company"],
                "source_type": row["source_type"],
                "published_at": row["published_at"],
                "source_url": row["source_url"],
            },
        )
        for row in source_rows
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"eval-chunk-{i:05d}"
    return chunks


def make_embedding(provider: str, model: str | None = None):
    provider = provider.lower()
    if provider == "bge":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model or "BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model or "text-embedding-3-small")
    if provider == "jina":
        try:
            from langchain_community.embeddings import JinaEmbeddings
        except Exception as exc:
            raise RuntimeError("JinaEmbeddings is unavailable. Install a compatible langchain-community version.") from exc
        if not os.environ.get("JINA_API_KEY"):
            raise RuntimeError("JINA_API_KEY is not set.")
        return JinaEmbeddings(jina_api_key=os.environ["JINA_API_KEY"], model_name=model or "jina-embeddings-v3")
    if provider == "voyage":
        try:
            from langchain_voyageai import VoyageAIEmbeddings
        except Exception as exc:
            raise RuntimeError("langchain-voyageai is not installed. Run: pip install langchain-voyageai voyageai") from exc
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY is not set.")
        return VoyageAIEmbeddings(model=model or "voyage-3-large")
    raise ValueError(f"Unknown embedding provider: {provider}")


def reciprocal_rank_fusion(dense: List[Document], sparse: List[Document], alpha: float) -> List[Document]:
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}
    for rank, doc in enumerate(dense, start=1):
        key = doc.metadata["chunk_id"]
        docs[key] = doc
        scores[key] = scores.get(key, 0.0) + alpha * (1.0 / (rank + 50))
    for rank, doc in enumerate(sparse, start=1):
        key = doc.metadata["chunk_id"]
        docs[key] = doc
        scores[key] = scores.get(key, 0.0) + (1.0 - alpha) * (1.0 / (rank + 50))
    return [docs[key] for key, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def evaluate_ranked(
    qa_rows: List[Dict[str, Any]],
    search_fn,
    ks: Iterable[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    max_k = max(ks)
    reciprocal_ranks: List[float] = []
    ndcg_values: List[float] = []
    hit_counts = {k: 0 for k in ks}
    recall_sums = {k: 0.0 for k in ks}

    for row in qa_rows:
        gt = set(row.get("ground_truth_doc_ids", []))
        if not gt:
            continue
        ranked = search_fn(row["question"], max_k)
        doc_ids = [doc.metadata["doc_id"] for doc in ranked]

        rr = 0.0
        dcg = 0.0
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id in gt:
                rr = rr or 1.0 / rank
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(gt), max_k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)) or 1.0
        reciprocal_ranks.append(rr)
        ndcg_values.append(dcg / idcg)

        for k in ks:
            top = set(doc_ids[:k])
            found = len(top.intersection(gt))
            if found:
                hit_counts[k] += 1
            recall_sums[k] += found / len(gt)

    n = max(len(qa_rows), 1)
    for k in ks:
        metrics[f"hit_rate@{k}"] = hit_counts[k] / n
        metrics[f"recall@{k}"] = recall_sums[k] / n
    metrics["mrr"] = sum(reciprocal_ranks) / n
    metrics[f"ndcg@{max_k}"] = sum(ndcg_values) / n
    return {key: round(value, 4) for key, value in metrics.items()}


def evaluate_provider(
    chunks: List[Document],
    qa_rows: List[Dict[str, Any]],
    provider: str,
    model: str | None,
    ks: List[int],
    alpha: float,
) -> Dict[str, Any]:
    emb = make_embedding(provider, model)
    dense_store = FAISS.from_documents(chunks, emb)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = max(ks) * 2

    def dense_search(query: str, k: int) -> List[Document]:
        return dense_store.similarity_search(query, k=k)

    def sparse_search(query: str, k: int) -> List[Document]:
        return bm25.invoke(query)[:k]

    def hybrid_search(query: str, k: int) -> List[Document]:
        dense = dense_store.similarity_search(query, k=max(k * 2, 10))
        sparse = bm25.invoke(query)[: max(k * 2, 10)]
        return reciprocal_rank_fusion(dense, sparse, alpha=alpha)[:k]

    return {
        "provider": provider,
        "model": model or "default",
        "dense": evaluate_ranked(qa_rows, dense_search, ks),
        "bm25": evaluate_ranked(qa_rows, sparse_search, ks),
        "hybrid": evaluate_ranked(qa_rows, hybrid_search, ks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embedding and retrieval choices on demo QA ground truth.")
    parser.add_argument("--sources", default="data/demo_semiconductor_sources.json")
    parser.add_argument("--qa", default="data/demo_qa_set.json")
    parser.add_argument("--providers", default="bge,openai,jina,voyage")
    parser.add_argument("--chunk-size", type=int, default=420)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--alpha", type=float, default=0.5, help="RRF dense weight for hybrid retrieval.")
    parser.add_argument("--output", default="outputs/retrieval_eval_results.json")
    args = parser.parse_args()

    sources = load_json(ROOT / args.sources)
    qa_rows = load_json(ROOT / args.qa)
    chunks = build_chunks(sources, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    ks = [int(k.strip()) for k in args.k.split(",") if k.strip()]

    results: List[Dict[str, Any]] = []
    for provider_spec in [p.strip() for p in args.providers.split(",") if p.strip()]:
        provider, _, model = provider_spec.partition(":")
        try:
            result = evaluate_provider(chunks, qa_rows, provider=provider, model=model or None, ks=ks, alpha=args.alpha)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            skipped = {"provider": provider, "model": model or "default", "skipped": str(exc)}
            results.append(skipped)
            print(json.dumps(skipped, ensure_ascii=False, indent=2))

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
