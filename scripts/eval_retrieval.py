from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def rrf(rankings: list[list[str]], k0: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k0 + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


@dataclass(frozen=True)
class Corpus:
    doc_ids: list[str]
    texts: list[str]


def load_corpus(sources_path: Path) -> Corpus:
    rows = json.loads(sources_path.read_text(encoding="utf-8-sig"))
    doc_ids: list[str] = []
    texts: list[str] = []
    for r in rows:
        doc_ids.append(str(r["doc_id"]))
        texts.append((str(r.get("title", "")) + "\n" + str(r.get("content", ""))).strip())
    return Corpus(doc_ids=doc_ids, texts=texts)


def load_qa(qa_path: Path) -> list[dict]:
    return json.loads(qa_path.read_text(encoding="utf-8-sig"))


def make_bm25(corpus: Corpus) -> Callable[[str, int], list[str]]:
    index = BM25Okapi([tokenize(t) for t in corpus.texts])

    def search(query: str, top_k: int) -> list[str]:
        scores = index.get_scores(tokenize(query))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [corpus.doc_ids[i] for i in idx]

    return search


def make_tfidf_word(corpus: Corpus) -> Callable[[str, int], list[str]]:
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(corpus.texts)

    def search(query: str, top_k: int) -> list[str]:
        qv = vectorizer.transform([query])
        sims = (X @ qv.T).toarray().ravel()
        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [corpus.doc_ids[i] for i in idx]

    return search


def make_tfidf_char(corpus: Corpus) -> Callable[[str, int], list[str]]:
    vectorizer = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vectorizer.fit_transform(corpus.texts)

    def search(query: str, top_k: int) -> list[str]:
        qv = vectorizer.transform([query])
        sims = (X @ qv.T).toarray().ravel()
        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [corpus.doc_ids[i] for i in idx]

    return search


def make_rrf(a: Callable[[str, int], list[str]], b: Callable[[str, int], list[str]]) -> Callable[[str, int], list[str]]:
    def search(query: str, top_k: int) -> list[str]:
        merged = rrf([a(query, max(top_k, 20)), b(query, max(top_k, 20))])
        return merged[:top_k]

    return search


def hit_rate_at_k(results: list[str], ground_truth: set[str]) -> float:
    return 1.0 if any(doc_id in ground_truth for doc_id in results) else 0.0


def reciprocal_rank(results: list[str], ground_truth: set[str]) -> float:
    for i, doc_id in enumerate(results, start=1):
        if doc_id in ground_truth:
            return 1.0 / i
    return 0.0


def evaluate(search: Callable[[str, int], list[str]], qa_rows: Iterable[dict], k: int) -> tuple[float, float]:
    hits = 0.0
    mrr = 0.0
    n = 0
    for row in qa_rows:
        n += 1
        gt = set(row.get("ground_truth_doc_ids", []))
        query = str(row.get("question", ""))
        res = search(query, k)
        hits += hit_rate_at_k(res, gt)
        mrr += reciprocal_rank(res, gt)
    if n == 0:
        return 0.0, 0.0
    return hits / n, mrr / n


def choose_best(scores: dict[str, dict[str, float]]) -> str:
    # Prefer higher MRR@5, then Hit@5, then MRR@1. Tie-break to simpler methods.
    order = ["bm25", "tfidf_word", "tfidf_char", "rrf_bm25_tfidf_word", "rrf_bm25_tfidf_char"]

    def key(name: str):
        s = scores[name]
        return (s["mrr@5"], s["hit@5"], s["mrr@1"], -order.index(name) if name in order else -999)

    return max(scores.keys(), key=key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrievers on the bundled QA set.")
    parser.add_argument("--sources", default="data/demo_semiconductor_sources.json")
    parser.add_argument("--qa", default="data/demo_qa_set.json")
    args = parser.parse_args()

    corpus = load_corpus(Path(args.sources))
    qa_rows = load_qa(Path(args.qa))

    bm25 = make_bm25(corpus)
    tfidf_word = make_tfidf_word(corpus)
    tfidf_char = make_tfidf_char(corpus)

    methods: dict[str, Callable[[str, int], list[str]]] = {
        "bm25": bm25,
        "tfidf_word": tfidf_word,
        "tfidf_char": tfidf_char,
        "rrf_bm25_tfidf_word": make_rrf(bm25, tfidf_word),
        "rrf_bm25_tfidf_char": make_rrf(bm25, tfidf_char),
    }

    scores: dict[str, dict[str, float]] = {}
    for name, fn in methods.items():
        hit1, mrr1 = evaluate(fn, qa_rows, k=1)
        hit3, mrr3 = evaluate(fn, qa_rows, k=3)
        hit5, mrr5 = evaluate(fn, qa_rows, k=5)
        scores[name] = {
            "hit@1": hit1,
            "mrr@1": mrr1,
            "hit@3": hit3,
            "mrr@3": mrr3,
            "hit@5": hit5,
            "mrr@5": mrr5,
        }

    best = choose_best(scores)

    print(f"Corpus: {len(corpus.doc_ids)} docs | QA: {len(qa_rows)} questions\n")
    print("name\tHit@1\tMRR@1\tHit@3\tMRR@3\tHit@5\tMRR@5")
    for name, s in sorted(scores.items(), key=lambda x: (x[1]["mrr@5"], x[1]["hit@5"]), reverse=True):
        print(
            f"{name}\t"
            f"{s['hit@1']:.3f}\t{s['mrr@1']:.3f}\t{s['hit@3']:.3f}\t{s['mrr@3']:.3f}\t{s['hit@5']:.3f}\t{s['mrr@5']:.3f}"
        )

    print(f"\nBest (by MRR@5/Hit@5): {best}")
    print(f"Recommendation: set `ARS_RETRIEVER_MODE={best}` to use this mode in the demo runtime.\n")
    if len(qa_rows) < 20:
        print("Note: QA set is very small; many methods may tie. Add more QA items to differentiate retrievers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
