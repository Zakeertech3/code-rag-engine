from dataclasses import dataclass

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient

from src import config, embedder

_DENSE_CANDIDATE_COUNT: int = 20
_RERANK_URL: str = "https://api.jina.ai/v1/rerank"
_CLIENT: QdrantClient = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


@dataclass
class RetrievedChunk:
    file_path: str
    start_line: int
    end_line: int
    text: str
    score: float


def _rerank_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.JINA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _rrf_fuse(
    dense_ids: list[int],
    sparse_ids: list[int],
    top_k: int,
    rrf_k: int = 60,
) -> list[int]:
    scores: dict[int, float] = {}
    for rank, item_id in enumerate(dense_ids):
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, item_id in enumerate(sparse_ids):
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]


def _jina_rerank(query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
    response = requests.post(
        _RERANK_URL,
        headers=_rerank_headers(),
        json={
            "model": config.JINA_RERANK_MODEL,
            "query": query,
            "documents": texts,
            "top_n": top_n,
            "return_documents": False,
        },
        timeout=30,
    )
    if response.status_code == 401:
        raise RuntimeError(
            "Jina API authentication failed: JINA_API_KEY is invalid or expired"
        )
    if response.status_code == 429:
        raise RuntimeError("Jina API rate limit exceeded — wait and retry")
    response.raise_for_status()
    data: dict = response.json()
    return [(item["index"], float(item["relevance_score"])) for item in data["results"]]


def retrieve(query: str, top_n: int = 5) -> list[RetrievedChunk]:
    query_vector = embedder.embed_query(query)

    dense_response = _CLIENT.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        limit=_DENSE_CANDIDATE_COUNT,
        with_payload=True,
    )
    dense_ids: list[int] = [p.id for p in dense_response.points]

    all_points, _ = _CLIENT.scroll(
        collection_name=config.COLLECTION_NAME,
        limit=10000,
        with_payload=True,
    )

    if not all_points:
        return []

    all_texts: list[str] = [p.payload["text"] for p in all_points]
    all_ids: list[int] = [p.id for p in all_points]

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1)
    corpus_matrix = vectorizer.fit_transform(all_texts)
    try:
        query_vec = vectorizer.transform([query])
        bm25_scores: np.ndarray = (corpus_matrix @ query_vec.T).toarray().flatten()
    except Exception:
        bm25_scores = np.zeros(len(all_texts))

    bm25_order = np.argsort(bm25_scores)[::-1]
    sparse_ids: list[int] = [all_ids[i] for i in bm25_order[:_DENSE_CANDIDATE_COUNT]]

    fused_ids = _rrf_fuse(dense_ids, sparse_ids, top_k=_DENSE_CANDIDATE_COUNT)

    id_to_point = {p.id: p for p in all_points}
    candidates = [id_to_point[pid] for pid in fused_ids if pid in id_to_point]

    if not candidates:
        return []

    candidate_texts: list[str] = [c.payload["text"] for c in candidates]
    reranked: list[tuple[int, float]] = _jina_rerank(query, candidate_texts, top_n=top_n)

    return [
        RetrievedChunk(
            file_path=candidates[idx].payload["file_path"],
            start_line=candidates[idx].payload["start_line"],
            end_line=candidates[idx].payload["end_line"],
            text=candidates[idx].payload["text"],
            score=score,
        )
        for idx, score in reranked
        if idx < len(candidates)
    ]
