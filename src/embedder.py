import requests

from src import config

_EMBED_URL: str = "https://api.jina.ai/v1/embeddings"
_BATCH_SIZE: int = 64


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.JINA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _embed(texts: list[str], task: str) -> list[list[float]]:
    results: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = requests.post(
            _EMBED_URL,
            headers=_headers(),
            json={
                "model": config.JINA_EMBEDDING_MODEL,
                "task": task,
                "input": batch,
                "normalized": True,
                "embedding_type": "float",
            },
            timeout=60,
        )
        if response.status_code == 401:
            raise RuntimeError(
                "Jina API authentication failed: JINA_API_KEY is invalid or expired"
            )
        if response.status_code == 429:
            raise RuntimeError("Jina API rate limit exceeded — wait and retry")
        response.raise_for_status()
        data: dict = response.json()
        for item in sorted(data["data"], key=lambda x: x["index"]):
            results.append(item["embedding"])
    return results


def _probe_dimension() -> int:
    vectors = _embed(["probe"], "nl2code.passage")
    return len(vectors[0])


embedding_dimension: int = _probe_dimension()


def embed_chunks(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "nl2code.passage")


def embed_query(text: str) -> list[float]:
    return _embed([text], "nl2code.query")[0]
