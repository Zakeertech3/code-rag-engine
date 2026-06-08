import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import RetrievedChunk, retrieve

_DEFAULT_K: int = 5
_EVAL_SET_PATH: Path = Path(__file__).resolve().parent / "eval_set.json"


def _load_eval_set() -> list[dict]:
    with open(_EVAL_SET_PATH) as f:
        return json.load(f)


def _retrieved_basenames(chunks: list[RetrievedChunk]) -> list[str]:
    return [Path(c.file_path).name for c in chunks]


def _hit(retrieved: list[str], expected: list[str]) -> bool:
    return any(e in retrieved for e in expected)


def _recall(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 0.0
    matched = sum(1 for e in expected if e in retrieved)
    return matched / len(expected)


def _reciprocal_rank(chunks: list[RetrievedChunk], expected: list[str]) -> float:
    for rank, chunk in enumerate(chunks, start=1):
        if Path(chunk.file_path).name in expected:
            return 1.0 / rank
    return 0.0


def run_eval(k: int = _DEFAULT_K) -> None:
    entries = _load_eval_set()

    hits: list[bool] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    print(f"eval set: {len(entries)} questions, k={k}")
    print()

    for entry in entries:
        question: str = entry["question"]
        expected: list[str] = entry["expected_files"]

        chunks = retrieve(question, top_n=k)
        retrieved = _retrieved_basenames(chunks)

        is_hit = _hit(retrieved, expected)
        rec = _recall(retrieved, expected)
        rr = _reciprocal_rank(chunks, expected)

        hits.append(is_hit)
        recalls.append(rec)
        reciprocal_ranks.append(rr)

        hit_marker = "HIT " if is_hit else "MISS"
        print(f"[{hit_marker}] {question!r}")
        print(f"  expected:  {expected}")
        print(f"  retrieved: {retrieved}")
        print(f"  recall={rec:.2f}  rr={rr:.4f}")
        print()

    hit_rate = sum(hits) / len(hits) if hits else 0.0
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    separator = "-" * 60
    print(separator)
    print(f"hit rate at k={k}:  {hit_rate:.3f}  ({sum(hits)}/{len(hits)})")
    print(f"mean recall:        {mean_recall:.3f}")
    print(f"MRR:                {mrr:.3f}")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_K
    run_eval(k)
