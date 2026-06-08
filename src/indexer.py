from dataclasses import dataclass

from src import embedder, store
from src.chunker import chunk_files
from src.github_client import fetch_python_files


@dataclass
class IndexSummary:
    files_fetched: int
    chunks_produced: int
    point_count: int


def run_index() -> IndexSummary:
    print("fetching Python files from GitHub...")
    files = fetch_python_files()
    print(f"fetched {len(files)} file(s): {[f.path for f in files]}")

    print("chunking files...")
    chunks = chunk_files(files)
    print(f"produced {len(chunks)} chunk(s)")

    print("setting up Qdrant collection...")
    store.setup_collection(embedder.embedding_dimension)

    if not chunks:
        count = store.point_count()
        print("no chunks to embed — collection is empty")
        return IndexSummary(files_fetched=0, chunks_produced=0, point_count=count)

    print("embedding chunks...")
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed_chunks(texts)
    print(f"embedded {len(vectors)} vector(s)")

    print("upserting to Qdrant...")
    store.upsert_chunks(chunks, vectors)
    count = store.point_count()
    print(f"done — collection now has {count} point(s)")

    return IndexSummary(
        files_fetched=len(files),
        chunks_produced=len(chunks),
        point_count=count,
    )
