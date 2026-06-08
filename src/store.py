from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src import config
from src.chunker import Chunk

_CLIENT: QdrantClient = QdrantClient(
    host=config.QDRANT_HOST,
    port=config.QDRANT_PORT,
)


def setup_collection(vector_size: int) -> None:
    if _CLIENT.collection_exists(config.COLLECTION_NAME):
        _CLIENT.delete_collection(config.COLLECTION_NAME)
    _CLIENT.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=qm.VectorParams(
            size=vector_size,
            distance=qm.Distance.COSINE,
        ),
    )


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    points: list[qm.PointStruct] = [
        qm.PointStruct(
            id=idx,
            vector=vector,
            payload={
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
            },
        )
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    _CLIENT.upsert(
        collection_name=config.COLLECTION_NAME,
        points=points,
    )


def point_count() -> int:
    result: qm.CountResult = _CLIENT.count(
        collection_name=config.COLLECTION_NAME,
        exact=True,
    )
    return result.count
