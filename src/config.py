import os
from dotenv import load_dotenv

load_dotenv()

_REQUIRED: tuple[str, ...] = (
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "GROQ_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_REPO",
    "GITHUB_WEBHOOK_SECRET",
    "JINA_API_KEY",
)

_missing: list[str] = [name for name in _REQUIRED if not os.getenv(name)]
if _missing:
    raise RuntimeError(
        "Missing required environment variables: " + ", ".join(_missing)
    )

SLACK_BOT_TOKEN: str = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN: str = os.environ["SLACK_APP_TOKEN"]
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GITHUB_TOKEN: str = os.environ["GITHUB_TOKEN"]
GITHUB_REPO: str = os.environ["GITHUB_REPO"]
GITHUB_WEBHOOK_SECRET: str = os.environ["GITHUB_WEBHOOK_SECRET"]
JINA_API_KEY: str = os.environ["JINA_API_KEY"]

GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code")
JINA_EMBEDDING_MODEL: str = os.getenv("JINA_EMBEDDING_MODEL", "jina-code-embeddings-0.5b")
JINA_RERANK_MODEL: str = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "code_chunks")
