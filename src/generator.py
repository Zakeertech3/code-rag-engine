from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    NotFoundError,
    RateLimitError,
)

from src import config
from src.retriever import RetrievedChunk

_CLIENT: Groq = Groq(api_key=config.GROQ_API_KEY)

_SYSTEM_PROMPT: str = (
    "You are a code assistant that answers questions about a codebase. "
    "You are given code chunks retrieved from the codebase. "
    "Answer using only the provided code. "
    "For every claim you make, cite the exact file and line range shown in the chunk label. "
    "If the provided chunks do not contain enough information to answer the question, "
    "say that you cannot find the answer in the provided code — do not invent or guess."
)

_NOT_FOUND: str = (
    "I could not find any relevant code in the indexed codebase to answer this question."
)


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    sections: list[str] = []
    for chunk in chunks:
        label = f"--- {chunk.file_path} lines {chunk.start_line}-{chunk.end_line} ---"
        sections.append(f"{label}\n{chunk.text}")
    joined = "\n\n".join(sections)
    return f"Code chunks:\n\n{joined}\n\nQuestion: {question}"


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return _NOT_FOUND

    user_message = _build_user_message(question, chunks)

    try:
        response = _CLIENT.chat.completions.create(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=config.GROQ_MODEL,
        )
    except AuthenticationError:
        raise RuntimeError(
            "Groq authentication failed: GROQ_API_KEY is invalid or expired"
        )
    except RateLimitError:
        raise RuntimeError("Groq rate limit exceeded — wait and retry")
    except NotFoundError:
        raise RuntimeError(
            f"Groq model not found: {config.GROQ_MODEL!r} — check GROQ_MODEL env var"
        )
    except APIConnectionError:
        raise RuntimeError("Groq API connection failed: check network")
    except APITimeoutError:
        raise RuntimeError("Groq API request timed out")

    return response.choices[0].message.content
