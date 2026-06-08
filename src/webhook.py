import hashlib
import hmac
import logging
import os

from flask import Flask, Response, request

from src import config
from src.indexer import run_index

_logger = logging.getLogger(__name__)
_WEBHOOK_PORT: int = int(os.getenv("WEBHOOK_PORT", "3000"))

_app: Flask = Flask(__name__)


def _verify_signature(body: bytes, signature_header: str | None) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    received_digest = signature_header[len("sha256="):]
    computed_digest = hmac.new(
        config.GITHUB_WEBHOOK_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(computed_digest, received_digest)


@_app.post("/webhook")
def handle_webhook() -> Response:
    body: bytes = request.get_data()
    signature_header: str | None = request.headers.get("X-Hub-Signature-256")

    if not _verify_signature(body, signature_header):
        return Response("Unauthorized", status=401)

    event: str = request.headers.get("X-GitHub-Event", "")

    if event == "push":
        try:
            summary = run_index()
            _logger.info(
                "re-indexed: files=%d chunks=%d points=%d",
                summary.files_fetched,
                summary.chunks_produced,
                summary.point_count,
            )
        except Exception as exc:
            _logger.error("indexing failed: %s", exc)
            return Response("Indexing error", status=500)

    return Response("OK", status=200)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _app.run(host="0.0.0.0", port=_WEBHOOK_PORT)
