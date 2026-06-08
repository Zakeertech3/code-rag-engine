import re

from slack_bolt import App, Say
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.web.client import WebClient

from src import config
from src.generator import generate
from src.retriever import retrieve

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
_EMPTY_REPLY = "Please include a question after mentioning me."
_ERROR_REPLY = "Something went wrong while looking up the codebase. Please try again."

_app: App = App(token=config.SLACK_BOT_TOKEN)


@_app.event("app_mention")
def handle_mention(event: dict, client: WebClient, say: Say) -> None:
    channel: str = event["channel"]
    message_ts: str = event["ts"]
    thread_ts: str = event.get("thread_ts", message_ts)
    raw_text: str = event.get("text", "")

    question: str = _MENTION_RE.sub("", raw_text).strip()

    if not question:
        say(text=_EMPTY_REPLY, thread_ts=thread_ts)
        return

    try:
        client.reactions_add(channel=channel, name="eyes", timestamp=message_ts)
    except Exception:
        pass

    try:
        chunks = retrieve(question)
        answer = generate(question, chunks)
    except Exception:
        say(text=_ERROR_REPLY, thread_ts=thread_ts)
        return

    say(text=answer, thread_ts=thread_ts)


if __name__ == "__main__":
    handler = SocketModeHandler(_app, app_token=config.SLACK_APP_TOKEN)
    handler.start()
