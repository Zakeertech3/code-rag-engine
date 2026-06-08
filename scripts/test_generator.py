import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import generate
from src.retriever import retrieve

_SEPARATOR = "-" * 60


def run(question: str) -> str:
    print(f"question: {question!r}")
    chunks = retrieve(question, top_n=5)
    print(f"chunks supplied ({len(chunks)}):")
    for c in chunks:
        print(f"  {c.file_path} lines {c.start_line}-{c.end_line}")
    answer = generate(question, chunks)
    print(f"\nanswer:\n{answer}")
    return answer


print(_SEPARATOR)
print("check 1: in-codebase question")
print(_SEPARATOR)
answer1 = run("how is an expense categorized")

print()
print(_SEPARATOR)
print("check 2: out-of-codebase question")
print(_SEPARATOR)
answer2 = run("how does user authentication work")

print()
print(_SEPARATOR)
print("check 3: verify answer2 does not invent an answer")
cannot_phrases = ("cannot find", "can't find", "not find", "no information", "not contain", "unable to find")
normalized2 = answer2.lower().replace("’", "'").replace("‘", "'")
assert any(p in normalized2 for p in cannot_phrases), (
    f"expected a cannot-find response for the auth question, got:\n{answer2}"
)
print("check 3 passed: out-of-codebase question produced a cannot-find response")
