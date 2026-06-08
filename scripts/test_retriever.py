import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import retrieve
from src.store import point_count

count = point_count()
print(f"collection point count: {count}")
assert count == 16, f"expected 16 points, got {count}"
print("check 1 passed: point_count == 16")

print()
query1 = "where are expenses stored"
print(f"query: {query1!r}")
results1 = retrieve(query1, top_n=3)
for i, r in enumerate(results1):
    print(f"  [{i+1}] {r.file_path}:{r.start_line}-{r.end_line}  score={r.score:.4f}")
top_files1 = [r.file_path for r in results1]
assert any("storage" in f for f in top_files1), f"expected storage.py in top 3, got {top_files1}"
print("check 2 passed: storage.py ranked in top 3 for 'where are expenses stored'")

print()
query2 = "how is an expense categorized"
print(f"query: {query2!r}")
results2 = retrieve(query2, top_n=3)
for i, r in enumerate(results2):
    print(f"  [{i+1}] {r.file_path}:{r.start_line}-{r.end_line}  score={r.score:.4f}")
top_files2 = [r.file_path for r in results2]
assert any("categorizer" in f for f in top_files2), f"expected categorizer.py in top 3, got {top_files2}"
print("check 3 passed: categorizer.py ranked in top 3 for 'how is an expense categorized'")

print()
print("all checks passed")
