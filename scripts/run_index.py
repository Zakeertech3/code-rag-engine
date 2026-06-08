import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexer import run_index

if __name__ == "__main__":
    summary = run_index()
    print()
    print("index complete")
    print(f"  files fetched:   {summary.files_fetched}")
    print(f"  chunks produced: {summary.chunks_produced}")
    print(f"  point count:     {summary.point_count}")
