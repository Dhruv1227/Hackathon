"""Pack precomputed datasets for low-memory serving (Render free tier: 512 MB).

For each data/processed/<name>_results.json this writes, next to it:
  <name>_results.json.gz          the full API payload, served as-is with Content-Encoding: gzip
  <name>_results.meta.json        {name, stats, rows} so the server never parses the big file
  <name>_results.summary.json.gz  slim rows for "Summarize this view" (relevant, one per group)

  .venv/bin/python -m deploy.pack
"""
import json
from pathlib import Path

from app.packing import write_packed

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data/processed"


def main():
    for src in sorted(PROCESSED.glob("*_results.json")):
        write_packed(json.loads(src.read_text()), src.with_suffix(""))
        gz = src.with_name(src.name + ".gz")
        print(f"{src.name}: {src.stat().st_size / 1e6:.1f} MB -> {gz.name} {gz.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
