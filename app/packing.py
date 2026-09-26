"""Compact storage for datasets served by the app (keeps the server within 512 MB).

A dataset is kept as its gzipped API payload (sent to the browser as-is) plus slim summary rows; the full
record objects never stay in memory.
"""
import gzip
import json
from pathlib import Path

# slim summary row: [row_id, group_id, text, shares, urgency, category, confidence, hazard, country, place names]


def summary_rows(records: list[dict]) -> list[list]:
    return [[r["id"], r["g"], r["text"], r["n"], r["urg"], r["cat"] or "OTHER", r["conf"], r.get("hz") or "",
             r.get("cc"), [p["name"] for p in r["places"][:3]]]
            for r in records if r["canon"] and r["rel"] == 1]


def compress_payload(payload: dict, level: int = 6) -> bytes:
    """payload = {"name", "stats", "records"}: exactly what /api/data returns."""
    return gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), level)


def write_packed(payload: dict, base: Path):
    """base = path without extension, e.g. data/processed/main_results."""
    base.with_name(base.name + ".json.gz").write_bytes(compress_payload(payload, 9))
    base.with_name(base.name + ".summary.json.gz").write_bytes(
        gzip.compress(json.dumps(summary_rows(payload["records"]), separators=(",", ":")).encode(), 9))
    base.with_name(base.name + ".meta.json").write_text(json.dumps(
        {"name": payload["name"], "stats": payload["stats"], "rows": len(payload["records"])}, separators=(",", ":")))
