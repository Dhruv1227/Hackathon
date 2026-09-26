"""Stage the Docker build context: exactly the files the Dockerfile copies, checked for secrets.

  .venv/bin/python -m deploy.pack        # (re)pack datasets after any rebuild
  .venv/bin/python -m deploy.stage       # -> .deploy/  (then: docker build .deploy)

Used by deploy/ec2.sh and deploy/push_space.py. The Gemini key is never staged: the host supplies it at run time.
"""
import gzip
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".deploy"
FILES = [
    "Dockerfile", "requirements-deploy.txt", "README.md", "app", "pipeline", "models", "data/gazetteer",
    *[f"data/processed/{d}_results.{ext}" for d in ("main", "bonus") for ext in ("json.gz", "meta.json", "summary.json.gz")],
    "data/cache/osm", "data/cache/geocode.json",
]
DOCKERIGNORE = "**/__pycache__\n*.pyc\n.env\n"


def stage(dest: Path = STAGE) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    for rel in FILES:
        src, dst = ROOT / rel, dest / rel
        if not src.exists():
            raise SystemExit(f"missing {rel} (run deploy.pack after building the datasets?)")
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (dest / ".dockerignore").write_text(DOCKERIGNORE)
    # the app's quota floor needs to know how many proxy requests are left right now
    from pipeline.gemini import estimated_remaining
    left = estimated_remaining()
    if left is not None:
        docker = dest / "Dockerfile"
        text = docker.read_text()
        import re
        docker.write_text(re.sub(r"GEMINI_QUOTA_TOTAL=\d+", f"GEMINI_QUOTA_TOTAL={left}", text))
    assert_no_secrets(dest)
    return dest


def assert_no_secrets(folder: Path):
    from dotenv import dotenv_values
    key = (dotenv_values(ROOT / ".env").get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    for f in folder.rglob("*"):
        if f.name == ".env" or f.suffix == ".pem":
            raise SystemExit(f"refusing to deploy: {f} is a secrets file")
        if key and f.is_file():
            data = gzip.open(f).read() if f.name.endswith(".gz") else f.read_bytes()
            if key.encode() in data:
                raise SystemExit(f"refusing to deploy: the Gemini key appears in {f.relative_to(folder)}")


if __name__ == "__main__":
    d = stage()
    files = [f for f in d.rglob("*") if f.is_file()]
    print(f"staged {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB in {d} (no secrets found)")
    print([l.strip() for l in (d / "Dockerfile").read_text().splitlines() if "QUOTA_TOTAL=" in l])
