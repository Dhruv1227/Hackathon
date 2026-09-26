"""Deploy the app to a Hugging Face Docker Space.

  .venv/bin/hf auth login                                      # once, with a write token (you run this)
  .venv/bin/python -m deploy.push_space <hf-user>/<space-name>

Stages only what the running app needs (code, trained model, gazetteers, precomputed datasets as .json.gz,
OSM/geocode caches), refuses to upload if the Gemini key appears in any staged file, then creates the Space
(public, Docker) if needed and uploads. The key itself is added by you as a Space secret: GEMINI_API_KEY.
"""
import gzip
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".deploy"
COPY = ["app", "pipeline", "models", "data/gazetteer", "data/cache/osm", "data/cache/geocode.json",
        "Dockerfile", "requirements.txt", "README.md"]
RESULTS = ["data/processed/main_results.json", "data/processed/bonus_results.json"]
DOCKERIGNORE = "__pycache__\n*.pyc\n.env\n"


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for rel in COPY:
        src, dst = ROOT / rel, STAGE / rel
        if not src.exists():
            raise SystemExit(f"missing {rel}")
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for rel in RESULTS:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing {rel} (build it first)")
        dst = STAGE / (rel + ".gz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as f, gzip.open(dst, "wb", compresslevel=9) as g:
            shutil.copyfileobj(f, g)
    (STAGE / ".dockerignore").write_text(DOCKERIGNORE)

    # the app's quota floor needs to know how many proxy requests are left right now
    from pipeline.gemini import estimated_remaining
    left = estimated_remaining()
    if left is not None:
        docker = STAGE / "Dockerfile"
        docker.write_text(docker.read_text().replace("GEMINI_QUOTA_TOTAL=1000", f"GEMINI_QUOTA_TOTAL={left}"))
    return STAGE


def assert_no_secrets(folder: Path):
    from dotenv import dotenv_values
    key = (dotenv_values(ROOT / ".env").get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    for f in folder.rglob("*"):
        if f.name == ".env":
            raise SystemExit(f"refusing to deploy: {f} is an env file")
        if key and f.is_file():
            data = gzip.open(f).read() if f.suffix == ".gz" else f.read_bytes()
            if key.encode() in data:
                raise SystemExit(f"refusing to deploy: the Gemini key appears in {f.relative_to(folder)}")


def main(repo_id: str):
    folder = stage()
    assert_no_secrets(folder)
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
    print(f"staged {sum(1 for f in folder.rglob('*') if f.is_file())} files, {size:.1f} MB -> {repo_id}")
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="space",
                      commit_message="Deploy Living Flood Map")
    owner, name = repo_id.split("/")
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"App:   https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        raise SystemExit("usage: python -m deploy.push_space <hf-user>/<space-name>")
    main(sys.argv[1])
