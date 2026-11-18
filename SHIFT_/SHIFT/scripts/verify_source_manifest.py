"""Dependency-free SHA-256 integrity check for a tagged SHIFT source archive.

Does not import torch, run the model, download data, or assert scientific correctness.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.json"


def main():
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        raise ValueError("Source manifest must be a mapping")
    actual = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST or any(
            segment in {".git", "__pycache__", ".pytest_cache", ".venv"}
            for segment in path.relative_to(ROOT).parts
        ) or path.suffix == ".pyc":
            continue
        relative = path.relative_to(ROOT).as_posix()
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    modified = sorted(k for k in expected.keys() & actual.keys()
                      if expected[k] != actual[k])
    if missing or unexpected or modified:
        raise SystemExit(
            f"Manifest mismatch: missing={missing}, unexpected={unexpected}, modified={modified}"
        )
    print(f"Manifest valid: {len(actual)} source files (SHA-256).")


if __name__ == "__main__":
    main()
