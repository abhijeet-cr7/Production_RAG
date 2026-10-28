"""Every third-party import must be declared: a missing line only shows up in a clean install."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# import name -> distribution name, where they differ
ALIASES = {
    "jwt": "pyjwt", "yaml": "pyyaml", "docx": "python-docx", "bs4": "beautifulsoup4", "PIL": "pillow",
    "dotenv": "python-dotenv", "google": "google-generativeai", "qdrant_client": "qdrant-client",
    "rank_bm25": "rank-bm25", "confluent_kafka": "confluent-kafka", "sentence_transformers": "sentence-transformers",
    "pydantic_settings": "pydantic-settings", "tavily": "tavily-python", "starlette": "fastapi",
}
# Needed at runtime although no module imports them by name.
RUNTIME_ONLY = {"python-multipart", "uvicorn"}


def declared() -> set[str]:
    names = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(re.split(r"[<>=\[ ;#]", line)[0].lower().replace("_", "-"))
    return names


def third_party_imports() -> dict[str, str]:
    local = {p.name for p in ROOT.iterdir() if p.is_dir()}
    found: dict[str, str] = {}
    for path in ROOT.rglob("*.py"):
        parts = set(path.relative_to(ROOT).parts)
        if parts & {".venv", "node_modules", "tests", "frontend"}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name not in sys.stdlib_module_names and name not in local:
                    found.setdefault(name, str(path.relative_to(ROOT)))
    return found


def test_every_third_party_import_is_in_requirements_txt():
    reqs = declared()
    missing = {
        name: where for name, where in third_party_imports().items()
        if ALIASES.get(name, name).lower().replace("_", "-") not in reqs
    }
    assert not missing, f"imported but not in requirements.txt: {missing}"


def test_runtime_only_dependencies_are_declared():
    assert RUNTIME_ONLY <= declared()
