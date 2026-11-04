#!/usr/bin/env python3
"""Record, or check, the Hugging Face commits of the model repositories MinerU downloads.

The manifest (src/two_tier_parser/accurate/model_revisions.json) makes image builds download
the same weights every time. Run this after changing the MinerU submodule, or on purpose when
you want newer weights, then review and commit the result.

    python scripts/pin_model_revisions.py           # write the newest commits
    python scripts/pin_model_revisions.py --check   # list repositories whose newest commit differs

It needs MinerU installed (for its repository registry), huggingface_hub, and the network.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping

MANIFEST = (
    Path(__file__).parents[1] / "src" / "two_tier_parser" / "accurate" / "model_revisions.json"
)


def build_manifest(repo_ids: Iterable[str], sha_for: Callable[[str], str]) -> dict:
    """Return the manifest for ``repo_ids`` using ``sha_for`` to look up each commit."""
    return {"huggingface": {repo_id: sha_for(repo_id) for repo_id in sorted(set(repo_ids))}}


def drift(manifest: Mapping[str, Mapping[str, str]], sha_for: Callable[[str], str]) -> list:
    """Return ``(repository, pinned, newest)`` for each pin that is no longer the newest."""
    return [
        (repo_id, pinned, newest)
        for repo_id, pinned in sorted(manifest["huggingface"].items())
        if (newest := sha_for(repo_id)) != pinned
    ]


def registered_repo_ids() -> list:
    from mineru.model.registry import MODEL_REPOS

    return [repo.repos["huggingface"] for repo in MODEL_REPOS if "huggingface" in repo.repos]


def newest_commit(repo_id: str) -> str:
    from huggingface_hub import HfApi

    sha = HfApi().model_info(repo_id).sha
    if not sha:
        raise RuntimeError(f"Hugging Face returned no commit for {repo_id}")
    return sha


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="report drift; write nothing")
    args = parser.parse_args(argv)

    if args.check:
        current = json.loads(MANIFEST.read_text(encoding="utf-8"))
        registered = set(registered_repo_ids())
        missing = sorted(registered - set(current["huggingface"]))
        stale = drift(current, newest_commit)
        for repo_id in missing:
            print(f"not pinned: {repo_id}")
        for repo_id, pinned, newest in stale:
            print(f"newer commit: {repo_id} pinned {pinned[:12]}, newest {newest[:12]}")
        return 1 if missing else 0

    manifest = build_manifest(registered_repo_ids(), newest_commit)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST} with {len(manifest['huggingface'])} repositories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
