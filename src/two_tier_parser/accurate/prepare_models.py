"""Prepare and validate the model repositories registered by MinerU.

Weights come from Hugging Face at the exact commits recorded in ``model_revisions.json``, so a
rebuild installs the same models instead of whatever the repositories hold that day. Pinning
MinerU's code does not pin its weights; this does.
"""

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

REVISIONS_ENV = "MINERU_MODEL_REVISIONS"
ALLOW_UNPINNED_ENV = "MINERU_ALLOW_UNPINNED_MODELS"


def _gpu_available() -> bool:
    """Check CUDA without making Torch a requirement for the fast parser."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def requested_tier(value: str, *, gpu_available: bool) -> str:
    """Map the established download selector to MinerU 4 deployment tiers."""
    if value in {"basic", "standard"}:
        return value
    if value == "pipeline":
        return "basic"
    if value == "vlm":
        return "standard"
    if value == "all":
        return "standard" if gpu_available else "basic"
    raise ValueError("MINERU_DOWNLOAD_MODELS must be all, pipeline, vlm, basic or standard")


def load_revisions() -> dict:
    """Return ``{repository: commit}`` for Hugging Face from the manifest.

    ``MINERU_MODEL_REVISIONS`` names another manifest file; otherwise the one bundled with
    this package is used.
    """
    override = os.getenv(REVISIONS_ENV)
    try:
        if override:
            text = open(override, encoding="utf-8").read()
        else:
            text = resources.files(__package__).joinpath("model_revisions.json").read_text("utf-8")
        revisions = json.loads(text)["huggingface"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"model revisions are unreadable ({REVISIONS_ENV}): {error}") from error
    if not isinstance(revisions, dict) or not all(
        isinstance(repo, str)
        and isinstance(sha, str)
        and len(sha) == 40
        and all(c in "0123456789abcdef" for c in sha)
        for repo, sha in revisions.items()
    ):
        raise ValueError(
            f"model revisions must map each repository to a full 40-character commit ({REVISIONS_ENV})"
        )
    return revisions


def cached_commits(local_dir: Any) -> set:
    """Return the commits huggingface_hub recorded for the files already in ``local_dir``.

    Empty when the files did not come from Hugging Face (another source, or a local copy).
    """
    download = Path(local_dir) / ".cache" / "huggingface" / "download"
    commits = set()
    for metadata in download.rglob("*.metadata"):
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].strip():
            commits.add(lines[0].strip())
    return commits


def require_pinned_cache(repo: Any, revisions: Mapping[str, str]) -> None:
    """Refuse a repository whose files on disk are not at the pinned commit.

    MinerU reuses any repository it considers complete without asking Hugging Face, so a
    pinned download never runs for it. A volume filled before the pin, or before the manifest
    changed, would otherwise keep serving the old weights while the manifest claims another.
    """
    repo_id = repo.repos.get("huggingface")
    pinned = revisions.get(repo_id) if repo_id else None
    if pinned is None:
        return
    commits = cached_commits(repo.local_dir())
    if commits and commits != {pinned}:
        found = ", ".join(sorted(commit[:12] for commit in commits))
        raise RuntimeError(
            f"Model repository {repo_id!r} in {repo.local_dir()} is not at the pinned revision "
            f"{pinned[:12]} (found {found}). Remove that directory, or the models volume, so "
            f"the pinned revision is downloaded, or set {ALLOW_UNPINNED_ENV}=true to keep "
            "what is there."
        )


def pin_huggingface_downloads(
    revisions: Mapping[str, str], *, allow_unpinned: bool = False
) -> None:
    """Make MinerU's Hugging Face downloader fetch the recorded commits.

    MinerU's downloader has no revision option and reads the repository's newest files, so
    its two Hugging Face entry points are wrapped to pass the pinned commit: the snapshot
    download, and the file listing it uses to check the download is complete. A repository
    with no recorded commit is refused unless ``allow_unpinned`` is set.
    """
    from mineru.model import download

    # shortcut: this wraps two module-level names in MinerU's downloader because it offers no
    # revision option. The submodule is pinned, a missing name fails here instead of silently
    # downloading unpinned, and the container tests cover a real download.
    real_snapshot = getattr(download, "hf_snapshot_download", None)
    real_api = getattr(download, "HfApi", None)
    if real_snapshot is None or real_api is None:
        raise RuntimeError(
            "MinerU's downloader changed, so model revisions cannot pin its Hugging Face "
            "downloads; update prepare_models.pin_huggingface_downloads for the new interface"
        )

    def revision_for(repo_id: str) -> Any:
        revision = revisions.get(repo_id)
        if revision is None and not allow_unpinned:
            raise RuntimeError(
                f"No pinned revision for model repository {repo_id!r}. Record one with "
                f"scripts/pin_model_revisions.py, or set {ALLOW_UNPINNED_ENV}=true to download "
                "the newest files (the build is then not reproducible)."
            )
        return revision

    def snapshot_download(repo_id: str, **kwargs: Any) -> Any:
        revision = revision_for(repo_id)
        if revision:
            kwargs["revision"] = revision
        return real_snapshot(repo_id, **kwargs)

    class PinnedHfApi(real_api):  # type: ignore[valid-type, misc]
        def list_repo_files(self, repo_id: str, **kwargs: Any) -> Any:
            revision = revision_for(repo_id)
            if revision:
                kwargs.setdefault("revision", revision)
            return super().list_repo_files(repo_id, **kwargs)

    download.hf_snapshot_download = snapshot_download
    download.HfApi = PinnedHfApi


def main() -> None:
    """Ensure the selected MinerU tier is present before serving requests.

    MinerU 4 loads ``model.base_dir`` from its YAML configuration and validates
    registered model files itself. ``MINERU_MODEL_SOURCE=local`` checks these
    exact files without downloading; Hugging Face and ModelScope use the
    upstream repository downloader and cache marker.
    """
    legacy_config = os.getenv("MINERU_TOOLS_CONFIG_JSON")
    if legacy_config and not os.getenv("MINERU_CONFIG"):
        # Preserve the established path variable as a migration bridge. MinerU
        # 4 reads YAML and its model layout changed; use model.base_dir in that
        # config (or MINERU_MODEL_BASE_DIR) for the new registered repositories.
        os.environ["MINERU_CONFIG"] = legacy_config

    from mineru.config import config
    from mineru.model.download import verify_model_repo
    from mineru.model.registry import model_repos_for_tier

    revisions = load_revisions()
    allow_unpinned = os.getenv(ALLOW_UNPINNED_ENV, "").strip().lower() in {"1", "true", "yes"}
    pin_huggingface_downloads(revisions, allow_unpinned=allow_unpinned)

    tier = requested_tier(
        os.getenv("MINERU_DOWNLOAD_MODELS", "all"), gpu_available=_gpu_available()
    )
    source = os.getenv("MINERU_MODEL_SOURCE", config.model.source)
    repos = model_repos_for_tier(tier)
    if not repos:
        raise RuntimeError(f"MinerU did not select model repositories for tier {tier!r}")
    for repo in repos:
        repo.ensure(source=source)
        if not allow_unpinned:
            require_pinned_cache(repo, revisions)
        status = verify_model_repo(repo)
        if not status.ready:
            raise RuntimeError(
                f"MinerU model repository {repo.name!r} is incomplete; missing: "
                + ", ".join(status.missing_paths)
            )


if __name__ == "__main__":
    main()
