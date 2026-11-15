"""Prepare and validate the model repositories registered by MinerU."""

import os


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

    tier = requested_tier(
        os.getenv("MINERU_DOWNLOAD_MODELS", "all"), gpu_available=_gpu_available()
    )
    source = os.getenv("MINERU_MODEL_SOURCE", config.model.source)
    repos = model_repos_for_tier(tier)
    if not repos:
        raise RuntimeError(f"MinerU did not select model repositories for tier {tier!r}")
    for repo in repos:
        repo.ensure(source=source)
        status = verify_model_repo(repo)
        if not status.ready:
            raise RuntimeError(
                f"MinerU model repository {repo.name!r} is incomplete; missing: "
                + ", ".join(status.missing_paths)
            )


if __name__ == "__main__":
    main()
