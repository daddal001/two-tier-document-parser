"""Model tier mapping remains compatible with current and legacy selectors."""

import sys
import types

import pytest

from two_tier_parser.accurate import prepare_models
from two_tier_parser.accurate.prepare_models import requested_tier


@pytest.mark.parametrize(
    ("requested", "gpu", "expected"),
    [
        ("all", False, "basic"),
        ("pipeline", False, "basic"),
        ("pipeline", True, "basic"),
        ("vlm", True, "standard"),
        ("vlm", False, "standard"),
        ("all", True, "standard"),
        ("basic", True, "basic"),
        ("standard", False, "standard"),
    ],
)
def test_legacy_and_current_model_selectors(requested, gpu, expected):
    assert requested_tier(requested, gpu_available=gpu) == expected


def test_invalid_model_selector_is_rejected():
    with pytest.raises(ValueError, match="MINERU_DOWNLOAD_MODELS"):
        requested_tier("unknown", gpu_available=False)


class _Repo:
    def __init__(self, name, missing=()):
        self.name = name
        self.missing = tuple(missing)
        self.sources = []

    def ensure(self, *, source):
        self.sources.append(source)


def _install_fake_mineru(monkeypatch, repos, configured_source="huggingface"):
    """Register fake ``mineru`` modules that expose only what ``main`` imports."""
    config = types.SimpleNamespace(model=types.SimpleNamespace(source=configured_source))
    modules = {
        "mineru": types.ModuleType("mineru"),
        "mineru.config": types.ModuleType("mineru.config"),
        "mineru.model": types.ModuleType("mineru.model"),
        "mineru.model.download": types.ModuleType("mineru.model.download"),
        "mineru.model.registry": types.ModuleType("mineru.model.registry"),
    }
    modules["mineru.config"].config = config
    modules["mineru.model.download"].verify_model_repo = lambda repo: types.SimpleNamespace(
        ready=not repo.missing, missing_paths=list(repo.missing)
    )
    tiers = []

    def model_repos_for_tier(tier):
        tiers.append(tier)
        return repos

    modules["mineru.model.registry"].model_repos_for_tier = model_repos_for_tier
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return tiers


def test_main_ensures_and_verifies_every_selected_repository(monkeypatch):
    repos = [_Repo("layout"), _Repo("tables")]
    tiers = _install_fake_mineru(monkeypatch, repos)
    monkeypatch.setenv("MINERU_DOWNLOAD_MODELS", "pipeline")
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "local")
    monkeypatch.setattr(prepare_models, "_gpu_available", lambda: False)

    prepare_models.main()

    assert tiers == ["basic"]
    assert [repo.sources for repo in repos] == [["local"], ["local"]]


def test_main_uses_the_configured_source_when_none_is_set(monkeypatch):
    repos = [_Repo("layout")]
    _install_fake_mineru(monkeypatch, repos, configured_source="modelscope")
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)
    monkeypatch.setattr(prepare_models, "_gpu_available", lambda: False)

    prepare_models.main()

    assert repos[0].sources == ["modelscope"]


def test_main_fails_when_a_repository_is_incomplete(monkeypatch):
    _install_fake_mineru(monkeypatch, [_Repo("layout", missing=["model.onnx"])])
    monkeypatch.setattr(prepare_models, "_gpu_available", lambda: False)

    with pytest.raises(RuntimeError, match="layout.*model.onnx"):
        prepare_models.main()


def test_main_fails_when_no_repositories_are_selected(monkeypatch):
    _install_fake_mineru(monkeypatch, [])
    monkeypatch.setattr(prepare_models, "_gpu_available", lambda: False)

    with pytest.raises(RuntimeError, match="did not select"):
        prepare_models.main()


def test_main_bridges_the_legacy_config_variable(monkeypatch):
    _install_fake_mineru(monkeypatch, [_Repo("layout")])
    monkeypatch.setattr(prepare_models, "_gpu_available", lambda: False)
    monkeypatch.setenv("MINERU_TOOLS_CONFIG_JSON", "/etc/mineru/legacy.json")
    # Set (empty) through monkeypatch so the value main() writes is rolled back afterwards.
    monkeypatch.setenv("MINERU_CONFIG", "")

    prepare_models.main()

    assert prepare_models.os.environ["MINERU_CONFIG"] == "/etc/mineru/legacy.json"


def test_gpu_probe_is_false_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError

    assert prepare_models._gpu_available() is False
