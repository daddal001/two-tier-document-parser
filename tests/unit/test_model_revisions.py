"""Model weights are downloaded at the revision recorded in the repository, not at the newest."""

import json
import re
import sys
import types
from importlib import resources

import pytest

from two_tier_parser.accurate import prepare_models

SHA = re.compile(r"^[0-9a-f]{40}$")
PINNED = "opendatalab/MinerU-4_models_onnx"
OTHER = "opendatalab/not-in-the-manifest"


def install_downloader(monkeypatch, *, with_api=True, with_snapshot=True):
    """Register a fake ``mineru.model.download`` that records the calls reaching Hugging Face."""
    calls = types.SimpleNamespace(snapshot=[], files=[])
    module = types.ModuleType("mineru.model.download")

    if with_snapshot:

        def snapshot_download(repo_id, **kwargs):
            calls.snapshot.append((repo_id, kwargs))
            return "/models/" + repo_id

        module.hf_snapshot_download = snapshot_download

    if with_api:

        class HfApi:
            def list_repo_files(self, repo_id, **kwargs):
                calls.files.append((repo_id, kwargs))
                return ["file.onnx"]

        module.HfApi = HfApi

    package = types.ModuleType("mineru.model")
    monkeypatch.setitem(sys.modules, "mineru", types.ModuleType("mineru"))
    monkeypatch.setitem(sys.modules, "mineru.model", package)
    monkeypatch.setitem(sys.modules, "mineru.model.download", module)
    package.download = module
    return module, calls


def test_the_bundled_manifest_pins_every_repository_to_a_full_commit():
    text = resources.files("two_tier_parser.accurate").joinpath("model_revisions.json").read_text()
    revisions = json.loads(text)["huggingface"]

    assert PINNED in revisions
    assert revisions and all(SHA.match(sha) for sha in revisions.values())


def test_the_manifest_is_loaded_from_the_package_by_default(monkeypatch):
    monkeypatch.delenv("MINERU_MODEL_REVISIONS", raising=False)

    assert prepare_models.load_revisions()[PINNED] == "358310b4f64b95f9fefc372ad899356e4111f376"


def test_the_manifest_location_can_be_overridden(monkeypatch, tmp_path):
    custom = tmp_path / "revisions.json"
    custom.write_text(json.dumps({"huggingface": {PINNED: "a" * 40}}))
    monkeypatch.setenv("MINERU_MODEL_REVISIONS", str(custom))

    assert prepare_models.load_revisions() == {PINNED: "a" * 40}


@pytest.mark.parametrize(
    "content",
    ['{"huggingface": {"x/y": "main"}}', '{"huggingface": []}', "[]", "not json"],
)
def test_a_malformed_manifest_is_rejected_by_name(monkeypatch, tmp_path, content):
    custom = tmp_path / "revisions.json"
    custom.write_text(content)
    monkeypatch.setenv("MINERU_MODEL_REVISIONS", str(custom))

    with pytest.raises(ValueError, match="MINERU_MODEL_REVISIONS|model revisions"):
        prepare_models.load_revisions()


def test_downloads_of_a_pinned_repository_request_that_revision(monkeypatch):
    module, calls = install_downloader(monkeypatch)
    prepare_models.pin_huggingface_downloads({PINNED: "b" * 40})

    module.hf_snapshot_download(PINNED, local_dir="/models/x", allow_patterns=["a/*"])

    assert calls.snapshot == [
        (PINNED, {"local_dir": "/models/x", "allow_patterns": ["a/*"], "revision": "b" * 40})
    ]


def test_the_expected_file_list_comes_from_the_pinned_revision_too(monkeypatch):
    module, calls = install_downloader(monkeypatch)
    prepare_models.pin_huggingface_downloads({PINNED: "b" * 40})

    assert module.HfApi().list_repo_files(PINNED) == ["file.onnx"]

    assert calls.files == [(PINNED, {"revision": "b" * 40})]


def test_a_repository_without_a_pin_is_refused_with_the_fix(monkeypatch):
    module, calls = install_downloader(monkeypatch)
    prepare_models.pin_huggingface_downloads({PINNED: "b" * 40})

    with pytest.raises(RuntimeError, match=f"{OTHER}.*pin_model_revisions"):
        module.hf_snapshot_download(OTHER, local_dir="/models/y")
    with pytest.raises(RuntimeError, match=OTHER):
        module.HfApi().list_repo_files(OTHER)
    assert calls.snapshot == [] and calls.files == []  # nothing reached Hugging Face


def test_unpinned_downloads_can_be_allowed_explicitly(monkeypatch):
    module, calls = install_downloader(monkeypatch)
    prepare_models.pin_huggingface_downloads({PINNED: "b" * 40}, allow_unpinned=True)

    module.hf_snapshot_download(OTHER, local_dir="/models/y")

    assert calls.snapshot == [(OTHER, {"local_dir": "/models/y"})]


@pytest.mark.parametrize("missing", ["with_snapshot", "with_api"])
def test_a_changed_downloader_fails_loudly_instead_of_downloading_unpinned(monkeypatch, missing):
    install_downloader(monkeypatch, **{missing: False})

    with pytest.raises(RuntimeError, match="cannot pin"):
        prepare_models.pin_huggingface_downloads({PINNED: "b" * 40})


def test_main_pins_the_downloads_before_any_repository_is_ensured(monkeypatch):
    module, calls = install_downloader(monkeypatch)
    config = types.SimpleNamespace(model=types.SimpleNamespace(source="huggingface"))
    seen = []

    class Repo:
        name = "onnx"

        def ensure(self, *, source):
            seen.append(source)
            module.hf_snapshot_download(PINNED, local_dir="/models/onnx")

    registry = types.ModuleType("mineru.model.registry")
    registry.model_repos_for_tier = lambda tier: [Repo()]
    module.verify_model_repo = lambda repo: types.SimpleNamespace(ready=True, missing_paths=[])
    monkeypatch.setitem(sys.modules, "mineru.model.registry", registry)
    monkeypatch.setitem(sys.modules, "mineru.config", types.SimpleNamespace(config=config))
    monkeypatch.setenv("MINERU_DOWNLOAD_MODELS", "pipeline")
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_MODEL_REVISIONS", raising=False)
    monkeypatch.setattr(prepare_models, "_gpu_available", lambda: False)

    prepare_models.main()

    assert seen == ["huggingface"]
    assert calls.snapshot[0][1]["revision"] == "358310b4f64b95f9fefc372ad899356e4111f376"
