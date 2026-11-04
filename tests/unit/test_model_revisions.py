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
        repos = {}  # this test is about the pinned download, not the cache comparison

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


def write_cache(root, commits):
    """Lay out the metadata files huggingface_hub writes next to a ``local_dir`` download."""
    download = root / ".cache" / "huggingface" / "download" / "sub"
    download.mkdir(parents=True)
    for index, commit in enumerate(commits):
        (download / f"file{index}.bin.metadata").write_text(f"{commit}\netag\n1700000000.0\n")
    return root


def test_the_commits_of_a_cached_download_are_read_from_its_metadata(tmp_path):
    cache = write_cache(tmp_path, ["a" * 40, "a" * 40, "b" * 40])

    assert prepare_models.cached_commits(cache) == {"a" * 40, "b" * 40}


def test_a_directory_without_download_metadata_has_no_commits(tmp_path):
    (tmp_path / "model.onnx").write_text("weights")

    assert prepare_models.cached_commits(tmp_path) == set()


class CachedRepo:
    """A repository MinerU already considers complete, with files at the given commits."""

    name = "onnx"

    def __init__(self, local_dir):
        self.repos = {"huggingface": PINNED}
        self._local_dir = local_dir
        self.ensured = 0

    def local_dir(self):
        return self._local_dir

    def ensure(self, *, source):
        self.ensured += 1  # a ready repository is returned as it is, never re-downloaded


def run_main_with(monkeypatch, repo, pinned="a" * 40, allow_unpinned=False):
    install_downloader(monkeypatch)
    config = types.SimpleNamespace(model=types.SimpleNamespace(source="huggingface"))
    registry = types.ModuleType("mineru.registry")
    registry.model_repos_for_tier = lambda tier: [repo]
    sys.modules["mineru.model"].download.verify_model_repo = lambda r: types.SimpleNamespace(
        ready=True, missing_paths=[]
    )
    monkeypatch.setitem(sys.modules, "mineru.model.registry", registry)
    monkeypatch.setitem(sys.modules, "mineru.config", types.SimpleNamespace(config=config))
    monkeypatch.setattr(prepare_models, "load_revisions", lambda: {PINNED: pinned})
    monkeypatch.setattr(prepare_models, "_gpu_available", lambda: False)
    monkeypatch.setenv("MINERU_DOWNLOAD_MODELS", "pipeline")
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)
    if allow_unpinned:
        monkeypatch.setenv("MINERU_ALLOW_UNPINNED_MODELS", "true")
    else:
        monkeypatch.delenv("MINERU_ALLOW_UNPINNED_MODELS", raising=False)
    prepare_models.main()


def test_a_cached_repository_at_the_pinned_commit_is_accepted(monkeypatch, tmp_path):
    repo = CachedRepo(write_cache(tmp_path, ["a" * 40, "a" * 40]))

    run_main_with(monkeypatch, repo)

    assert repo.ensured == 1


def test_a_cached_repository_at_another_commit_is_refused_with_the_fix(monkeypatch, tmp_path):
    repo = CachedRepo(write_cache(tmp_path, ["b" * 40]))

    with pytest.raises(RuntimeError) as raised:
        run_main_with(monkeypatch, repo)

    message = str(raised.value)
    assert PINNED in message and "b" * 12 in message and "a" * 12 in message
    assert str(tmp_path) in message and "MINERU_ALLOW_UNPINNED_MODELS" in message


def test_a_cached_repository_with_mixed_commits_is_refused(monkeypatch, tmp_path):
    repo = CachedRepo(write_cache(tmp_path, ["a" * 40, "c" * 40]))

    with pytest.raises(RuntimeError, match="not at the pinned revision"):
        run_main_with(monkeypatch, repo)


def test_files_that_did_not_come_from_hugging_face_are_not_checked(monkeypatch, tmp_path):
    repo = CachedRepo(tmp_path)  # no download metadata: a local copy or another source

    run_main_with(monkeypatch, repo)

    assert repo.ensured == 1


def test_a_mismatch_is_allowed_when_unpinned_models_are_allowed(monkeypatch, tmp_path):
    repo = CachedRepo(write_cache(tmp_path, ["b" * 40]))

    run_main_with(monkeypatch, repo, allow_unpinned=True)

    assert repo.ensured == 1
