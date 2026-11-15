"""Reject untrusted or incomplete release metadata before changing the pin."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "codeql_updater", Path(__file__).resolve().parents[2] / "scripts/update_codeql_bundle.py"
)
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


def release(version="2.27.3"):
    tag = f"codeql-bundle-v{version}"
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": name,
                "browser_download_url": (
                    f"https://github.com/github/codeql-action/releases/download/{tag}/{name}"
                ),
                "digest": "sha256:" + "a" * 64,
            }
            for name in updater.ASSETS
        ],
    }


def test_official_stable_update_retains_every_platform():
    result = updater.manifest_from_release(release(), "2.27.2")
    assert result["version"] == "2.27.3"
    assert set(result["assets"]) == set(updater.ASSETS)


@pytest.mark.parametrize("flag", ["draft", "prerelease"])
def test_unreleased_bundles_are_rejected(flag):
    payload = release()
    payload[flag] = True
    with pytest.raises(ValueError, match="stable"):
        updater.manifest_from_release(payload, "2.27.2")


def test_downgrades_are_rejected():
    with pytest.raises(ValueError, match="downgrade"):
        updater.manifest_from_release(release("2.27.1"), "2.27.2")


@pytest.mark.parametrize("defect", ["missing", "duplicate", "digest", "external_url"])
def test_invalid_asset_metadata_cannot_be_pinned(defect):
    payload = release()
    if defect == "missing":
        payload["assets"].pop()
    elif defect == "duplicate":
        payload["assets"].append(copy.deepcopy(payload["assets"][0]))
    elif defect == "digest":
        payload["assets"][0]["digest"] = None
    else:
        payload["assets"][0]["browser_download_url"] = "https://example.com/untrusted.tar.gz"
    with pytest.raises(ValueError):
        updater.manifest_from_release(payload, "2.27.2")


# ---- choosing the bundle release among everything the repository publishes ----


def action_release(tag="v4.39.3"):
    """The CodeQL *action* releases share the repository; they carry no bundle assets."""
    return {"tag_name": tag, "draft": False, "prerelease": False, "assets": []}


def test_the_newest_bundle_is_chosen_among_action_releases():
    releases = [
        action_release("v4.39.3"),
        action_release("v3.39.3"),
        release("2.27.4"),
        action_release("v4.39.2"),
        release("2.27.3"),
    ]

    assert updater.select_release(releases)["tag_name"] == "codeql-bundle-v2.27.4"


def test_the_version_decides_not_the_order_of_the_list():
    releases = [release("2.27.3"), release("2.27.10"), release("2.27.9")]

    assert updater.select_release(releases)["tag_name"] == "codeql-bundle-v2.27.10"


def test_unstable_bundles_are_skipped():
    draft, pre = release("2.28.0"), release("2.27.9")
    draft["draft"], pre["prerelease"] = True, True

    assert updater.select_release([draft, pre, release("2.27.4")])["tag_name"].endswith("2.27.4")


def test_a_list_with_no_bundle_is_an_error():
    with pytest.raises(ValueError, match="No stable CodeQL bundle"):
        updater.select_release([action_release("v4.39.3")])
