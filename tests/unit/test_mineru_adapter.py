"""MinerU 4 compatibility contracts without downloading model weights."""

import sys
import types

import pytest

from two_tier_parser.accurate import mineru_adapter as adapter


@pytest.fixture(autouse=True)
def _fake_mineru(monkeypatch):
    """Stand in for the pinned MinerU so these contracts run without it or its models.

    The adapter imports MinerU lazily, so installing the fake modules in ``sys.modules``
    is enough. Each test then patches the one function it exercises.
    """
    mineru = types.ModuleType("mineru")
    backend = types.ModuleType("mineru.backend")
    analyze_module = types.ModuleType("mineru.backend.analyze")
    render_module = types.ModuleType("mineru.render")
    analyze_module.doc_analyze = lambda **kwargs: (None, None)
    render_module.render_content_list = lambda value: []
    render_module.render_markdown = lambda value: ""
    mineru.backend = backend
    mineru.render = render_module
    backend.analyze = analyze_module
    for name, module in {
        "mineru": mineru,
        "mineru.backend": backend,
        "mineru.backend.analyze": analyze_module,
        "mineru.render": render_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(("use_vlm", "effort"), [(False, "medium"), (True, "high")])
def test_uses_public_in_memory_analysis_and_render_apis(monkeypatch, use_vlm, effort):
    from mineru import render as render_module
    from mineru.backend import analyze as analyze_module

    seen = {}
    middle = type("Middle", (), {"pages": [1, 2], "model_dump": lambda self: {}})()

    def doc_analyze(**kwargs):
        seen.update(kwargs)
        return middle, object()

    monkeypatch.setattr(analyze_module, "doc_analyze", doc_analyze)
    monkeypatch.setattr(render_module, "render_content_list", lambda value: [])
    monkeypatch.setattr(render_module, "render_markdown", lambda value: "parsed text")

    result = adapter.analyze(b"pdf bytes", use_vlm=use_vlm)

    assert seen == {
        "file_bytes": b"pdf bytes",
        "effort": effort,
        "parse_mode": "auto",
        "file_suffix": "pdf",
    }
    assert result["markdown"] == "parsed text"
    assert result["pages"] == 2
    assert result["images"] == result["tables"] == result["formulas"] == []


def test_selected_inference_failure_propagates_without_cpu_retry(monkeypatch):
    from mineru.backend import analyze as analyze_module

    calls = []

    def fail(**kwargs):
        calls.append(kwargs["effort"])
        raise RuntimeError("inference failed")

    monkeypatch.setattr(analyze_module, "doc_analyze", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        adapter.analyze(b"pdf bytes", use_vlm=True)
    assert calls == ["high"]


def test_normalizes_multimodal_content_list_fields(monkeypatch):
    import mineru.backend.analyze as analyze_module
    import mineru.render as render_module

    middle = type(
        "Middle",
        (),
        {
            "pages": [1, 2],
            "model_dump": lambda self: {
                "pages": [{"blocks": [{"type": "image_body", "image_base64": "aGVsbG8="}]}]
            },
        },
    )()
    monkeypatch.setattr(analyze_module, "doc_analyze", lambda **kwargs: (middle, object()))
    monkeypatch.setattr(
        render_module,
        "render_content_list",
        lambda _: [
            {"type": "image", "page_idx": 0, "bbox": [1, 2, 3, 4]},
            {"type": "table", "table_body": "<table></table>", "page_idx": 1},
            {"type": "equation", "text": "E=mc^2", "page_idx": 1},
        ],
    )
    monkeypatch.setattr(render_module, "render_markdown", lambda _: "document")

    result = adapter.analyze(b"pdf bytes", use_vlm=False)

    assert result["images"] == [
        {
            "image_id": "page_0_img_0",
            "image_base64": "aGVsbG8=",
            "page": 0,
            "bbox": [1, 2, 3, 4],
        }
    ]
    assert result["tables"][0]["markdown"] == "<table></table>"
    assert result["formulas"][0]["latex"] == "E=mc^2"


def test_image_payload_walker_collects_only_image_body_blocks():
    value = {
        "type": "page",
        "blocks": [
            {"type": "image_body", "image_base64": "aGVsbG8="},
            {"type": "table_body", "image_base64": "ignored"},
        ],
    }
    assert list(adapter._image_payloads(value)) == ["aGVsbG8="]
