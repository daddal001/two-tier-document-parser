"""MinerU 4 compatibility contracts without downloading model weights."""

import sys
import types

import pytest

from two_tier_parser.accurate import mineru_adapter as adapter

REAL_PAGE_SIZES = adapter._page_sizes  # the fixture below replaces it for every other test


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
    # Reading page sizes needs pypdfium2, which arrives with MinerU; tests set their own sizes.
    monkeypatch.setattr(adapter, "_page_sizes", lambda _: [])


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
    monkeypatch.setattr(adapter, "_page_sizes", lambda _: [(1000.0, 1000.0), (1000.0, 1000.0)])

    result = adapter.analyze(b"pdf bytes", use_vlm=False)

    assert result["images"] == [
        {
            "image_id": "page_0_img_0",
            "image_base64": "aGVsbG8=",
            "page": 0,
            "bbox": [1.0, 2.0, 3.0, 4.0],
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


def _analyze_items(monkeypatch, items, payloads=(), sizes=((600.0, 800.0),)):
    """Run the adapter over content-list ``items`` with the given page sizes (points)."""
    import mineru.backend.analyze as analyze_module
    import mineru.render as render_module

    blocks = [{"type": "image_body", "image_base64": payload} for payload in payloads]
    middle = type(
        "Middle",
        (),
        {"pages": list(sizes), "model_dump": lambda self: {"pages": [{"blocks": blocks}]}},
    )()
    monkeypatch.setattr(analyze_module, "doc_analyze", lambda **kwargs: (middle, object()))
    monkeypatch.setattr(render_module, "render_content_list", lambda _: items)
    monkeypatch.setattr(render_module, "render_markdown", lambda _: "document")
    monkeypatch.setattr(adapter, "_page_sizes", lambda _: list(sizes))
    return adapter.analyze(b"pdf bytes", use_vlm=False)


@pytest.mark.parametrize("key", ["img_path", "content"])
def test_an_image_data_uri_is_returned_as_raw_base64(monkeypatch, key):
    result = _analyze_items(
        monkeypatch, [{"type": "image", "page_idx": 0, key: "data:image/jpeg;base64,aGVsbG8="}]
    )

    assert result["images"][0]["image_base64"] == "aGVsbG8="  # the contract is raw base64 bytes


def test_a_data_uri_from_the_payload_fallback_is_also_stripped(monkeypatch):
    result = _analyze_items(
        monkeypatch,
        [{"type": "image", "page_idx": 0}],
        payloads=["data:image/png;base64,d29ybGQ="],
    )

    assert result["images"][0]["image_base64"] == "d29ybGQ="


def test_the_image_source_in_the_item_wins_over_the_ordered_payload_list(monkeypatch):
    result = _analyze_items(
        monkeypatch,
        [
            {"type": "image", "page_idx": 0, "img_path": "data:image/png;base64,Zmlyc3Q="},
            {"type": "image", "page_idx": 0, "img_path": "data:image/png;base64,c2Vjb25k"},
        ],
        payloads=["data:image/png;base64,d3Jvbmc="],  # must not be paired with either image
    )

    assert [i["image_base64"] for i in result["images"]] == ["Zmlyc3Q=", "c2Vjb25k"]


def test_boxes_are_converted_from_thousandths_of_the_page_to_page_points(monkeypatch):
    result = _analyze_items(
        monkeypatch,
        [
            {
                "type": "image",
                "page_idx": 0,
                "img_path": "data:image/png;base64,eA==",
                "bbox": [500, 100, 1000, 500],
            },
            {"type": "table", "page_idx": 1, "table_body": "<table/>", "bbox": [0, 0, 500, 250]},
            {"type": "equation", "page_idx": 0, "text": "x", "bbox": [250, 250, 750, 500]},
        ],
        sizes=((600.0, 800.0), (400.0, 200.0)),
    )

    assert result["images"][0]["bbox"] == [300.0, 80.0, 600.0, 400.0]
    assert result["tables"][0]["bbox"] == [0.0, 0.0, 200.0, 50.0]
    assert result["formulas"][0]["bbox"] == [150.0, 200.0, 450.0, 400.0]


def test_a_missing_or_unplaceable_box_is_returned_empty_not_wrong(monkeypatch):
    result = _analyze_items(
        monkeypatch,
        [
            {"type": "image", "page_idx": 0, "img_path": "data:image/png;base64,eA=="},
            {"type": "equation", "page_idx": 5, "text": "x", "bbox": [1, 2, 3, 4]},  # no such page
        ],
    )

    assert result["images"][0]["bbox"] == []
    assert result["formulas"][0]["bbox"] == []


def test_page_sizes_come_from_the_pdf_in_points(monkeypatch):
    class Page:
        def __init__(self, size):
            self.size = size

        def get_size(self):
            return self.size

    class Document:
        closed = False

        def __init__(self, data):
            assert data == b"pdf bytes"
            self.pages = [Page((576.0, 782.5)), Page((300.0, 400.0))]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def close(self):
            Document.closed = True

    pdfium = types.ModuleType("pypdfium2")
    pdfium.PdfDocument = Document
    monkeypatch.setitem(sys.modules, "pypdfium2", pdfium)
    monkeypatch.setattr(adapter, "_page_sizes", REAL_PAGE_SIZES)

    assert adapter._page_sizes(b"pdf bytes") == [(576.0, 782.5), (300.0, 400.0)]
    assert Document.closed


# ---- Markdown refers to images instead of embedding them ----


def _with_markdown(monkeypatch, markdown, items, **kwargs):
    import mineru.render as render_module

    result = _analyze_items(monkeypatch, items, **kwargs)
    monkeypatch.setattr(render_module, "render_markdown", lambda _: markdown)
    import mineru.backend.analyze as analyze_module

    middle = type("M", (), {"pages": [1], "model_dump": lambda self: {"pages": []}})()
    monkeypatch.setattr(analyze_module, "doc_analyze", lambda **k: (middle, object()))
    monkeypatch.setattr(render_module, "render_content_list", lambda _: items)
    monkeypatch.setattr(adapter, "_page_sizes", lambda _: [(600.0, 800.0)])
    return adapter.analyze(b"pdf", use_vlm=False) if result is not None else None


def _image(payload, page=0):
    return {"type": "image", "page_idx": page, "img_path": f"data:image/png;base64,{payload}"}


def test_inline_image_data_is_replaced_by_the_image_id(monkeypatch):
    markdown = (
        "Intro\n\n![figure](data:image/png;base64,QUJD)\n\nText ![](data:image/png;base64,REVG)\n"
    )

    result = _with_markdown(monkeypatch, markdown, [_image("QUJD"), _image("REVG", page=1)])

    assert "base64" not in result["markdown"]
    assert "![figure](page_0_img_0)" in result["markdown"]
    assert "![](page_1_img_1)" in result["markdown"]
    assert [i["image_id"] for i in result["images"]] == ["page_0_img_0", "page_1_img_1"]


def test_images_are_matched_by_content_not_by_order(monkeypatch):
    # the Markdown shows the second image first; ids must still follow the bytes
    markdown = "![](data:image/png;base64,REVG) then ![](data:image/png;base64,QUJD)"

    result = _with_markdown(monkeypatch, markdown, [_image("QUJD"), _image("REVG", page=1)])

    assert result["markdown"] == "![](page_1_img_1) then ![](page_0_img_0)"


def test_the_same_image_used_twice_keeps_one_reference(monkeypatch):
    markdown = "![a](data:image/png;base64,QUJD) ![b](data:image/png;base64,QUJD)"

    result = _with_markdown(monkeypatch, markdown, [_image("QUJD")])

    assert result["markdown"] == "![a](page_0_img_0) ![b](page_0_img_0)"


def test_images_inside_html_are_replaced_too(monkeypatch):
    markdown = '<table><tr><td><img src="data:image/jpeg;base64,QUJD" /></td></tr></table>'

    result = _with_markdown(monkeypatch, markdown, [_image("QUJD")])

    assert result["markdown"] == '<table><tr><td><img src="page_0_img_0" /></td></tr></table>'


def test_an_image_with_no_matching_entry_stays_embedded(monkeypatch):
    # it is the only copy of that picture, so it is kept rather than lost
    markdown = "![](data:image/png;base64,UNKNOWN0)"

    result = _with_markdown(monkeypatch, markdown, [_image("QUJD")])

    assert result["markdown"] == markdown


def test_an_unrecognized_table_keeps_its_picture(monkeypatch):
    markdown = "![](data:image/png;base64,QUJD)"
    table = {
        "type": "table",
        "page_idx": 0,
        "table_body": "",
        "img_path": "data:image/png;base64,QUJD",
    }

    result = _with_markdown(monkeypatch, markdown, [table])

    assert result["markdown"] == "![](page_0_img_0)"
    assert result["images"][0]["image_base64"] == "QUJD"
    assert result["tables"] == []


def test_markdown_without_images_is_untouched(monkeypatch):
    text = "# Title\n\nA paragraph with no pictures and a data: mention.\n"

    assert _with_markdown(monkeypatch, text, [])["markdown"] == text


def test_the_markdown_stays_small_however_large_the_images_are(monkeypatch):
    payload = "A" * 2_000_000  # a 1.5 MB picture

    result = _with_markdown(monkeypatch, f"![](data:image/png;base64,{payload})", [_image(payload)])

    assert len(result["markdown"]) < 100
    assert len(result["images"][0]["image_base64"]) == 2_000_000  # it is returned, once, in images


def test_an_unrecognized_equation_keeps_its_picture(monkeypatch):
    markdown = "before ![](data:image/png;base64,QUJD) after"
    equation = {
        "type": "equation",
        "page_idx": 0,
        "text": "",
        "img_path": "data:image/png;base64,QUJD",
    }

    result = _with_markdown(monkeypatch, markdown, [equation])

    assert result["markdown"] == "before ![](page_0_img_0) after"
    assert result["images"][0]["image_base64"] == "QUJD"
    assert result["formulas"] == []
