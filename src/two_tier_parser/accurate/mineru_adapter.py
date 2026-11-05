"""Compatibility boundary for the pinned MinerU public API.

Keep upstream imports lazy so API and contract checks run without model weights.
"""

import re
from typing import Any, Iterator


def analyze(pdf_bytes: bytes, *, use_vlm: bool) -> dict[str, Any]:
    """Parse a PDF in memory and normalize MinerU 4 output for the service.

    ``high`` selects the VLM tier; ``medium`` selects MinerU's small-model tier
    and does not initialize a VLM. A selected backend failure is propagated;
    inference is never retried through a different backend.
    """
    from mineru.backend.analyze import doc_analyze
    from mineru.render import render_content_list, render_markdown

    effort = "high" if use_vlm else "medium"
    middle_json, _ = doc_analyze(
        file_bytes=pdf_bytes,
        effort=effort,
        parse_mode="auto",
        file_suffix="pdf",
    )
    blocks = render_content_list(middle_json)
    payloads = list(_image_payloads(middle_json.model_dump()))
    sizes = _page_sizes(pdf_bytes)
    images: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    image_ids: dict[str, str] = {}
    for item in blocks:
        kind = item.get("type")
        page = item.get("page_idx", 0)
        bbox = _page_bbox(item.get("bbox"), sizes, page)
        if kind == "image":
            # MinerU 4 puts the image in the item itself as a data URI. Pairing with the
            # ordered payload list is only the fallback for an item that carries no source.
            source = item.get("img_path") or item.get("content") or ""
            _add_image(
                images,
                image_ids,
                _raw_base64(source or (payloads.pop(0) if payloads else "")),
                page,
                bbox,
            )
        elif kind == "table":
            content = item.get("table_body") or ""
            if content:
                tables.append(
                    {
                        "table_id": f"page_{page}_table_{len(tables)}",
                        "markdown": content,
                        "page": page,
                        "bbox": bbox,
                    }
                )
            elif item.get("img_path"):
                # a table MinerU could not recognize is rendered as its picture
                _add_image(images, image_ids, _raw_base64(item["img_path"]), page, bbox)
        elif kind == "equation":
            content = item.get("text") or ""
            if not content and item.get("img_path"):
                # an equation MinerU could not recognize is rendered as its picture
                _add_image(images, image_ids, _raw_base64(item["img_path"]), page, bbox)
            if content:
                formulas.append(
                    {
                        "formula_id": f"page_{page}_formula_{len(formulas)}",
                        "latex": content,
                        "page": page,
                        "bbox": bbox,
                    }
                )

    return {
        "markdown": _reference_images(render_markdown(middle_json), image_ids),
        "images": images,
        "tables": tables,
        "formulas": formulas,
        "pages": len(middle_json.pages),
    }


def _add_image(
    images: list[dict[str, Any]],
    image_ids: dict[str, str],
    image_base64: str,
    page: int,
    bbox: list[float],
) -> None:
    image_id = f"page_{page}_img_{len(images)}"
    image_ids.setdefault(image_base64, image_id)
    images.append({"image_id": image_id, "image_base64": image_base64, "page": page, "bbox": bbox})


_INLINE_IMAGE = re.compile(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=]+)")


def _reference_images(markdown: str, image_ids: dict[str, str]) -> str:
    """Replace the base64 images MinerU 4 inlines in Markdown with their ``image_id``.

    Each picture is returned once, in ``images``; embedding it again in the Markdown made
    an 11 page response over a thousand times larger. Images are matched by content, so
    the ids stay right whatever order the Markdown shows them in. A payload with no entry
    in ``images`` stays embedded: it is the only copy, and dropping it would lose content.
    """
    return _INLINE_IMAGE.sub(lambda m: image_ids.get(m.group(1), m.group(0)), markdown)


def _image_payloads(value: Any) -> Iterator[str]:
    """Collect embedded image payloads from nested MinerU image body blocks."""
    if isinstance(value, dict):
        if value.get("type") == "image_body" and value.get("image_base64"):
            yield value["image_base64"]
        for child in value.values():
            yield from _image_payloads(child)
    elif isinstance(value, list):
        for child in value:
            yield from _image_payloads(child)


def _raw_base64(value: str) -> str:
    """Return the base64 bytes of an image, without a ``data:image/...;base64,`` prefix.

    The response field is documented as the image's base64 bytes, as the fast parser returns.
    """
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _page_sizes(pdf_bytes: bytes) -> list[tuple[float, float]]:
    """Return each page's (width, height) in points, as the PDF displays it."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        return [tuple(document[index].get_size()) for index in range(len(document))]
    finally:
        document.close()


def _page_bbox(bbox: Any, sizes: list[tuple[float, float]], page: int) -> list[float]:
    """Convert MinerU 4's box (integers in thousandths of the page) to page points.

    MinerU 4's output carries no page size, so the PDF's own sizes are used. The values are
    rounded to a thousandth of the page by MinerU, which is under a point on a normal page.
    A box that is missing, malformed or on an unknown page is returned empty rather than
    guessed, because callers use it to place highlights and crops.
    """
    if not bbox or len(bbox) != 4 or not 0 <= page < len(sizes):
        return []
    width, height = sizes[page]
    x0, y0, x1, y1 = bbox
    return [
        round(x0 * width / 1000, 2),
        round(y0 * height / 1000, 2),
        round(x1 * width / 1000, 2),
        round(y1 * height / 1000, 2),
    ]
