"""Compatibility boundary for the pinned MinerU public API.

Keep upstream imports lazy so API and contract checks run without model weights.
"""

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
    images: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    for item in blocks:
        kind = item.get("type")
        page = item.get("page_idx", 0)
        bbox = item.get("bbox", [])
        if kind == "image":
            payload = item.get("content", "")
            image_base64 = (
                payload.split(",", 1)[1]
                if payload.startswith("data:image/") and "," in payload
                else (payloads.pop(0) if payloads else "")
            )
            images.append(
                {
                    "image_id": f"page_{page}_img_{len(images)}",
                    "image_base64": image_base64,
                    "page": page,
                    "bbox": bbox,
                }
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
        elif kind == "equation":
            content = item.get("text") or ""
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
        "markdown": render_markdown(middle_json),
        "images": images,
        "tables": tables,
        "formulas": formulas,
        "pages": len(middle_json.pages),
    }


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
