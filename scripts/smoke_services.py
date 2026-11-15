"""Build/test Docker parser deployments and save a visual evidence report."""

import argparse
import base64
import hashlib
import html
import json
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pdf_page_count(path):
    """Read the fixture page count independently of either parser."""
    import pymupdf

    with pymupdf.open(path) as document:
        if document.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported by this validation harness")
        if document.page_count < 1:
            raise ValueError("PDF contains no pages")
        return document.page_count


def check_health(url, mode, timeout):
    """Check a service health endpoint and its public response fields."""
    response = requests.get(f'{url.rstrip("/")}/health', timeout=min(timeout, 10))
    response.raise_for_status()
    health = response.json()
    assert health.get("status") in {"healthy", "degraded"}, "Invalid health status"
    assert isinstance(health.get("workers"), int) and health["workers"] > 0
    assert isinstance(health.get("parser"), str) and health["parser"]
    if mode == "accurate":
        assert isinstance(health.get("gpu_available"), bool), "Missing GPU availability"
    else:
        assert "no_gil" in health, "Missing fast-parser no_gil health field"
    return health


def image_payload(image, index):
    """Normalize MinerU data URIs or wrapped base64 into displayable image data."""
    assert isinstance(image, dict), f"Image {index} must be an object"
    for key in ("image_id", "image_base64", "page"):
        assert key in image, f"Image {index} is missing {key}"
    assert isinstance(image["image_id"], str) and image["image_id"]
    assert isinstance(image["page"], int) and image["page"] >= 0
    payload = image["image_base64"]
    mime = image.get("mime")
    if not isinstance(payload, str):
        raise AssertionError(f"Image {index} data must be a base64 string")
    if payload.startswith("data:image/") and "," in payload:
        header, payload = payload.split(",", 1)
        mime = mime or header[5:].split(";", 1)[0]
    payload = re.sub(r"[^A-Za-z0-9+/=_-]", "", payload)
    payload += "=" * (-len(payload) % 4)
    decoded = base64.b64decode(payload, altchars=b"-_", validate=True)
    assert decoded, f"Image {index} has empty data"
    return mime or "image/png", base64.b64encode(decoded).decode("ascii"), decoded


def validate_image(image, index):
    """Validate an extracted image and its encoded bytes."""
    return image_payload(image, index)[2]


def validate_result(result, filename, expected, page_count, mode, elapsed):
    """Validate one parser response and summarize its extracted content.

    Args:
        filename: Multipart upload filename.
        expected: Text fragments required in the extracted Markdown.
        page_count: Independently determined number of pages in the input.
        mode: ``fast`` or ``accurate`` parser mode.
        elapsed: Measured end-to-end parse duration in seconds.

    Returns:
        A pair containing the summary statistics and decoded image payloads.

    Raises:
        AssertionError: Required fields or expected content are absent.
    """
    assert isinstance(result, dict), "Response must be a JSON object"
    if "error" in result:
        raise AssertionError(f"Parser returned an error: {result['error']}")
    assert result.get("markdown", "").strip(), "Empty markdown"
    metadata = result.get("metadata")
    assert isinstance(metadata, dict), "Missing metadata object"
    assert (
        metadata.get("pages") == page_count
    ), f"Reported {metadata.get('pages')} pages, expected {page_count}"
    assert isinstance(metadata.get("processing_time_ms"), (int, float))
    assert metadata["processing_time_ms"] >= 0
    assert metadata.get("filename") == filename, "Response filename does not match upload"
    assert isinstance(result.get("images"), list), "Missing images list"
    assert isinstance(result.get("tables", []), list), "Tables must be a list"
    assert isinstance(result.get("formulas", []), list), "Formulas must be a list"
    if mode == "accurate":
        assert "tables" in result and "formulas" in result
        for key in ("backend", "gpu_used", "accuracy_tier", "version"):
            assert key in metadata, f"Accurate metadata is missing {key}"
        assert isinstance(metadata["gpu_used"], bool)
        assert metadata["backend"] == ("vlm" if metadata["gpu_used"] else "pipeline")
    else:
        for key in ("parser", "version"):
            assert key in metadata, f"Fast metadata is missing {key}"
    for text in expected:
        assert text in result["markdown"], f"Missing expected content: {text}"
    image_bytes = [validate_image(image, index) for index, image in enumerate(result["images"])]
    for index, image in enumerate(result["images"]):
        assert image["page"] < page_count, f"images[{index}] refers to a page outside the PDF"
    for key in ("tables", "formulas"):
        if mode == "accurate":
            for index, item in enumerate(result[key]):
                assert isinstance(item, dict), f"{key}[{index}] must be an object"
                assert (
                    isinstance(item.get("page"), int) and 0 <= item["page"] < page_count
                ), f"{key}[{index}] refers to a page outside the PDF"
                content_key = "markdown" if key == "tables" else "latex"
                assert (
                    isinstance(item.get(content_key), str) and item[content_key].strip()
                ), f"{key}[{index}] is missing {content_key}"
    return {
        "elapsed_seconds": elapsed,
        "metadata": metadata,
        "markdown_code_points": len(result["markdown"]),
        "images": len(result["images"]),
        "tables": len(result["tables"]) if mode == "accurate" else None,
        "formulas": len(result["formulas"]) if mode == "accurate" else None,
    }, image_bytes


COMPOSE_FILE = PROJECT_ROOT / "deploy" / "docker-compose.yml"
SAMPLE_PDF = PROJECT_ROOT / "examples" / "data" / "sample.pdf"


def docker_gpu_available():
    """Return whether the host exposes an NVIDIA GPU for Docker inference."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [executable, "-L"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def deployment_profiles(gpu_available):
    """Always test CPU containers and add CUDA containers when host GPU exists."""
    return ["cpu", "cuda"] if gpu_available else ["cpu"]


def compose(profile, *arguments):
    """Run a command against this repository's Compose profile."""
    command = [
        "docker",
        "compose",
        "--profile",
        profile,
        "-f",
        str(COMPOSE_FILE),
        *arguments,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def check_health_when_ready(url, mode, timeout):
    """Wait for the containerized service to become healthy before parsing."""
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return check_health(url, mode, min(timeout, 10))
        except (requests.RequestException, ValueError, AssertionError) as exc:
            last_error = exc
            time.sleep(2)
    raise TimeoutError(f"{mode} Docker service did not become healthy: {last_error}")


def render_html(report, results, output_path):
    """Write a self-contained visual comparison for local review."""

    def metric(value):
        return "N/A" if value is None else f"{value:,}"

    rows = []
    for mode, summary in report["results"].items():
        rows.append(
            "<tr>"
            f"<th>{html.escape(mode)}</th>"
            f"<td>{html.escape(str(summary['metadata'].get('parser', 'MinerU')))}</td>"
            f"<td>{summary['metadata']['pages']}</td>"
            f"<td>{summary['elapsed_seconds']:.2f}s</td>"
            f"<td>{summary['markdown_code_points']:,}</td>"
            f"<td>{summary['images']}</td><td>{metric(summary['tables'])}</td>"
            f"<td>{metric(summary['formulas'])}</td></tr>"
        )
    sections = []
    for mode, result in results.items():
        parser_mode = mode.rsplit("/", 1)[-1]
        markdown = result.get("markdown", "")
        markdown = re.sub(
            r"!\[[^]]*\]\(data:image/[^;]+;base64,[^)]*\)",
            "![embedded image omitted here; see extracted images below]",
            markdown,
        )
        md = html.escape(markdown)
        media = []
        for image in result.get("images", []):
            mime, normalized_base64, _ = image_payload(image, image["image_id"])
            media.append(
                f'<figure><img alt="Extracted image {html.escape(str(image["image_id"]))}" '
                f'src="data:{html.escape(mime)};base64,{normalized_base64}">'
                f'<figcaption>Zero-based page index {image["page"]} · '
                f'{html.escape(str(image["image_id"]))}'
                "</figcaption></figure>"
            )
        if parser_mode == "accurate":
            tables = "".join(
                f"<details><summary>Table {index + 1} (zero-based page index "
                f"{item.get('page', '?')})</summary>"
                f"<pre>{html.escape(item.get('markdown', ''))}</pre></details>"
                for index, item in enumerate(result["tables"])
            )
            formulas = "".join(
                f"<details><summary>Formula {index + 1} (zero-based page index "
                f"{item.get('page', '?')})</summary>"
                f"<pre>{html.escape(item.get('latex', ''))}</pre></details>"
                for index, item in enumerate(result["formulas"])
            )
            table_content = tables or "<p>None returned.</p>"
            formula_content = formulas or "<p>None returned.</p>"
            table_count = str(len(result["tables"]))
            formula_count = str(len(result["formulas"]))
        else:
            table_content = (
                "<p>Not reported separately by the fast API; inspect Markdown above.</p>"
            )
            formula_content = (
                "<p>Not reported separately by the fast API; inspect Markdown above.</p>"
            )
            table_count = "N/A"
            formula_count = "N/A"
        sections.append(
            f"<section><h2>{html.escape(mode.title())} output</h2>"
            f"<h3>Markdown ({len(result.get('markdown', '')):,} Python string code points)</h3>"
            f"<pre>{md}</pre>"
            f"<h3>Extracted images ({len(media)})</h3><div class=images>{''.join(media)}</div>"
            f"<h3>Tables ({table_count})</h3>{table_content}"
            f"<h3>Formulas ({formula_count})</h3>{formula_content}</section>"
        )
    health = html.escape(json.dumps(report["health"], indent=2, ensure_ascii=False))
    document = (
        "<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport "
        "content='width=device-width,initial-scale=1'><title>Parser sample validation</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;"
        "color:#17212b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4dc;"
        "padding:.5rem;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f4f6f8;padding:1rem;border-radius:.4rem;max-height:35rem;overflow:auto}"
        ".images{display:flex;flex-wrap:wrap;gap:1rem}figure{margin:0;max-width:30rem}"
        "img{max-width:100%;max-height:24rem}figcaption{font-size:.85rem;color:#52606d}"
        "section{border-top:2px solid #ccd4dc;margin-top:2rem;padding-top:1rem}</style>"
        f"<h1>Parser sample validation</h1><p>Transport: {report['transport']} · "
        f"Fixture: {html.escape(report['fixture'])} · "
        f"{report['page_count']} pages · SHA-256 {report['fixture_sha256']}</p>"
        f"<details><summary>Service health</summary><pre>{health}</pre></details>"
        f"<table><thead><tr><th>Mode</th><th>Parser</th><th>Pages</th><th>Elapsed</th>"
        "<th>Markdown code points</th><th>Images</th><th>Tables</th><th>Formulas</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{''.join(sections)}</html>"
    )
    output_path.write_text(document, encoding="utf-8")


def main():
    """Build, run, and validate the repository's Docker parser deployments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument(
        "--server-metadata",
        type=Path,
        help="JSON with server hardware, model revision and settings",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=Path("parser_results/smoke.json"))
    args = parser.parse_args()
    if not SAMPLE_PDF.is_file():
        parser.error(f"required checked-in fixture is missing: {SAMPLE_PDF}")
    data = SAMPLE_PDF.read_bytes()
    page_count = pdf_page_count(SAMPLE_PDF)
    gpu_available = docker_gpu_available()
    profiles = deployment_profiles(gpu_available)
    report = {
        "server_metadata": (
            json.loads(args.server_metadata.read_text()) if args.server_metadata else None
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "client_platform": platform.platform(),
        "transport": "docker-compose-http",
        "compose_file": str(COMPOSE_FILE.relative_to(PROJECT_ROOT)),
        "cuda_gpu_detected": gpu_available,
        "cuda_status": (
            "scheduled" if gpu_available else "skipped: nvidia-smi did not detect a host NVIDIA GPU"
        ),
        "deployment_profiles": profiles,
        "fixture": str(SAMPLE_PDF.relative_to(PROJECT_ROOT)),
        "fixture_sha256": hashlib.sha256(data).hexdigest(),
        "page_count": page_count,
        "health": {},
        "results": {},
    }
    results = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        expected_backend = "vlm" if profile == "cuda" else "pipeline"
        try:
            compose(profile, "up", "--build", "--detach")
            for mode, port in (("fast", 8004), ("accurate", 8005)):
                key = f"{profile}/{mode}"
                artifact_tag = f"{profile}-{mode}"
                url = f"http://127.0.0.1:{port}"
                report["health"][key] = check_health_when_ready(url, mode, args.timeout)
                if mode == "accurate":
                    expected_gpu = profile == "cuda"
                    assert (
                        report["health"][key]["gpu_available"] is expected_gpu
                    ), f"{profile} accurate container reported unexpected GPU availability"
                start = time.perf_counter()
                response = requests.post(
                    f'{url.rstrip("/")}/parse',
                    files={"file": (SAMPLE_PDF.name, data, "application/pdf")},
                    timeout=args.timeout,
                )
                response.raise_for_status()
                result = response.json()
                args.output.with_name(
                    f"{args.output.stem}-{artifact_tag}-response.json"
                ).write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                if isinstance(result, dict) and isinstance(result.get("markdown"), str):
                    args.output.with_name(f"{args.output.stem}-{artifact_tag}.md").write_text(
                        result["markdown"], encoding="utf-8"
                    )
                summary, _ = validate_result(
                    result,
                    SAMPLE_PDF.name,
                    args.expect,
                    page_count,
                    mode,
                    time.perf_counter() - start,
                )
                if mode == "accurate":
                    actual_backend = result["metadata"]["backend"]
                    assert actual_backend == expected_backend, (
                        f"{profile} Compose profile selected {actual_backend}, "
                        f"expected {expected_backend}"
                    )
                    assert result["metadata"]["gpu_used"] == (profile == "cuda")
                report["results"][key] = summary
                results[key] = result
        except Exception as exc:
            report["results"][f"{profile}/error"] = {"error": str(exc)}
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            raise
        finally:
            compose(profile, "down", "--remove-orphans")
    html_path = args.output.with_suffix(".html")
    render_html(report, results, html_path)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nVisual review: {html_path}")


if __name__ == "__main__":
    main()
