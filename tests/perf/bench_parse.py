"""Fast-parser benchmark harness — ship gate for ADR-0049.

Captures latency + throughput across fixture sizes and concurrency levels
and writes a JSON snapshot so pre/post-change deltas are grep-able.

Usage:
    pytest services/two_tier_document_parser/tests/perf/bench_parse.py --run-bench -q

Without --run-bench the module exits as skipped so regular CI doesn't pay
the multi-minute cost. Snapshots land under tests/perf/results/.

The ship gate (enforced at merge, not here) requires:
- single-doc p50 latency ≥ 2× improvement on the 100-page fixture
- 16-concurrent throughput ≥ 5× improvement on the 10-page fixture
- Stage 1 p95 (the /parse-pages call) remains ≤ 1500ms

We deliberately avoid heavy pytest-benchmark framework; plain time.perf_counter
keeps the harness hermetic and self-describing.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "perf_results" if (HERE.parent / "perf_results").exists() else HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SRC_ROOT = HERE.parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_addoption(parser):  # pragma: no cover - pytest hook wiring
    parser.addoption(
        "--run-bench",
        action="store_true",
        default=False,
        help="Enable fast-parser benchmark harness (slow; writes JSON snapshot)",
    )


@pytest.fixture(scope="module")
def run_bench(request):
    if not request.config.getoption("--run-bench", default=False):
        pytest.skip("benchmark disabled — pass --run-bench to run")
    return True


def _make_synthetic_pdf(num_pages: int) -> bytes:
    """Deterministic synthetic PDF so fixtures are reproducible across machines."""
    import pymupdf

    doc = pymupdf.open()
    for page_idx in range(num_pages):
        page = doc.new_page(width=612, height=792)
        tw = pymupdf.TextWriter(page.rect)
        tw.append(
            (72, 72),
            (
                f"Benchmark page {page_idx + 1} of {num_pages}. "
                "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
                * 20
            ),
        )
        tw.write_text(page)
    out = doc.tobytes()
    doc.close()
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return statistics.quantiles(values, n=100)[max(0, min(98, int(q) - 1))]


def _time_call(fn, *args, **kwargs) -> tuple[float, object]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def _bench_single(pdf_bytes: bytes, runs: int) -> dict:
    """Run parse_pdf `runs` times on the same bytes; report cold + steady."""
    from two_tier_parser.fast.service import parse_pdf

    timings: list[float] = []
    # Cold-start sample (first call, includes pymupdf4llm warmup)
    cold_s, _ = _time_call(parse_pdf, pdf_bytes, "bench.pdf")
    for _ in range(runs - 1):
        dt, _ = _time_call(parse_pdf, pdf_bytes, "bench.pdf")
        timings.append(dt)
    return {
        "cold_s": cold_s,
        "steady_p50_s": statistics.median(timings) if timings else cold_s,
        "steady_p95_s": _percentile(timings, 95) if len(timings) >= 3 else cold_s,
        "steady_mean_s": statistics.fmean(timings) if timings else cold_s,
        "runs": runs,
    }


def _bench_concurrent(pdf_bytes: bytes, concurrency: int, total_calls: int) -> dict:
    """Submit `total_calls` parse_pdf calls through a ProcessPool of `concurrency`."""
    from two_tier_parser.fast.service import parse_pdf

    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(parse_pdf, pdf_bytes, f"bench-{i}.pdf") for i in range(total_calls)]
        concurrent.futures.wait(futs)
    wall_s = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "total_calls": total_calls,
        "wall_s": wall_s,
        "throughput_calls_per_s": total_calls / wall_s if wall_s > 0 else float("inf"),
    }


def _snapshot_name() -> str:
    label = os.getenv("BENCH_LABEL") or ("baseline" if not _has_layout_available() else "post")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    return f"{label}-{stamp}.json"


def _has_layout_available() -> bool:
    try:
        import pymupdf.layout  # noqa: F401
        return True
    except Exception:
        return False


def test_bench_fast_parser(run_bench):
    """Single-doc latency and concurrent-throughput snapshot."""
    fixtures = {
        "1p": _make_synthetic_pdf(1),
        "10p": _make_synthetic_pdf(10),
        "100p": _make_synthetic_pdf(100),
    }

    out: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layout_available": _has_layout_available(),
        "python_version": sys.version.split()[0],
        "single_doc": {},
        "concurrent": {},
        "fixture_sizes_bytes": {k: len(v) for k, v in fixtures.items()},
    }

    for name, pdf in fixtures.items():
        out["single_doc"][name] = _bench_single(pdf, runs=5)

    bursts = fixtures["10p"]
    for c in (1, 4, 16):
        out["concurrent"][f"c{c}"] = _bench_concurrent(bursts, concurrency=c, total_calls=16)

    result_path = RESULTS_DIR / _snapshot_name()
    result_path.write_text(json.dumps(out, indent=2))
    print(f"\nbench snapshot → {result_path}")
    # Soft assertion: smoke-level sanity — throughput must be >0, latency finite
    assert out["concurrent"]["c1"]["throughput_calls_per_s"] > 0
    assert out["single_doc"]["1p"]["steady_mean_s"] > 0
