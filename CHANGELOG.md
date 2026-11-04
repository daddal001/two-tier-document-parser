# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fast parser: changes callers can observe

- `metadata.version` reports the installed `pymupdf4llm` version instead of a fixed `0.2.0`.
- A claim-check request with an empty or non-string `filename`, or a non-string `task_id`,
  returns 400 (an empty `filename` used to surface as 502 from the storage read).
- Unexpected failures return a generic 500 whose detail carries only a `correlation_id`;
  the exception text is in the service log under the same id.
- `/health` reports the real `no_gil` value and whether the layout engine loaded
  (`layout_enabled`). It is `healthy` when no object store is configured and `degraded`
  only when a configured store is unreachable; the new `storage_configured` field says which.
- `/metrics` returns the documented Prometheus series (`prometheus-client` is now part of
  the `fast` extra).
- When the wall-clock deadline stops a task-aware parse, `metadata.truncated` is
  `"deadline"` and `metadata.last_page_parsed` names the last page included. The key is
  absent on a complete parse.
- Oversized input is rejected before it is buffered (same 413, same 100 MiB limit). Uploads are counted as they stream, so a chunked body, or one that understates its `Content-Length`, is cut off at the limit before the rest is spooled to temporary storage. A JSON request body is capped at 1 MiB.
- Numeric settings (`WORKERS`, `PAGES_WORKERS`, `MAX_INFLIGHT_DOCUMENTS`, the health timeout and so on) treat an empty value as unset, and a malformed or non-positive value stops startup with a message naming the variable. A `--log-config` handler on uvicorn's loggers is left in place. The ANSI `color_message` field uvicorn adds is no longer written.
- Output written by native libraries outside Python logging is not JSON. The one known case is onnxruntime's `pthread_setaffinity_np failed` line from the layout model when the process runs on a restricted CPU set; it is harmless and appears once per process start.
- Request limits and admission control match `/parse` and `/parse-pages` after removing the server's `root_path`, so they also apply behind a path-prefixing proxy (`uvicorn --root-path`).
- Under the default container command every log line is JSON: uvicorn's startup, error and access records go through the same formatter. Hosts that configure logging themselves are left alone.
- Parser worker processes apply the same JSON logging configuration as the server, including pools that replace a failed one, so worker log lines are structured and honor `LOG_LEVEL` under every multiprocessing start method.
- A claim-check request with no object store configured returns 501.
- Log lines are one JSON object each with `timestamp`, `level`, `logger`, `message` and
  `service_name`, plus `trace_id`/`span_id` inside a span and the request fields
  (`task_id`, `duration_ms`, `error_type`, `correlation_id`). `LOG_LEVEL` is accepted in any
  case; emitted `level` values stay upper case.
- Trace export needs an explicit `OTEL_EXPORTER_OTLP_ENDPOINT`.
- Shutdown has a time budget (`SHUTDOWN_DRAIN_SECONDS`, default 10). Queued work is dropped,
  running work gets the budget to finish, and worker processes still running afterwards are
  terminated. Shutdown no longer blocks the event loop, and a stuck worker can no longer
  keep the process alive until the platform kills it.

### Fast parser: additions that change nothing by default

- `OBJECT_STORE_*` settings (endpoint, access key, secret key, TLS, region) for any
  S3-compatible store. The `MINIO_*` names remain fallbacks.
- `PARSER_ALLOWED_BUCKETS` restricts claim-check reads to a comma-separated bucket list
  (403 otherwise). Unset keeps the current behavior.
- `LOG_FORMAT=text` and `SERVICE_NAME` for local readability and log attribution.
- The image installs exactly the versions in `uv.lock`, runs as user 1000 and works with a
  read-only root filesystem. `PYTHON_GIL=0` was removed from the image and the sample
  Compose file; it has no effect on the standard CPython build.

### Fast parser: upgrade notes

Before deploying, check that:

1. Log parsers accept the new top-level keys and a timestamp without fractional seconds
   (the format is unchanged: `YYYY-MM-DDTHH:MM:SSZ`).
2. Nothing keys on `no_gil` being `true`; on the standard build it is `false`.
3. `OTEL_EXPORTER_OTLP_ENDPOINT` is set explicitly wherever traces are expected.
4. Callers send a non-empty string `filename` and a string `task_id` (or omit them).
5. Callers do not retry a 413.
6. Health checks keep working: `/health` still answers 200 in every state.
7. Nothing reads the text of a 500 response: the detail was `Parsing failed: <error>` and is
   now `Parsing failed (correlation_id=<id>)`.
8. Deployments that relied on the implicit `minio:9000` default with no `MINIO_*` or
   `OBJECT_STORE_*` variable set must now set `OBJECT_STORE_ENDPOINT`; with nothing set,
   claim-check requests return 501 and `/health` stays `healthy`.
9. A read-only root filesystem needs a writable `/tmp` (for example `--tmpfs /tmp`) and the
   default `/dev/shm`.
10. The running image matches the commit you believe it is; `pip freeze` in the image
   should be compared with the `uv.lock` of the release before switching.
11. Uvicorn with `--workers N` starts N independent copies of the service, each with its own pools and admission limits, and its supervisor still prints a few plain-text startup lines. Behind a TLS-terminating proxy, set uvicorn's `FORWARDED_ALLOW_IPS` so trailing-slash redirects keep `https`.
12. `SERVICE_NAME`, when set, overrides the `service_name` field on every log line.

### Accurate parser: changes callers can observe

- `POST /parse` admits at most `MAX_INFLIGHT_DOCUMENTS` documents at once, running plus
  queued (default: twice `WORKERS`). Further requests get an immediate 503 with
  `Retry-After: 5` and are rejected before their upload is read, including behind a
  path-prefixing proxy (`uvicorn --root-path`). `/health` is never queued behind parsing.
  A slot stays taken until the worker thread finishes, even if the client disconnects.
- Unexpected failures, including errors reported by MinerU, return a generic 500 whose
  detail carries only a `correlation_id` (`Parsing failed (correlation_id=<id>)`); the error
  text and traceback are in the service log under the same id.
- Numeric settings (`WORKERS`, `WORKER_MAX_TASKS`, `MAX_INFLIGHT_DOCUMENTS`) treat an empty
  value as unset, and a malformed or non-positive value stops startup with a message naming
  the variable.
- The response size reported in the "Parse complete" log line is measured on the worker
  thread, so serializing a large result no longer blocks the event loop, and the periodic
  cleanup log reports the number of tasks it covered (it used to report zero).
- Log lines are JSON with `service_name: accurate-parser` (`LOG_LEVEL` in any case, `SERVICE_NAME` to override), uvicorn's own records included. MinerU's loguru records go through the same JSON formatter (logger name `mineru.*`), and its progress bars are switched off, because a bar redrawn on stderr could leave the next JSON record unparseable. Output written by native libraries outside Python logging stays plain text; the one known case is MuPDF's `Object 0 0 not defined.` line.
- The 500 MiB upload limit is enforced while the upload streams, before it is spooled to disk or read into memory (same 413, same `File too large (max 500MB)` detail). A declared length above the limit is refused without reading a byte; a chunked body, or one that understates its length, is cut off at the limit.
- Shutdown has a time budget (`SHUTDOWN_DRAIN_SECONDS`, default 10). A thread running MinerU inference cannot be interrupted, so if inference is still running when the budget ends the process exits instead of waiting for the platform to kill it.
- The response shape is unchanged. `metadata.version` is the installed MinerU version.
- `image_base64` is the image's base64 bytes, without a `data:` prefix, and every `bbox` is in page points (origin at the top left), as before. MinerU 4 reports boxes in thousandths of the page and embeds images as data URIs, so the service converts both; the boxes inherit MinerU's rounding to a thousandth of the page, under one point on a normal page. A box that cannot be placed is returned empty rather than guessed.

### Accurate parser: additions and packaging changes

- MinerU is pinned to the 4.0.12 release and used through its public in-memory API behind
  one compatibility module (`mineru_adapter.py`), so an upstream change touches one file.
- A CPU image (`deploy/Dockerfile.accurate.cpu`) runs MinerU's medium-effort pipeline with
  ONNX models and no GPU. The CUDA image runs the high-effort VLM path on an NVIDIA host.
- `deploy/entrypoint-accurate.sh` prepares and verifies the model files before serving and
  fails the container if a repository is incomplete, instead of serving a service that
  fails on its first request. `MINERU_DOWNLOAD_MODELS` accepts `all`, `pipeline`, `vlm`,
  `basic` or `standard`.
- Docker Compose services are now profile based: `accurate-parser` became
  `accurate-parser-cuda` (profile `cuda`) and `accurate-parser-cpu` (profile `cpu`); start
  one with `docker compose --profile cpu up`. The CPU profile keeps its model files in the
  `cpu-models` volume. The `mineru-models` and `huggingface-cache` volumes are no longer
  used: the CUDA image bakes its models in, and a volume over them would hide the models of
  a newer image.
- Both accurate images run as user 1000 and keep everything MinerU writes under
  `MINERU_HOME=/opt/mineru` (also `HOME`), so they work when the platform forces that user
  and sets no home directory. Nothing lives under `/root` any more.
- Model weights are downloaded at the commits recorded in
  `src/two_tier_parser/accurate/model_revisions.json`, not at whatever the repositories hold
  on build day, and the CUDA image bakes them in through the same pinned path. A repository
  with no recorded commit stops the download with the command that records one
  (`scripts/pin_model_revisions.py`); `MINERU_ALLOW_UNPINNED_MODELS=true` allows newest
  files instead, and `MINERU_MODEL_REVISIONS` points at another manifest. Only the
  Hugging Face source is pinned. MinerU reuses any repository it considers complete, so
  startup also compares the commits recorded in an existing download with the manifest and
  stops with the directory to remove when they differ, instead of serving older weights.
- The CPU image installs the exact versions in `deploy/constraints/accurate-cpu.txt`
  (regenerate with `make constraints`). Both images install with
  `deploy/constraints/mineru-compat.txt`, which limits `docvortex` to versions MinerU 4.0.12
  works with: its own metadata allows 0.6.8 and later, which fail every parse with
  `TypeError: argument 'private_ranges'`. The CUDA image also freezes the base image's torch
  and CUDA packages, so an install that needs a different version fails the build instead of
  replacing the stack vLLM was built for. The rest of the CUDA image's dependencies still
  resolve at build time.
- The legacy `MINERU_TOOLS_CONFIG_JSON` variable is still honored as an alias for MinerU 4's
  `MINERU_CONFIG`; its JSON keys are not read, so move model locations to `model.base_dir`
  or `MINERU_MODEL_BASE_DIR`.

### Accurate parser: upgrade notes

1. Replace `accurate-parser` with `accurate-parser-cuda` or `accurate-parser-cpu` in scripts
   and Compose overrides, and pass the profile on the command line.
2. Treat a 503 from `/parse` as retriable; honor `Retry-After`. Callers that already retry
   the fast parser's 503 need no other change.
3. Do not parse the text of a 500 response.
4. Existing model downloads in the old `huggingface-cache` volume are not reused; the first
   start downloads the MinerU 4 model repositories.
5. A tool that read MinerU 2 or 3 internals (magic-pdf JSON keys, `mineru-models-download`)
   must move to MinerU 4's configuration; the service itself needs no change.
6. Remove any `mineru-models` volume mount from overrides. A volume mounted at
   `/root/.mineru` or `/opt/mineru` shadows the models baked into the image.
7. Build the CUDA image where the models can be downloaded from Hugging Face; the build
   fails if the recorded commits cannot be fetched. Before the first rollout, run the
   container on the target GPU and parse a document: this repository's tests cannot run it.
8. A models volume filled before this release, or before the manifest last changed, is at a
   different commit from the manifest and makes startup fail with a message naming the
   directory. Remove the volume (or set `MINERU_ALLOW_UNPINNED_MODELS=true` to keep it).
9. Pin the image by digest, not `latest`, in the deployment, so a rebuild cannot change
   what runs until the deployment is updated on purpose.

### Added
- OpenTelemetry API-only instrumentation for distributed tracing (74+ span points)
- NullHandler logging pattern for library portability
- Structured logging with JSON format support
- `core/telemetry.py` module with `traced_operation` context manager
- GPU detection tracing spans
- Per-page parsing spans for debugging
- MinerU operation wrapping with detailed attributes

### Changed
- Made public documentation self-contained: removed application ADR references,
  private deployment settings and external repository links; documented storage
  ownership, configurable admission and optional cancellation for independent callers.
- **BREAKING:** Renamed entry points from `app.py` → `main.py` (industry standard)
- Added backwards-compatible `app.py` wrappers with deprecation warnings
- Updated Dockerfiles to use `main.py` entry points
- Raised FastAPI, `python-multipart`, and `transformers` dependency floors to remove
  known vulnerable versions from parser service manifests
- Enhanced error handling with structured logging

### Deprecated
- `two_tier_parser.fast.app` module (use `two_tier_parser.fast.main` instead)
- `two_tier_parser.accurate.app` module (use `two_tier_parser.accurate.main` instead)

## [1.0.0] - 2026-01-15

### Added
- Initial release with two-tier architecture
- **Fast Parser** (Port 8004)
  - PyMuPDF4LLM-based text extraction
  - CPU-only operation
  - Sub-second per-page processing
  - Page-by-page fallback for complex PDFs
- **Accurate Parser** (Port 8005)
  - MinerU-based deep extraction
  - Automatic GPU detection and fallback
  - VLM backend for 95%+ accuracy (GPU)
  - Pipeline backend for 80-85% accuracy (CPU)
  - Multimodal extraction (images, tables, formulas)
- Docker Compose deployment configuration
- Health check endpoints
- Comprehensive API documentation
- Python package installation support

### Infrastructure
- FastAPI-based REST APIs
- ProcessPoolExecutor for parallel parsing
- Temporary file management with cleanup
- Configurable timeouts and resource limits

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-01-15 | Initial two-tier architecture release |
| (Unreleased) | - | OpenTelemetry instrumentation, `main.py` rename |
