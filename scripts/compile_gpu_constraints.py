"""Pin the CUDA image's Python dependencies without replacing what its base image ships.

The vLLM base image is the tested stack: it owns torch, CUDA and vLLM, and every other package
in it is a version that vLLM was run with. Resolving the dependencies from scratch picks the
newest release of everything (dozens of major-version moves the base was never tested with), so
the compile starts from the base image's own versions (``deploy/constraints/vllm-base.txt``) and
changes only what MinerU or this service needs. ``uv`` prefers the versions already in its
output file, which is how the base versions are carried in.

    python scripts/compile_gpu_constraints.py                 # recompile accurate-gpu.txt
    python scripts/compile_gpu_constraints.py --refresh-base  # re-read vllm-base.txt from the image

The first form needs ``uv``; the second needs Docker and the network, and should be run when the
base image tag in ``deploy/Dockerfile.accurate`` changes.
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deploy/constraints/accurate-gpu.txt"
BASE_FREEZE = ROOT / "deploy/constraints/vllm-base.txt"
BASE_NATIVE = ROOT / "deploy/constraints/vllm-base-native.txt"
DOCKERFILE = ROOT / "deploy/Dockerfile.accurate"
# These distributions are supplied by the vLLM image, including its CUDA-specific wheel builds.
# A version with a local tag (+cu130) is one of those builds, such as torchcodec, which vLLM
# requires: PyPI's plain build of the same version must not replace it.
BASE_PACKAGES = re.compile(
    r"^(torch|torchvision|torchaudio|triton|vllm)==|^(nvidia-|cuda-)|^[A-Za-z0-9_.-]+==[^\s;]*\+"
)


def _name(requirement: str) -> str:
    """The normalized distribution name of a pinned requirement line."""
    return re.match(r"[A-Za-z0-9_.-]+", requirement)[0].lower().replace("_", "-")


def refresh_base() -> None:
    """Record the base image's Python packages (without its native stack) in vllm-base.txt."""
    tag = re.search(r"^FROM (vllm/vllm-openai:\S+)", DOCKERFILE.read_text(), re.M)[1]
    frozen = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python3", tag, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    candidates = [line for line in frozen.splitlines() if "==" in line and "@" not in line]
    lines = sorted(line for line in candidates if not BASE_PACKAGES.match(line))
    direct = [line for line in frozen.splitlines() if " @ " in line]  # installed from a local wheel
    owned = sorted(
        {_name(line) for line in candidates if BASE_PACKAGES.match(line)}
        | {_name(line) for line in direct}
    )
    BASE_NATIVE.write_text(
        f"# Packages the base image {tag} owns: its native stack and the CUDA builds it ships.\n"
        "# The pins never name them, at any version: the Dockerfile freezes the base's own, and a\n"
        "# pin the resolver chose can contradict that freeze (torchcodec 0.17.0 against the\n"
        "# base's 0.16.0+cu130) and fail the build.\n"
        "# Regenerate with: python scripts/compile_gpu_constraints.py --refresh-base\n"
        + "\n".join(owned)
        + "\n"
    )
    BASE_FREEZE.write_text(
        f"# Python packages of {tag} (linux/arm64 image; the pure-Python versions match amd64),\n"
        "# without its native torch, CUDA and vLLM packages. The CUDA pins start from these versions.\n"
        "# Regenerate with: python scripts/compile_gpu_constraints.py --refresh-base\n"
        + "\n".join(lines)
        + "\n"
    )


def compile_pins() -> None:
    """Resolve for each architecture starting from the base versions, then merge the results."""
    pins: dict[str, set[str]] = {}
    owned = {
        line.strip()
        for line in BASE_NATIVE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    with tempfile.TemporaryDirectory(prefix="gpu-constraints-") as directory:
        for architecture in ("x86_64", "aarch64"):
            compiled = Path(directory) / f"{architecture}.txt"
            compiled.write_text(
                BASE_FREEZE.read_text()
            )  # uv prefers the pins already in its output
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    "deploy/constraints/accurate-gpu.in",
                    "--python-version",
                    "3.12",
                    "--python-platform",
                    f"{architecture}-manylinux_2_28",
                    "--no-emit-package",
                    "mineru",
                    "--no-emit-package",
                    "two-tier-document-parser",
                    "-c",
                    "deploy/constraints/mineru-compat.txt",
                    "--override",
                    "deploy/constraints/overrides.txt",
                    "-o",
                    str(compiled),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            for line in compiled.read_text().splitlines():
                if line and not line.lstrip().startswith("#") and not BASE_PACKAGES.match(line):
                    if _name(line) not in owned:
                        pins.setdefault(line, set()).add(architecture)
    lines = []
    for requirement, architectures in sorted(pins.items()):
        if len(architectures) == 1:
            requirement += f" ; platform_machine == '{next(iter(architectures))}'"
        lines.append(requirement)
    OUTPUT.write_text(
        "# Generated for Linux x86_64 and aarch64, Python 3.12. Run make constraints-gpu.\n"
        "# Starts from the versions in vllm-base.txt; native CUDA/Torch/vLLM packages come from\n"
        "# the base image and are protected by its runtime-generated base-constraints.txt.\n"
        + "\n".join(lines)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--refresh-base", action="store_true", help="re-read vllm-base.txt from the image"
    )
    arguments = parser.parse_args()
    if arguments.refresh_base:
        refresh_base()
    compile_pins()


if __name__ == "__main__":
    main()
