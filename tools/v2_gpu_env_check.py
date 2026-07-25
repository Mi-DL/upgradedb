"""Fail-closed compatibility/provenance check for strict v2 GPU workers."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import shutil
from pathlib import Path


# PyKEEN 1.11.1's non-extra Requires-Dist entries, split by the intentionally
# layered deployment. Import every direct dependency so an incomplete
# --no-deps overlay fails before a model is started.
OVERLAY_DIRECT_MODULES = {
    "class-resolver": "class_resolver",
    "click": "click",
    "click-default-group": "click_default_group",
    "dataclasses-json": "dataclasses_json",
    "docdata": "docdata",
    "more-click": "more_click",
    "more-itertools": "more_itertools",
    "optuna": "optuna",
    "pykeen": "pykeen",
    "pystow": "pystow",
    "tabulate": "tabulate",
    "torch-max-mem": "torch_max_mem",
    "torch-ppr": "torch_ppr",
}

# ``pip check`` against the layered reference environment identified this single
# non-extra transitive dependency that is absent from the proven base venv.
OVERLAY_TRANSITIVE_MODULES = {
    "packaging": "packaging",
}

BASE_DIRECT_MODULES = {
    "pyyaml": "yaml",
    "requests": "requests",
    "tqdm": "tqdm",
    "typing-extensions": "typing_extensions",
}

# These versions are the audited direct dependencies of the reference base.
EXPECTED_BASE_DIRECT = {
    "pyyaml": "6.0.2",
    "requests": "2.34.2",
    "tqdm": "4.68.2",
    "typing-extensions": "4.15.0",
}

BASE_PYKEEN_NUMERICAL_MODULES = {
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
}

BASE_GRAPH_MODULES = {
    "torch-geometric": "torch_geometric",
    "torch-scatter": "torch_scatter",
}

BASE_MODULES = {
    **BASE_PYKEEN_NUMERICAL_MODULES,
    **BASE_GRAPH_MODULES,
    **BASE_DIRECT_MODULES,
}

# Audited from importlib.metadata.requires("pykeen") for PyKEEN 1.11.1;
# environment markers for extras are intentionally excluded.
PYKEEN_DIRECT_DISTRIBUTIONS = frozenset(
    {
        "class-resolver",
        "click",
        "click-default-group",
        "dataclasses-json",
        "docdata",
        "more-click",
        "more-itertools",
        "numpy",
        "optuna",
        "pandas",
        "pystow",
        "pyyaml",
        "requests",
        "scikit-learn",
        "scipy",
        "tabulate",
        "torch",
        "torch-max-mem",
        "torch-ppr",
        "tqdm",
        "typing-extensions",
    }
)

_accounted_direct = (
    set(OVERLAY_DIRECT_MODULES).difference({"pykeen"})
    | set(BASE_DIRECT_MODULES)
    | set(BASE_PYKEEN_NUMERICAL_MODULES)
)
if _accounted_direct != PYKEEN_DIRECT_DISTRIBUTIONS:
    raise RuntimeError(
        "internal PyKEEN direct-dependency inventory drift: "
        f"missing={sorted(PYKEEN_DIRECT_DISTRIBUTIONS - _accounted_direct)}, "
        f"extra={sorted(_accounted_direct - PYKEEN_DIRECT_DISTRIBUTIONS)}"
    )


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise SystemExit(f"unsupported lock entry at {path}:{line_number}: {raw!r}")
        name, version = (part.strip() for part in line.split("==", maxsplit=1))
        key = _canonical_distribution(name)
        if not name or not version or key in pins:
            raise SystemExit(f"invalid or duplicate lock entry at {path}:{line_number}: {raw!r}")
        pins[key] = version
    return pins


def _module_path(module) -> str:
    return str(Path(module.__file__).resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-torch", default="2.6.0+cu126")
    parser.add_argument("--expected-pykeen", default="1.11.1")
    parser.add_argument("--expected-click", default="8.4.2")
    parser.add_argument("--expected-ninja", default="1.13.0")
    parser.add_argument(
        "--overlay-lock",
        default="requirements/v2-gpu-nodeps-lock.txt",
        help="exact --no-deps overlay lock; every distribution is version-checked",
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--forbid-prefix",
        action="append",
        default=[],
        help="base numerical/graph module paths must not resolve below this prefix",
    )
    args = parser.parse_args()

    lock_path = Path(args.overlay_lock).resolve()
    pins = _read_pins(lock_path)
    expected_cli_pins = {
        "pykeen": args.expected_pykeen,
        "click": args.expected_click,
        "ninja": args.expected_ninja,
    }
    for distribution, expected in expected_cli_pins.items():
        locked = pins.get(distribution)
        if locked != expected:
            raise SystemExit(
                f"CLI/lock mismatch for {distribution}: expected={expected!r}, locked={locked!r}"
            )

    missing_direct_pins = sorted(set(OVERLAY_DIRECT_MODULES).difference(pins))
    if missing_direct_pins:
        raise SystemExit(
            "overlay lock omits PyKEEN direct runtime dependencies: "
            + ", ".join(missing_direct_pins)
        )

    installed_overlay: dict[str, str] = {}
    for distribution, expected in sorted(pins.items()):
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SystemExit(f"locked distribution is missing: {distribution}=={expected}") from exc
        if actual != expected:
            raise SystemExit(f"{distribution} mismatch: {actual} != {expected}")
        installed_overlay[distribution] = actual

    overlay_direct_modules = {
        distribution: importlib.import_module(module_name)
        for distribution, module_name in OVERLAY_DIRECT_MODULES.items()
    }
    overlay_transitive_modules = {
        distribution: importlib.import_module(module_name)
        for distribution, module_name in OVERLAY_TRANSITIVE_MODULES.items()
    }
    base_modules = {
        distribution: importlib.import_module(module_name)
        for distribution, module_name in BASE_MODULES.items()
    }
    torch = base_modules["torch"]

    base_direct_versions: dict[str, str] = {}
    for distribution, expected in EXPECTED_BASE_DIRECT.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise SystemExit(f"base {distribution} mismatch: {actual} != {expected}")
        base_direct_versions[distribution] = actual

    if torch.__version__ != args.expected_torch:
        raise SystemExit(f"torch mismatch: {torch.__version__} != {args.expected_torch}")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but unavailable")
    ninja_executable = shutil.which("ninja")
    if ninja_executable is None:
        raise SystemExit(
            f"ninja executable is unavailable for locked distribution {pins['ninja']!r}"
        )
    provenance = {name: _module_path(module) for name, module in base_modules.items()}
    provenance.update(
        {name: _module_path(module) for name, module in overlay_direct_modules.items()}
    )
    provenance.update(
        {name: _module_path(module) for name, module in overlay_transitive_modules.items()}
    )
    forbidden = [Path(raw).resolve() for raw in args.forbid_prefix]
    for name in BASE_MODULES:
        module_path = Path(provenance[name]).resolve()
        for prefix in forbidden:
            try:
                module_path.relative_to(prefix)
            except ValueError:
                continue
            raise SystemExit(f"{name} incorrectly resolves inside forbidden overlay: {module_path}")
    report = {
        "status": "compatible",
        "overlay_lock": str(lock_path),
        "overlay_pins": installed_overlay,
        "base_direct_versions": base_direct_versions,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "pykeen": installed_overlay["pykeen"],
        "click": installed_overlay["click"],
        "ninja": installed_overlay["ninja"],
        "ninja_executable": ninja_executable,
        "numpy": base_modules["numpy"].__version__,
        "pandas": base_modules["pandas"].__version__,
        "scipy": base_modules["scipy"].__version__,
        "scikit_learn": base_modules["scikit-learn"].__version__,
        "torch_geometric": base_modules["torch-geometric"].__version__,
        "torch_scatter": importlib.metadata.version("torch-scatter"),
        "provenance": provenance,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
