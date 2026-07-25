#!/usr/bin/env python3
"""Build the ACM review PDF without promoting paper-number artifacts.

The canonical paper-number writer intentionally fails while the registry-audit
hold is active.  This builder uses the resolver's non-canonical preview path,
stores an ignored local cache with a complete hash binding, overlays that TeX
interface into a fresh temporary paper tree, and atomically publishes only the
review PDF.  After the hold is resolved, the same cache is sealed from the
receipt-verified canonical JSON/TeX bytes instead.  The builder never writes
``paper/generated/v2_numbers.tex`` or ``results_v2/paper_numbers.json``.

Refresh the governed number snapshot when a release audit needs to bind the
paper to the current data, metrics, verifiers, and hold state::

    python tools/build_paper_review.py --refresh-numbers

Normal wording, author, bibliography, and layout edits use the fast path.  It
accepts the last sealed number snapshot with a visible warning when its
governed binding is older than the repository state::

    python tools/build_paper_review.py
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReviewBuildError(RuntimeError):
    """The review PDF or its sealed number interface failed validation."""


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import generate_v2_benchmark_profile as benchmark_profile  # noqa: E402
    import generate_v2_paper_numbers as paper_numbers  # noqa: E402
    import public_release_policy as public_policy  # noqa: E402
    import resolve_v2_invalidation as resolution  # noqa: E402
    import summarize_v2_contemporary_references as contemporary_results  # noqa: E402
except ImportError as exc:
    if __name__ == "__main__":
        print(f"REVIEW BUILD REFUSED: cannot import a required verifier: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise


CACHE_SCHEMA = "upgrade-bench-v2/noncanonical-paper-review-cache/1"
CACHE_STATUS = "NON_CANONICAL_REVIEW_ONLY"
CACHE_RELATIVE = Path("output/paper-review-cache/current")
MANIFEST_NAME = "review_cache_manifest.json"
OUTPUT_RELATIVE = Path("output/pdf/upgrade-bench-kdd27-review.pdf")
PROFILE_JSON_RELATIVE = "results_v2/metrics/v2_benchmark_profile.json"
PROFILE_TEX_RELATIVE = "paper/generated/v2_benchmark_profile.tex"
CONTEMPORARY_CONFIG_RELATIVE = "configs/v2_contemporary_references.json"
CONTEMPORARY_JSON_RELATIVE = "results_v2/metrics/v2_contemporary_references.json"
CONTEMPORARY_CSV_RELATIVE = "results_v2/metrics/v2_contemporary_references.csv"
CONTEMPORARY_TEX_RELATIVE = "paper/generated/v2_contemporary_references.tex"
CONTEMPORARY_TOOL_RELATIVE = "tools/summarize_v2_contemporary_references.py"
CANONICAL_GUARD_PATHS = (
    resolution.NOTICE_PATH,
    resolution.PAPER_JSON_PATH,
    resolution.PAPER_TEX_PATH,
    CONTEMPORARY_CONFIG_RELATIVE,
    CONTEMPORARY_JSON_RELATIVE,
    CONTEMPORARY_CSV_RELATIVE,
    CONTEMPORARY_TEX_RELATIVE,
    CONTEMPORARY_TOOL_RELATIVE,
)
PAPER_SOURCE_FILES = (
    "main-acm.tex",
    "abstract.tex",
    "body.tex",
    "acknowledgments.tex",
    "appendix.tex",
    "refs.bib",
)
LATEX_ERROR_MARKERS = (
    "LaTeX Error",
    "Undefined control sequence",
    "Fatal error occurred",
    "There were undefined references",
    "There were undefined citations",
    "Overfull \\hbox",
    "Overfull \\vbox",
)
LATEX_RERUN_MARKERS = (
    "rerun to",
    "please rerun",
    "please (re)run",
    "label(s) may have changed",
    "labels may have changed",
)


@dataclass(frozen=True)
class ReviewInterface:
    """A verified paper-number interface sealed for an isolated review build."""

    cache_root: Path
    json_path: Path
    tex_path: Path
    manifest: Mapping[str, Any]
    binding_current: bool = True


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReviewBuildError(f"cannot render strict cache JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def _load_canonical_json(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    _require_regular_file(path, role)
    try:
        content = path.read_bytes()
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReviewBuildError(f"{role} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewBuildError(f"{role} root is not an object")
    if content != _canonical_json_bytes(payload):
        raise ReviewBuildError(f"{role} is not canonical sorted JSON")
    return payload, content


def _require_regular_file(path: Path, role: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReviewBuildError(f"{role} must be a regular file: {path}")
    return path


def _require_real_directory(path: Path, role: str) -> Path:
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    if path.is_symlink() or is_junction(path) or not path.is_dir():
        raise ReviewBuildError(f"{role} must be a real directory: {path}")
    return path


def _review_output_root(root: Path) -> Path:
    """Return the literal in-repository output directory, rejecting redirects."""

    resolved_root = root.resolve(strict=True)
    output = resolved_root / "output"
    if output.exists() or output.is_symlink():
        _require_real_directory(output, "review output directory")
    else:
        output.mkdir()
    resolved_output = output.resolve(strict=True)
    if resolved_output != output:
        raise ReviewBuildError(
            f"review output directory must not redirect away from {output}"
        )
    return resolved_output


def _require_local_output_path(root: Path, path: Path, role: str) -> Path:
    """Keep review caches and products on the ignored local output surface."""

    resolved_output = _review_output_root(root)
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(resolved_output)
    except ValueError as exc:
        raise ReviewBuildError(f"{role} must stay below {resolved_output}") from exc
    if not relative.parts:
        raise ReviewBuildError(f"{role} must be a child of {resolved_output}")
    return resolved


def _require_disjoint_review_paths(cache_root: Path, output: Path) -> None:
    """Prevent a custom PDF target from replacing any cache path or ancestor."""

    for child, parent in ((output, cache_root), (cache_root, output)):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise ReviewBuildError(
            f"review cache and PDF output must not overlap: {cache_root} versus {output}"
        )


@contextlib.contextmanager
def _review_build_lock(output_root: Path):
    """Serialize refresh/build transactions that share the local output surface."""

    lock_path = output_root / "paper-review-cache" / ".review-build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (OSError, ImportError) as exc:
            raise ReviewBuildError(
                "another review refresh/build is already using the local cache"
            ) from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            with contextlib.suppress(OSError):
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _safe_repository_file(root: Path, relative: str, role: str) -> Path:
    lexical = root / Path(relative)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReviewBuildError(f"cannot resolve {role}: {relative}: {exc}") from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ReviewBuildError(f"{role} escapes the repository: {relative}") from exc
    if lexical.is_symlink() or not lexical.is_file():
        raise ReviewBuildError(f"{role} must be a regular repository file: {relative}")
    return resolved


def _hash_inventory(root: Path, paths: Sequence[str], role: str) -> dict[str, str]:
    return {
        relative: _sha256_file(_safe_repository_file(root, relative, role))
        for relative in sorted(paths)
    }


def _validated_review_notice(root: Path) -> tuple[dict[str, Any], bytes]:
    """Validate either supported invalidation state without conflating schemas."""

    try:
        notice, notice_bytes = resolution._strict_canonical_json_file(
            root / resolution.NOTICE_PATH,
            "invalidation notice or resolution receipt",
        )
        status = notice.get("status")
        if status == resolution.ACTIVE_STATUS:
            resolution._validate_active_notice(notice)
        elif status == resolution.RESOLVED_STATUS:
            verified = resolution.verify_public_receipt(root, profile="full")
            if verified != notice:
                raise resolution.ResolutionError(
                    "resolved receipt changed while validating the review state"
                )
        else:
            raise resolution.ResolutionError(
                f"unsupported invalidation notice status: {status!r}"
            )
    except (resolution.ResolutionError, OSError, ValueError) as exc:
        raise ReviewBuildError(f"cannot validate the review notice state: {exc}") from exc
    return notice, notice_bytes


def _current_governed_binding(root: Path) -> dict[str, Any]:
    """Bind everything whose drift requires a new full review snapshot."""

    try:
        notice, notice_bytes = _validated_review_notice(root)
        fixed = dict(resolution._fixed_replacement_hashes(root))
        verifiers = dict(resolution._current_verifier_hashes(root))
    except (resolution.ResolutionError, OSError, ValueError) as exc:
        raise ReviewBuildError(f"cannot bind the active review state: {exc}") from exc

    paper_sources = _hash_inventory(
        root,
        tuple(public_policy.V2_PAPER_SOURCE_PATHS),
        "paper-number source",
    )
    profile = _hash_inventory(
        root,
        (PROFILE_JSON_RELATIVE, PROFILE_TEX_RELATIVE),
        "benchmark-profile interface",
    )
    binding = {
        # Historical key retained so existing sealed caches remain readable.
        "active_invalidation_notice_sha256": _sha256_bytes(notice_bytes),
        "invalidation_notice_status": notice["status"],
        "fixed_replacement_sha256": fixed,
        "paper_source_sha256": paper_sources,
        "benchmark_profile_sha256": profile,
        "resolution_verifier_sha256": verifiers,
    }
    try:
        if (root / resolution.NOTICE_PATH).read_bytes() != notice_bytes:
            raise ReviewBuildError(
                "invalidation notice changed while binding the review state"
            )
    except OSError as exc:
        raise ReviewBuildError(
            f"cannot recheck the invalidation notice while binding: {exc}"
        ) from exc
    return binding


def _generated_cache_paths(cache_root: Path) -> dict[str, Path]:
    return {
        resolution.PAPER_JSON_PATH: cache_root / resolution.PAPER_JSON_PATH,
        resolution.PAPER_TEX_PATH: cache_root / resolution.PAPER_TEX_PATH,
    }


def _manifest_path(cache_root: Path) -> Path:
    return cache_root / MANIFEST_NAME


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_cache_manifest(
    *,
    binding: Mapping[str, Any],
    preview: resolution.PaperPreview,
    cache_root: Path,
) -> dict[str, Any]:
    generated = _generated_cache_paths(cache_root)
    return {
        "schema_version": CACHE_SCHEMA,
        "status": CACHE_STATUS,
        "binding": dict(binding),
        "generated_sha256": {
            relative: _sha256_file(path) for relative, path in generated.items()
        },
        "number_key_count": preview.number_key_count,
        "number_keys_sha256": preview.number_keys_sha256,
        "number_values_sha256": preview.number_values_sha256,
    }


def _write_resolved_paper_snapshot(
    root: Path,
    *,
    preview_root: Path,
) -> resolution.PaperPreview:
    """Seal exact canonical number bytes after verifying the RESOLVED receipt."""

    try:
        receipt = resolution.verify_public_receipt(root, profile="full")
    except (resolution.ResolutionError, OSError, ValueError) as exc:
        raise ReviewBuildError(f"resolved receipt verification failed: {exc}") from exc
    if receipt.get("status") != resolution.RESOLVED_STATUS:
        raise ReviewBuildError("resolved review snapshot requires a RESOLVED receipt")

    canonical_paths = {
        relative: _require_regular_file(
            root / relative,
            f"resolved canonical paper output {relative}",
        )
        for relative in resolution.GENERATED_PATHS
    }
    captured = {
        relative: path.read_bytes() for relative, path in canonical_paths.items()
    }
    replacements = receipt.get("replacement_sha256")
    if not isinstance(replacements, dict):
        raise ReviewBuildError("resolved receipt lacks replacement hashes")
    for relative, content in captured.items():
        expected = replacements.get(relative)
        if not isinstance(expected, str) or _sha256_bytes(content) != expected:
            raise ReviewBuildError(
                f"resolved canonical paper output hash mismatch: {relative}"
            )

    paper, paper_bytes = _load_canonical_json(
        canonical_paths[resolution.PAPER_JSON_PATH],
        "resolved canonical paper-number JSON",
    )
    if paper_bytes != captured[resolution.PAPER_JSON_PATH]:
        raise ReviewBuildError("resolved paper-number JSON changed while being captured")
    sources = paper.get("sources")
    recorded_sources = receipt.get("resolution_source_sha256")
    if not isinstance(sources, dict) or sources != recorded_sources:
        raise ReviewBuildError(
            "resolved paper-number sources differ from the resolution receipt"
        )
    try:
        numbers = public_policy._validate_paper_numbers(paper.get("numbers"))
    except ValueError as exc:
        raise ReviewBuildError(
            f"resolved paper-number interface is invalid: {exc}"
        ) from exc

    try:
        targets = resolution._preview_targets(root, preview_root)
    except (resolution.ResolutionError, OSError, ValueError) as exc:
        raise ReviewBuildError(
            f"resolved review snapshot destination is unsafe: {exc}"
        ) from exc
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    for relative, target in targets.items():
        _write_atomic(target, captured[relative])
        if target.read_bytes() != captured[relative]:
            raise ReviewBuildError(
                f"resolved review snapshot copy changed bytes: {relative}"
            )

    return resolution.PaperPreview(
        root=root,
        preview_root=Path(preview_root).resolve(strict=True),
        generated_bytes=captured,
        source_sha256=dict(sources),
        number_key_count=len(numbers),
        number_keys_sha256=public_policy._paper_number_key_digest(numbers),
        number_values_sha256=public_policy._paper_number_value_digest(numbers),
    )


def _write_review_snapshot(
    root: Path,
    *,
    preview_root: Path,
    notice_status: str,
) -> resolution.PaperPreview:
    """Dispatch snapshot generation without weakening either state contract."""

    if notice_status == resolution.ACTIVE_STATUS:
        return resolution.write_paper_preview(root, preview_root=preview_root)
    if notice_status == resolution.RESOLVED_STATUS:
        return _write_resolved_paper_snapshot(root, preview_root=preview_root)
    raise ReviewBuildError(
        f"cannot refresh review numbers for unsupported notice status: {notice_status!r}"
    )


def refresh_review_cache(root: Path, cache_root: Path) -> ReviewInterface:
    """Seal a reusable cache from an ACTIVE preview or RESOLVED canonical bytes."""

    root = root.resolve(strict=True)
    cache_root = cache_root.resolve(strict=False)
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    if cache_root.exists() or cache_root.is_symlink():
        _require_real_directory(cache_root, "existing review cache")
    before = _current_governed_binding(root)
    promotion_root: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{cache_root.name}.refresh-", dir=cache_root.parent
        ) as temporary:
            staging_root = Path(temporary) / "sealed-cache"
            try:
                preview = _write_review_snapshot(
                    root,
                    preview_root=staging_root,
                    notice_status=str(before["invalidation_notice_status"]),
                )
            except (resolution.ResolutionError, OSError, ValueError, ImportError) as exc:
                raise ReviewBuildError(f"review-number refresh failed: {exc}") from exc
            after = _current_governed_binding(root)
            if before != after:
                raise ReviewBuildError(
                    "governed inputs changed around review preview generation; rerun after edits stop"
                )
            if dict(preview.source_sha256) != after["paper_source_sha256"]:
                raise ReviewBuildError(
                    "preview source map differs from the sealed paper-source binding"
                )

            manifest = _build_cache_manifest(
                binding=after,
                preview=preview,
                cache_root=staging_root,
            )
            _write_atomic(_manifest_path(staging_root), _canonical_json_bytes(manifest))
            staged = validate_review_cache(root, staging_root)
            if staged.manifest != manifest:
                raise ReviewBuildError("staged review cache differs from its sealed manifest")
            if _current_governed_binding(root) != after:
                raise ReviewBuildError("governed inputs changed while sealing the review cache")

            promotion_root = Path(
                tempfile.mkdtemp(prefix=f".{cache_root.name}.promote-", dir=cache_root.parent)
            )
            promotion_root.rmdir()
            os.replace(staging_root, promotion_root)

        # The temporary verifier tree has now cleaned up successfully.  Only
        # after that may the sealed cache replace the last-good cache.
        if _current_governed_binding(root) != after:
            raise ReviewBuildError("governed inputs changed before review-cache promotion")
        backup_root: Path | None = None
        promoted = False
        try:
            if cache_root.exists():
                backup_root = Path(
                    tempfile.mkdtemp(
                        prefix=f".{cache_root.name}.backup-", dir=cache_root.parent
                    )
                )
                backup_root.rmdir()
                os.replace(cache_root, backup_root)
            os.replace(promotion_root, cache_root)
            promoted = True
            final = validate_review_cache(root, cache_root)
            if final.manifest != manifest:
                raise ReviewBuildError("promoted review cache differs from its sealed manifest")
        except BaseException:
            promotion_completed = (
                promoted
                or (
                    cache_root.exists()
                    and promotion_root is not None
                    and not promotion_root.exists()
                )
            )
            if promotion_completed:
                os.replace(cache_root, promotion_root)
            if backup_root is not None and backup_root.exists():
                os.replace(backup_root, cache_root)
            raise
        else:
            if backup_root is not None:
                shutil.rmtree(backup_root, ignore_errors=True)
            return final
    finally:
        if promotion_root is not None and promotion_root.exists():
            shutil.rmtree(promotion_root, ignore_errors=True)


def validate_review_cache(
    root: Path,
    cache_root: Path,
    *,
    allow_stale_binding: bool = False,
) -> ReviewInterface:
    """Validate a sealed preview without rerunning scientific computations.

    ``allow_stale_binding`` relaxes only the comparison with the repository's
    current governed inputs.  The cache manifest, generated bytes, JSON/TeX
    projection, and number digests must still verify against the binding under
    which the snapshot was sealed.
    """

    root = root.resolve(strict=True)
    cache_root = cache_root.resolve(strict=True)
    _require_real_directory(cache_root, "review cache")
    manifest, _ = _load_canonical_json(_manifest_path(cache_root), "review cache manifest")
    expected_fields = {
        "schema_version",
        "status",
        "binding",
        "generated_sha256",
        "number_key_count",
        "number_keys_sha256",
        "number_values_sha256",
    }
    if set(manifest) != expected_fields:
        raise ReviewBuildError("review cache manifest has an unexpected field inventory")
    if manifest.get("schema_version") != CACHE_SCHEMA or manifest.get("status") != CACHE_STATUS:
        raise ReviewBuildError("review cache manifest has an unsupported schema or status")

    manifest_binding = manifest.get("binding")
    if not isinstance(manifest_binding, dict):
        raise ReviewBuildError("review cache manifest has an invalid governed binding")
    current_binding = _current_governed_binding(root)
    binding_current = manifest_binding == current_binding
    if not binding_current and not allow_stale_binding:
        raise ReviewBuildError(
            "review-number cache is stale because data, metrics, the hold, or a verifier changed"
        )

    generated = _generated_cache_paths(cache_root)
    observed_generated = {
        relative: _sha256_file(_require_regular_file(path, f"cached {relative}"))
        for relative, path in generated.items()
    }
    if manifest.get("generated_sha256") != observed_generated:
        raise ReviewBuildError("cached review-number JSON/TeX bytes changed after verification")

    json_path = generated[resolution.PAPER_JSON_PATH]
    tex_path = generated[resolution.PAPER_TEX_PATH]
    payload, json_bytes = _load_canonical_json(json_path, "cached paper-number JSON")
    try:
        numbers = public_policy._validate_paper_numbers(
            payload.get("numbers"),
            allow_unfrozen_inventory=True,
        )
    except ValueError as exc:
        raise ReviewBuildError(f"cached paper-number map is invalid: {exc}") from exc
    sources = payload.get("sources")
    sealed_sources = manifest_binding.get("paper_source_sha256")
    if not isinstance(sealed_sources, dict):
        raise ReviewBuildError("review cache manifest has an invalid paper-source binding")
    if not isinstance(sources, dict) or sources != sealed_sources:
        raise ReviewBuildError("cached paper-number source map is stale or malformed")
    if json_bytes != paper_numbers.render_json(numbers, sources).encode("utf-8"):
        raise ReviewBuildError("cached paper-number JSON is not the canonical rendered interface")
    try:
        public_policy._verify_paper_tex_interface(cache_root, sources, numbers)
    except ValueError as exc:
        raise ReviewBuildError(
            f"cached paper-number TeX is not the exact JSON projection: {exc}"
        ) from exc

    key_count = len(numbers)
    key_digest = public_policy._paper_number_key_digest(numbers)
    value_digest = public_policy._paper_number_value_digest(numbers)
    if manifest.get("number_key_count") != key_count:
        raise ReviewBuildError("cached paper-number key count differs from its receipt")
    if manifest.get("number_keys_sha256") != key_digest:
        raise ReviewBuildError("cached paper-number key digest differs from its receipt")
    if manifest.get("number_values_sha256") != value_digest:
        raise ReviewBuildError("cached paper-number value digest differs from its receipt")

    return ReviewInterface(
        cache_root=cache_root,
        json_path=json_path,
        tex_path=tex_path,
        manifest=manifest,
        binding_current=binding_current,
    )


def _canonical_guard_snapshot(root: Path) -> dict[str, str]:
    return _hash_inventory(root, CANONICAL_GUARD_PATHS, "canonical guard input")


def _verify_benchmark_profile(root: Path) -> None:
    try:
        benchmark_profile.verify_outputs(
            root / PROFILE_JSON_RELATIVE,
            root / PROFILE_TEX_RELATIVE,
            mode="repository",
        )
    except (benchmark_profile.ProfileError, OSError, ValueError) as exc:
        raise ReviewBuildError(f"benchmark-profile verification failed: {exc}") from exc


def _verify_contemporary_references(root: Path) -> None:
    """Verify the independent post-freeze reference interface before staging it."""

    try:
        contemporary_results.verify_outputs(
            root / CONTEMPORARY_JSON_RELATIVE,
            root / CONTEMPORARY_CSV_RELATIVE,
            root / CONTEMPORARY_TEX_RELATIVE,
            config_path=root / CONTEMPORARY_CONFIG_RELATIVE,
            profile="repository",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ReviewBuildError(
            f"contemporary-reference verification failed: {exc}"
        ) from exc


def _copy_paper_tree(root: Path, build_root: Path, interface: ReviewInterface) -> None:
    paper_root = _require_real_directory(root / "paper", "paper source directory")
    build_root.mkdir(parents=True, exist_ok=False)
    for name in PAPER_SOURCE_FILES:
        source = _require_regular_file(paper_root / name, f"paper source {name}")
        shutil.copy2(source, build_root / name)

    figures = _require_real_directory(paper_root / "figures", "paper figures directory")
    shutil.copytree(figures, build_root / "figures")
    generated = build_root / "generated"
    generated.mkdir()
    shutil.copy2(root / PROFILE_TEX_RELATIVE, generated / "v2_benchmark_profile.tex")
    shutil.copy2(interface.tex_path, generated / "v2_numbers.tex")
    contemporary_source = _require_regular_file(
        root / CONTEMPORARY_TEX_RELATIVE,
        "contemporary-reference TeX interface",
    )
    contemporary_copy = generated / "v2_contemporary_references.tex"
    shutil.copy2(contemporary_source, contemporary_copy)

    copied_numbers = generated / "v2_numbers.tex"
    if _sha256_file(copied_numbers) != _sha256_file(interface.tex_path):
        raise ReviewBuildError("review-number overlay changed while preparing the build tree")
    if _sha256_file(contemporary_copy) != _sha256_file(contemporary_source):
        raise ReviewBuildError(
            "contemporary-reference interface changed while preparing the build tree"
        )


def _find_tex_tool(name: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        try:
            candidate = explicit.expanduser().resolve(strict=False)
            if candidate.is_file():
                return candidate
        except OSError as exc:
            raise ReviewBuildError(
                f"cannot inspect configured {name} executable: {explicit}: {exc}"
            ) from exc
        raise ReviewBuildError(f"configured {name} executable does not exist: {candidate}")

    discovered = shutil.which(name)
    if discovered:
        return Path(discovered).resolve(strict=True)

    executable = f"{name}.exe" if os.name == "nt" else name
    candidates: list[Path] = []
    if os.name == "nt":
        for variable in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(variable)
            if not base:
                continue
            candidates.extend(
                (
                    Path(base) / "Programs/MiKTeX/miktex/bin/x64" / executable,
                    Path(base) / "MiKTeX/miktex/bin/x64" / executable,
                )
            )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve(strict=True)
        except OSError:
            continue
    raise ReviewBuildError(
        f"cannot find {name}; install a current TeX distribution or pass --{name} PATH"
    )


def _run_command(command: Sequence[str], *, cwd: Path) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise ReviewBuildError(f"cannot run {command[0]}: {exc}") from exc
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise ReviewBuildError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{tail}"
        )
    return result.stdout


def _compile_acm_pdf(build_root: Path, pdflatex: Path, bibtex: Path) -> Path:
    latex = (
        str(pdflatex),
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main-acm.tex",
    )
    _run_command(latex, cwd=build_root)
    _run_command((str(bibtex), "main-acm"), cwd=build_root)

    previous_auxiliary: dict[str, bytes | None] | None = None
    converged = False
    for pass_index in range(1, 6):
        _run_command(latex, cwd=build_root)
        auxiliary = {
            suffix: (build_root / f"main-acm{suffix}").read_bytes()
            if (build_root / f"main-acm{suffix}").is_file()
            else None
            for suffix in (".aux", ".out", ".toc")
        }
        log_path = _require_regular_file(build_root / "main-acm.log", "LaTeX log")
        log = log_path.read_text(encoding="utf-8", errors="replace")
        needs_rerun = any(marker in log.lower() for marker in LATEX_RERUN_MARKERS)
        if pass_index >= 2 and auxiliary == previous_auxiliary and not needs_rerun:
            converged = True
            break
        previous_auxiliary = auxiliary
    if not converged:
        raise ReviewBuildError(
            "LaTeX cross-references did not converge after five post-BibTeX passes"
        )

    log_path = _require_regular_file(build_root / "main-acm.log", "final LaTeX log")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    defects = [marker for marker in LATEX_ERROR_MARKERS if marker in log]
    defects.extend(
        marker for marker in LATEX_RERUN_MARKERS if marker in log.lower()
    )
    if defects:
        raise ReviewBuildError(
            "final LaTeX log failed review QA: " + ", ".join(defects)
        )
    pdf = _require_regular_file(build_root / "main-acm.pdf", "compiled review PDF")
    content = pdf.read_bytes()
    if len(content) < 1024 or not content.startswith(b"%PDF-") or b"%%EOF" not in content[-2048:]:
        raise ReviewBuildError("compiled review output is not a complete PDF")
    return pdf


def build_review_pdf(
    root: Path,
    interface: ReviewInterface,
    *,
    output: Path,
    pdflatex: Path,
    bibtex: Path,
    canonical_before: Mapping[str, str],
) -> Path:
    _verify_benchmark_profile(root)
    _verify_contemporary_references(root)
    initial_manifest = _canonical_json_bytes(interface.manifest)
    temporary_parent = root / "tmp" / "pdfs"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes: bytes | None = None
    with tempfile.TemporaryDirectory(prefix="paper-review-", dir=temporary_parent) as temporary:
        build_root = Path(temporary) / "paper"
        _copy_paper_tree(root, build_root, interface)
        pdf = _compile_acm_pdf(build_root, pdflatex, bibtex)
        final_interface = validate_review_cache(
            root,
            interface.cache_root,
            allow_stale_binding=not interface.binding_current,
        )
        if _canonical_json_bytes(final_interface.manifest) != initial_manifest:
            raise ReviewBuildError("review-number cache changed during LaTeX compilation")
        if _canonical_guard_snapshot(root) != dict(canonical_before):
            raise ReviewBuildError(
                "governed review inputs changed during review build"
            )
        pdf_bytes = pdf.read_bytes()

    if pdf_bytes is None:
        raise ReviewBuildError("compiled review PDF disappeared before publication")
    final_interface = validate_review_cache(
        root,
        interface.cache_root,
        allow_stale_binding=not interface.binding_current,
    )
    if _canonical_json_bytes(final_interface.manifest) != initial_manifest:
        raise ReviewBuildError("review-number cache changed before PDF publication")
    if _canonical_guard_snapshot(root) != dict(canonical_before):
        raise ReviewBuildError(
            "governed review inputs changed before PDF publication"
        )
    output_existed = output.exists() or output.is_symlink()
    previous_output = (
        _require_regular_file(output, "previous review PDF").read_bytes()
        if output_existed
        else None
    )
    try:
        _write_atomic(output, pdf_bytes)
        published_interface = validate_review_cache(
            root,
            interface.cache_root,
            allow_stale_binding=not interface.binding_current,
        )
        if _canonical_json_bytes(published_interface.manifest) != initial_manifest:
            raise ReviewBuildError("review-number cache changed during PDF publication")
        if _canonical_guard_snapshot(root) != dict(canonical_before):
            raise ReviewBuildError(
                "governed review inputs changed during PDF publication"
            )
        if output.read_bytes() != pdf_bytes:
            raise ReviewBuildError("published review PDF bytes differ from the compiled PDF")
    except BaseException as exc:
        try:
            if output_existed and previous_output is not None:
                _write_atomic(output, previous_output)
            elif output.exists() or output.is_symlink():
                output.unlink()
        except OSError as restore_exc:
            raise ReviewBuildError(
                "review state changed during PDF publication and the prior PDF could not be restored: "
                f"{restore_exc}"
            ) from exc
        if not isinstance(exc, Exception):
            raise
        raise ReviewBuildError(
            f"review state changed during PDF publication; the prior PDF was restored: {exc}"
        ) from exc
    return output.resolve(strict=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-numbers",
        action="store_true",
        help=(
            "refresh from a verified non-canonical preview while ACTIVE or from "
            "receipt-verified canonical bytes while RESOLVED"
        ),
    )
    parser.add_argument(
        "--require-current-numbers",
        action="store_true",
        help="fail unless the sealed number snapshot matches current governed inputs",
    )
    parser.add_argument("--pdflatex", type=Path, help="path to pdflatex executable")
    parser.add_argument("--bibtex", type=Path, help="path to bibtex executable")
    parser.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_RELATIVE)
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        output_root = _review_output_root(root)
        cache_root = args.cache_dir
        if not cache_root.is_absolute():
            cache_root = root / cache_root
        cache_root = _require_local_output_path(root, cache_root, "review cache")
        output = args.output
        if not output.is_absolute():
            output = root / output
        output = _require_local_output_path(root, output, "review PDF output")
        _require_disjoint_review_paths(cache_root, output)
        cache_control_root = output_root / "paper-review-cache"
        if cache_root == cache_control_root:
            raise ReviewBuildError(
                f"review cache must be below, not equal to, {cache_control_root}"
            )
        _require_disjoint_review_paths(cache_control_root, output)

        with _review_build_lock(output_root):
            canonical_before = _canonical_guard_snapshot(root)
            if args.refresh_numbers:
                print(
                    "Refreshing the verified paper-number cache for the current "
                    "invalidation state; this may take several minutes."
                )
                interface = refresh_review_cache(root, cache_root)
            else:
                try:
                    interface = validate_review_cache(
                        root,
                        cache_root,
                        allow_stale_binding=not args.require_current_numbers,
                    )
                except (ReviewBuildError, OSError) as exc:
                    raise ReviewBuildError(
                        f"no reusable sealed review-number cache: {exc}; run once with "
                        "--refresh-numbers to create or renew it"
                    ) from exc
                if interface.binding_current:
                    print("Reusing the current sealed paper-number snapshot.")
                else:
                    print(
                        "WARNING: number-audit inputs differ from the last sealed paper-number "
                        "snapshot; compiling with that intact snapshot. Use --refresh-numbers "
                        "or --require-current-numbers for a release audit.",
                        file=sys.stderr,
                    )

            pdflatex = _find_tex_tool("pdflatex", args.pdflatex)
            bibtex = _find_tex_tool("bibtex", args.bibtex)
            built = build_review_pdf(
                root,
                interface,
                output=output,
                pdflatex=pdflatex,
                bibtex=bibtex,
                canonical_before=canonical_before,
            )
        print(f"Built review-only PDF: {built}")
        print("Canonical paper-number source files were not modified.")
    except (ReviewBuildError, OSError, ValueError, ImportError) as exc:
        print(f"REVIEW BUILD REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
