from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import temporal_backtest as TB  # noqa: E402
import universe as U  # noqa: E402
from baci_filtered_cache import (  # noqa: E402
    BaciFilteredCache,
    CACHE_COLUMNS,
    CacheValidationError,
    MANIFEST_NAME,
    REQUIRED_YEARS,
    build_cache,
    sha256_file,
)


def _write_synthetic_baci(path: Path) -> None:
    country_codes = "country_code,country_iso3\n1,AAA\n2,BBB\n3,CCC\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("country_codes_V202401b.csv", country_codes)
        for year in REQUIRED_YEARS:
            rows = ["i,j,k,v,q"]
            # One audited sheep row plus a deliberately unregistered code.  The
            # varying values exercise fixed-calendar aggregation and float
            # round-tripping through deterministic gzip CSV.
            rows.append(f"1,2,010410,{500.25 + (year % 3)},1")
            rows.append(f"2,3,999999,{900 + year % 7},1")
            archive.writestr(
                f"BACI_HS92_Y{year}_V202401b.csv", "\n".join(rows) + "\n"
            )


class BaciFilteredCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.temp = Path(self._temp.name)
        self.raw = self.temp / "synthetic-baci.zip"
        _write_synthetic_baci(self.raw)

        # Use private copies so the stale-registry test never edits the project.
        self.chains = self.temp / "chains"
        self.chains.mkdir()
        for source in (ROOT / "chains").glob("*.json"):
            shutil.copy2(source, self.chains / source.name)
        self.audit = self.temp / "registry_audit.json"
        self.evidence = self.temp / "registry_evidence.json"
        shutil.copy2(ROOT / "docs" / "registry_audit.json", self.audit)
        shutil.copy2(
            ROOT / "chains" / "evidence" / "registry_evidence.json", self.evidence
        )
        self.cache = self.temp / "private" / "baci-filtered"
        build_cache(
            self.raw,
            self.cache,
            chains_dir=self.chains,
            audit_path=self.audit,
            evidence_path=self.evidence,
            chunk_rows=1,
        )

    def tearDown(self) -> None:
        self._temp.cleanup()
        U.set_active_chain("sheep")

    def _reader(self, requested_years=(2008, 2009, 2010, 2011, 2012)):
        return BaciFilteredCache(
            self.cache,
            requested_years=requested_years,
            chains_dir=self.chains,
            audit_path=self.audit,
            evidence_path=self.evidence,
        )

    def test_direct_and_cache_window_aggregation_are_equivalent(self) -> None:
        years = [2008, 2009, 2010, 2011, 2012]
        iso = {1: "AAA", 2: "BBB", 3: "CCC"}
        U.set_active_chain("sheep")
        with zipfile.ZipFile(self.raw) as source_zip:
            direct_stage, direct_hs6 = TB.load_window(source_zip, iso, years)
            reader = self._reader(years)
            country_payload = reader.country_codes_bytes(
                source_zip, archive_path=self.raw
            )
            self.assertIn(b"AAA", country_payload)
        cached_stage, cached_hs6 = TB.load_window(reader, iso, years)
        pd.testing.assert_frame_equal(
            direct_stage.sort_values(["i_iso", "j_iso", "stage"]).reset_index(drop=True),
            cached_stage.sort_values(["i_iso", "j_iso", "stage"]).reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            direct_hs6.sort_values(["i_iso", "j_iso", "k"]).reset_index(drop=True),
            cached_hs6.sort_values(["i_iso", "j_iso", "k"]).reset_index(drop=True),
            check_dtype=False,
        )
        raw_rows = reader.read_year(2008)
        self.assertEqual(list(raw_rows.columns), list(CACHE_COLUMNS))
        self.assertEqual(set(raw_rows.k.astype(str)), {"010410"})

    def test_manifest_binds_strong_source_and_contains_no_private_paths(self) -> None:
        manifest_path = self.cache / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["archive_sha256"], sha256_file(self.raw))
        self.assertEqual(manifest["years"], list(REQUIRED_YEARS))
        self.assertEqual(len(manifest["registry"]["chains"]), 6)
        self.assertEqual(len(manifest["files"]), len(REQUIRED_YEARS))
        rendered = manifest_path.read_text(encoding="utf-8")
        self.assertNotIn(str(self.temp), rendered)
        for entry in manifest["files"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertEqual(entry["year"], int(Path(entry["path"]).stem.split("_")[-1].split(".")[0]))

    def test_repeated_build_is_byte_reproducible(self) -> None:
        second = self.temp / "private" / "baci-filtered-second"
        build_cache(
            self.raw,
            second,
            chains_dir=self.chains,
            audit_path=self.audit,
            evidence_path=self.evidence,
            chunk_rows=2,
        )
        first_files = sorted(
            path.relative_to(self.cache).as_posix()
            for path in self.cache.rglob("*")
            if path.is_file()
        )
        second_files = sorted(
            path.relative_to(second).as_posix()
            for path in second.rglob("*")
            if path.is_file()
        )
        self.assertEqual(first_files, second_files)
        for relative in first_files:
            with self.subTest(path=relative):
                self.assertEqual(
                    (self.cache / relative).read_bytes(), (second / relative).read_bytes()
                )

    def test_tampered_file_is_rejected(self) -> None:
        manifest = json.loads((self.cache / MANIFEST_NAME).read_text(encoding="utf-8"))
        target = self.cache / manifest["files"][0]["path"]
        target.write_bytes(target.read_bytes() + b"tamper")
        with self.assertRaisesRegex(CacheValidationError, "size mismatch|SHA-256 mismatch"):
            self._reader()

    def test_stale_registry_is_rejected(self) -> None:
        target = self.chains / "sheep.json"
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(CacheValidationError, "stale"):
            self._reader()

    def test_unsupported_included_stage_fit_is_rejected(self) -> None:
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        included = next(
            decision
            for decision in evidence["chains"]["sheep"]["decisions"]
            if decision["decision"] == "include"
        )
        included["stage_fit"]["status"] = "unsupported"
        self.evidence.write_text(json.dumps(evidence), encoding="utf-8")
        with self.assertRaisesRegex(CacheValidationError, "supported stage_fit gate"):
            self._reader()

    def test_missing_year_is_rejected(self) -> None:
        manifest_path = self.cache / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        removed = manifest["files"].pop()
        (self.cache / removed["path"]).unlink()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(CacheValidationError, "exactly one file per year"):
            self._reader()

    def test_extra_code_is_rejected_even_with_updated_file_hash(self) -> None:
        manifest_path = self.cache / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(item for item in manifest["files"] if item["year"] == 2008)
        target = self.cache / entry["path"]
        frame = pd.read_csv(io.BytesIO(gzip.decompress(target.read_bytes())), dtype={"k": str})
        frame.loc[len(frame)] = [1, 2, "999999", 2008, 1.0]
        payload = gzip.compress(frame.to_csv(index=False, lineterminator="\n").encode(), mtime=0)
        target.write_bytes(payload)
        entry["rows"] = len(frame)
        entry["bytes"] = len(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["observed_hs6_codes"] = sorted(set(frame.k.astype(str)))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(CacheValidationError, "outside the audited union"):
            self._reader()

    def test_public_location_is_rejected(self) -> None:
        with self.assertRaisesRegex(CacheValidationError, "private inputs"):
            BaciFilteredCache(
                Path(ROOT.anchor) / "upgrade-bench-public-cache-test",
                chains_dir=self.chains,
                audit_path=self.audit,
                evidence_path=self.evidence,
            )


if __name__ == "__main__":
    unittest.main()
