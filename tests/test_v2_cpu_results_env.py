from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "requirements" / "verify_v2_cpu_results_env.py"
SPEC = importlib.util.spec_from_file_location("verify_v2_cpu_results_env", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class V2CPUResultsEnvironmentTest(unittest.TestCase):
    def test_lock_is_minimal_and_records_result_host(self) -> None:
        metadata, pins = MODULE.parse_lock()
        self.assertEqual(metadata["python"], "3.12.13")
        self.assertEqual(metadata["implementation"], "CPython")
        self.assertEqual(metadata["platform"], "Windows-11-10.0.26200-SP0")
        self.assertEqual(
            set(pins),
            {
                "joblib",
                "numpy",
                "pandas",
                "python-dateutil",
                "scikit-learn",
                "scipy",
                "threadpoolctl",
                "tzdata",
            },
        )

    def test_lock_matches_recorded_runtime_and_stale_source_is_gated(self) -> None:
        metadata, pins = MODULE.parse_lock()
        result = json.loads(MODULE.RESULT.read_text(encoding="utf-8"))
        runtime = result["runtime"]
        self.assertEqual(runtime["python"], metadata["python"])
        self.assertEqual(runtime["platform"], metadata["platform"])
        self.assertEqual(runtime["numpy"], pins["numpy"])
        self.assertEqual(runtime["pandas"], pins["pandas"])
        self.assertEqual(runtime["scikit_learn"], pins["scikit-learn"])
        current_hash = MODULE.sha256(MODULE.SCRIPT)
        if runtime["script_sha256"] != current_hash:
            invalidation = json.loads(
                (ROOT / "results_v2" / "metrics" / "INVALIDATED.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(invalidation["status"], "INVALIDATED_REGISTRY_AUDIT")
            self.assertIn("rolling_cpu_baselines.json", invalidation["scope"])


if __name__ == "__main__":
    unittest.main()
