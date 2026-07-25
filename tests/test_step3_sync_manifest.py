from __future__ import annotations

import os
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from tools import step3_sync_manifest as sync


FULL_PAYLOAD_TESTS_ENABLED = os.environ.get("UPGRADE_BENCH_FULL_PAYLOAD_TESTS") == "1"
requires_full_payload = unittest.skipUnless(
    FULL_PAYLOAD_TESTS_ENABLED,
    "requires the externally mounted full data payload",
)


class Step3SyncManifestTests(unittest.TestCase):
    def test_manifest_pins_exact_main_and_history_candidate_inputs(self) -> None:
        relative = {
            path.relative_to(ROOT).as_posix() for path in sync.expected_files()
        }
        expected = set()
        for chain in sync.ROOT.joinpath("chains").glob("*.json"):
            chain_id = chain.stem
            for prefix in ("candidates", "candidates_firsttime"):
                expected.add(f"data/processed_v2/{prefix}_{chain_id}.csv")
                expected.add(f"data/processed_v2/{prefix}_{chain_id}_fold2.csv")
        candidates = {name for name in relative if name.startswith("data/processed_v2/")}
        self.assertEqual(candidates, expected)
        self.assertEqual(len(candidates), 24)
        self.assertIn("src/baci_filtered_cache.py", relative)

    @requires_full_payload
    def test_render_is_canonical_and_contains_no_absolute_paths(self) -> None:
        rendered = sync.render()
        lines = rendered.splitlines()
        self.assertEqual(lines, sorted(lines, key=lambda line: line.split("  ", 1)[1]))
        for line in lines:
            self.assertRegex(line, r"^[0-9a-f]{64}  [^\\\r\n]+$")
            logical = line.split("  ", 1)[1]
            self.assertFalse(Path(logical).is_absolute())
            self.assertNotIn("..", Path(logical).parts)


if __name__ == "__main__":
    unittest.main()
