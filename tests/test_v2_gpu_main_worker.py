import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "jobs" / "v2_gpu_main_worker.sh"


class V2GpuMainWorkerTest(unittest.TestCase):
    def test_worker_has_two_gates_and_no_marker_shortcut(self):
        worker = WORKER.read_text(encoding="utf-8")
        self.assertIn('MAIN_EVALUATION_CONFIRM=FROZEN_36_HASH_VERIFIED', worker)
        self.assertIn('tools/step3_sync_manifest.py --verify', worker)
        self.assertIn('CMD=("$BASE_PYTHON" src/v2_gpu_rolling.py evaluate-chain', worker)
        self.assertIn('"${CMD[@]}" --dry-run', worker)
        self.assertIn('"${CMD[@]}"', worker)
        self.assertIn('CLAIM_DIR="$CLAIM_ROOT/${CHAIN}_${FAMILY}.lock"', worker)
        self.assertIn('if ! mkdir "$CLAIM_DIR"', worker)
        self.assertNotIn('--overwrite', worker)
        # The worker may document the marker, but only v2_gpu_rolling.py may
        # write it.  Guard against shell-side touch/mv/redirection shortcuts.
        self.assertNotRegex(worker, r'(touch|mv|>)\s+[^\n]*MAIN_EVALUATION_STARTED')

    def test_worker_pins_canonical_paths_and_all_gate_arguments(self):
        worker = WORKER.read_text(encoding="utf-8")
        self.assertIn('OUTPUT_ROOT="$RUN_ROOT/results_v2/gpu_rolling"', worker)
        self.assertIn('MANIFEST="$OUTPUT_ROOT/frozen_manifest.json"', worker)
        self.assertIn('PILOT_INVALIDATED.json', worker)
        self.assertIn('RUN_CONFIG="$RUN_ROOT/configs/v2_gpu_rolling.json"', worker)
        self.assertIn('--candidate-root "$CANDIDATE_ROOT" --output-root "$OUTPUT_ROOT"', worker)
        self.assertIn('--run-config "$RUN_CONFIG" --manifest "$MANIFEST"', worker)
        self.assertIn('--seeds "$SEEDS" --require-cuda', worker)
        self.assertIn('candidates_${CHAIN}.csv', worker)
        self.assertIn('candidates_firsttime_${CHAIN}.csv', worker)

    def test_worker_restricts_one_canonical_chain_and_family(self):
        worker = WORKER.read_text(encoding="utf-8")
        self.assertIn('kge|nbfnet)', worker)
        self.assertIn('sheep|cotton|aluminium|nickel|cocoa|oilseed-soy)', worker)
        self.assertIn('if [[ ! "$PHYSICAL_GPU" =~ ^[0-9]+$ ]]', worker)

    def test_bash_syntax_when_a_native_bash_is_available(self):
        bash = shutil.which("bash")
        if bash is None or "system32" in bash.lower():
            self.skipTest("native bash is unavailable on this host")
        completed = subprocess.run(
            [bash, "-n", str(WORKER)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
