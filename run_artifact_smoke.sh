#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/env.sh" ]]; then
  # Optional local interpreter override; the smoke itself needs no raw/GPU paths.
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi

cd "$ROOT"
"${PYTHON:-python}" tools/release_smoke.py "$@"
