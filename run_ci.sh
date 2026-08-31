#!/usr/bin/env bash
# Local equivalent of .github/workflows/ci.yml. Run before pushing.
set -euo pipefail
cd "$(dirname "$0")"
echo "== model weights =="        && python3 fetch_models.py --verify
echo "== DSP validation =="       && python3 tests/test_rppg.py && python3 tests/test_face.py
echo "== end-to-end selftest ==" && python3 run_live.py --selftest
echo "== dependency pins =="     && python3 -c "
import mediapipe, sys
major = int(mediapipe.__version__.split('.')[0])
print('mediapipe', mediapipe.__version__)
sys.exit(0 if major < 1 else 'mediapipe 1.x reintroduces the macOS Metal abort')
"
echo
echo "CI PASS"
