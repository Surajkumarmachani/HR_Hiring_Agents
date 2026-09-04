#!/usr/bin/env bash
# Local equivalent of .github/workflows/ci.yml. Run before pushing.
set -euo pipefail
cd "$(dirname "$0")"
echo "== model weights =="        && python3 fetch_models.py --verify
echo "== DSP validation =="       && python3 tests/test_rppg.py && python3 tests/test_face.py
echo "== property tests =="       && python3 tests/test_properties.py
echo "== parameter ranges =="     && python3 tests/test_parameter_ranges.py
echo "== audio prosody =="        && python3 tests/test_audio.py
echo "== interview engine =="     && python3 tests/test_interview.py
echo "== linguistic content ==" && python3 tests/test_text.py
echo "== transcript lines =="   && python3 tests/test_transcript_lines.py
echo "== adaptive roi =="       && python3 tests/test_roi.py
echo "== face identity =="     && python3 tests/test_identity.py
echo "== subject record =="    && python3 tests/test_record.py
echo "== linkedin import =="   && python3 tests/test_linkedin.py
echo "== interviewer camera ==" && python3 tests/test_presence.py
echo "== capture gaps =="     && python3 tests/test_capture_gap.py
echo "== generated probes ==" && python3 tests/test_generate.py
echo "== consent enforcement ==" && python3 tests/test_consent_enforcement.py
echo "== accuracy apparatus ==" && python3 tests/test_rppg_tuning.py
echo "== answer cycle ==" && python3 tests/test_answer_cycle.py
echo "== end-to-end selftest ==" && python3 run_live.py --selftest
echo "== dependency pins =="     && python3 -c "
import mediapipe, sys
major = int(mediapipe.__version__.split('.')[0])
print('mediapipe', mediapipe.__version__)
sys.exit(0 if major < 1 else 'mediapipe 1.x reintroduces the macOS Metal abort')
"
echo
echo "CI PASS"
