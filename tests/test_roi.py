"""Adaptive ROI selection: does it drop the regions that are not skin?

Run:  python3 tests/test_roi.py

Synthetic faces with a KNOWN pulse and known bad regions, so the correct
selection is known in advance rather than judged by eye.

The defect this replaces: three fixed regions -- forehead and two cheeks --
assume bare skin. A beard fills the cheek polygons with hair, which has no
blood volume; glasses fill their upper half with reflection, which moves with
the head. Measured on a real recording, the three fixed regions disagreed by
47.7 BPM while reporting a confident fused number.

Selection is by BEHAVIOUR, never by colour. A colour-based skin test is
calibrated on some range of skin tones and fails outside it, which would hand
darker-skinned subjects a worse measurement by construction. Section 4 asserts
that property directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CONFIG
from signals.roi import AdaptiveROI, PATCHES

FPS = 30.0
RNG = np.random.default_rng(11)
failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


class FakeROI(AdaptiveROI):
    """Drives selection from per-patch signals, bypassing pixel extraction.

    The polygon-to-pixels path is exercised on real video elsewhere; what
    needs asserting here is the SELECTION logic, and feeding it known signals
    is the only way to know what the right answer was.
    """

    def feed(self, n, producer):
        for i in range(n):
            self.frames += 1
            for name in self.names:
                self.patches[name].update(producer(name, i))


def pulsing(bpm, amp=1.6, base=(120, 135, 165), noise=0.25, phase=0.0):
    """A cardiac-SHAPED waveform, not a sine.

    A heartbeat has a sharp upstroke and slow decay, which is what gives it a
    second harmonic. Testing with a pure sinusoid was testing the wrong
    signal: the harmonic check correctly identifies a sinusoid as
    non-cardiac, so the old fixtures described flicker rather than a pulse.
    """
    def f(name, i):
        ph = ((bpm / 60.0) * (i / FPS) + phase / (2 * np.pi)) % 1.0
        a = amp * (np.exp(-4.0 * ph) * 2.0 - 0.6)
        return np.array([base[0] + a * 0.45 + RNG.normal(0, noise),
                         base[1] + a + RNG.normal(0, noise),
                         base[2] + a * 0.5 + RNG.normal(0, noise)])
    return f


print("1. A clean face: every patch pulses at the same rate")
r = FakeROI(fps=FPS)
sig = pulsing(72)
r.feed(600, lambda n, i: sig(n, i))
out = r.estimate()
check("recovers the true rate", out["bpm"] is not None
      and abs(out["bpm"] - 72) < 3, f"{out.get('bpm')} vs 72")
check("keeps several regions", out.get("n_regions", 0) >= 3,
      f"{out.get('n_regions')} regions")
check("reports it was cross-checked", out.get("cross_checked") is True)
check("spread is small when regions agree", out.get("roi_spread_bpm", 99) < 6,
      f"spread {out.get('roi_spread_bpm')}")

print("\n2. A beard: the malar patches carry no pulse at all")
r = FakeROI(fps=FPS)
good, dead = pulsing(72), pulsing(72, amp=0.0, noise=0.9)
r.feed(600, lambda n, i: (dead if n.startswith("malar") else good)(n, i))
out = r.estimate()
sel = set(out.get("regions") or {})
check("both malar patches are rejected",
      not (sel & {"malar_l", "malar_r"}), f"selected: {sorted(sel)}")
check("the rate is still recovered from what remains",
      out["bpm"] is not None and abs(out["bpm"] - 72) < 3, f"{out.get('bpm')}")
for m in ("malar_l", "malar_r"):
    if m in (out.get("rejected") or {}):
        print(f"          {m}: {out['rejected'][m][:56]}")

print("\n3. Glasses: a reflection swings one patch's brightness")
r = FakeROI(fps=FPS)
good = pulsing(72)
def glare(name, i):
    v = good(name, i)
    # A reflection crossing the patch as the head turns: a slow, large
    # brightness excursion unrelated to the cardiac band.
    return v + 26 * np.sin(2 * np.pi * 0.11 * i / FPS)
r.feed(600, lambda n, i: (glare if n == "malar_r" else good)(n, i))
out = r.estimate()
check("the glared patch is dropped", "malar_r" not in (out.get("regions") or {}),
      (out.get("rejected") or {}).get("malar_r", "")[:58])
check("the rate survives", out["bpm"] is not None and abs(out["bpm"] - 72) < 3,
      f"{out.get('bpm')}")

print("\n4. Unsteady room light must not reject the whole face")
# The bug this catches: an ABSOLUTE brightness threshold discarded seven of
# nine patches on a real recording, including three parts of one flat
# forehead, because the room light was varying. That is a fact about the room,
# not about any region.
r = FakeROI(fps=FPS)
good = pulsing(72)
def flicker(name, i):
    return good(name, i) + 22 * np.sin(2 * np.pi * 0.09 * i / FPS)
r.feed(600, flicker)
out = r.estimate()
check("global brightness variation keeps the face usable",
      out.get("n_regions", 0) >= 3, f"{out.get('n_regions')} regions survived")

print("\n5. A patch tracking a different frequency is outvoted")
r = FakeROI(fps=FPS)
good, rogue = pulsing(72), pulsing(140)
r.feed(600, lambda n, i: (rogue if n == "forehead_r" else good)(n, i))
out = r.estimate()
check("the disagreeing patch is dropped",
      "forehead_r" not in (out.get("regions") or {}),
      (out.get("rejected") or {}).get("forehead_r", "")[:52])
check("the consensus rate is unaffected",
      out["bpm"] is not None and abs(out["bpm"] - 72) < 3, f"{out.get('bpm')}")

print("\n6. Refusals, measured rather than asserted")
# Selection cannot be perfect against pure noise and it is dishonest to write
# a test that pretends otherwise. With nine patches drawing random rates over
# a 138 BPM band, some subset agreeing is ordinary -- before the guards were
# added this reported a rate on 27% of pure-noise trials. What the guards buy
# is measurable, so it is measured: an absolute assertion here would just be
# tuned until it passed.
TRIALS = 40


def noise_trial(seed):
    g = np.random.default_rng(seed)
    r = FakeROI(fps=FPS)
    r.feed(600, lambda n, i: np.array([120 + g.normal(0, 1.2),
                                       135 + g.normal(0, 1.2),
                                       165 + g.normal(0, 1.2)]))
    return r.estimate()


hits = [o for o in (noise_trial(s) for s in range(TRIALS))
        if o.get("bpm") is not None]
rate = len(hits) / TRIALS
check("pure noise rarely produces a number", rate <= 0.10,
      f"{len(hits)}/{TRIALS} = {rate:.0%} (was 27% before the guards)")


def real_trial(seed, amp=1.6):
    g = np.random.default_rng(seed)
    r = FakeROI(fps=FPS)

    def sig(name, i):
        a = amp * np.sin(2 * np.pi * (72 / 60.0) * i / FPS)
        return np.array([120 + a * .45 + g.normal(0, .25),
                         135 + a + g.normal(0, .25),
                         165 + a * .5 + g.normal(0, .25)])

    r.feed(600, sig)
    return r.estimate()


# Rejecting noise is only worth anything if real signal still gets through.
strong = [o for o in (real_trial(s) for s in range(12))
          if o.get("bpm") is not None]
check("a real signal is never refused", len(strong) == 12, f"{len(strong)}/12")
check("and stays accurate", strong and max(abs(o["bpm"] - 72) for o in strong) < 2,
      f"worst error {max(abs(o['bpm'] - 72) for o in strong):.2f} BPM" if strong else "")

weak = [o for o in (real_trial(s, amp=0.7) for s in range(12))
        if o.get("bpm") is not None]
check("a weak signal survives too", len(weak) >= 10, f"{len(weak)}/12 at amp 0.7")

r = FakeROI(fps=FPS)
r.feed(30, pulsing(72))
out = r.estimate()
check("before warm-up it says so rather than guessing",
      out["bpm"] is None and out["status"] == "warming up",
      f"{out.get('warmup_remaining_s')}s remaining")

print("\n7. Selection never reads absolute colour")
# Same signal, very different skin luminance. If selection depended on colour
# the two would select differently -- which is how a skin-tone bias gets in.
sel_by_tone = {}
for tone, base in (("lighter", (205, 175, 165)), ("darker", (78, 58, 52))):
    # The SAME noise realisation for both, so luminance is the only variable.
    # Two independent draws would compare noise against noise and say nothing
    # about colour sensitivity.
    g = np.random.default_rng(7)
    r = FakeROI(fps=FPS)

    def sig(name, i, base=base, g=g):
        a = 1.6 * np.sin(2 * np.pi * (72 / 60.0) * i / FPS)
        e = g.normal(0, 0.25, 3)
        return np.array([base[0] + a * .45 + e[0],
                         base[1] + a + e[1],
                         base[2] + a * .5 + e[2]])

    r.feed(600, sig)
    o = r.estimate()
    sel_by_tone[tone] = (set(o.get("regions") or {}), o.get("bpm"))
    print(f"          {tone:<8} {len(sel_by_tone[tone][0])} regions, "
          f"{o.get('bpm') and round(o['bpm'], 2)} bpm")
a, b = sel_by_tone["lighter"], sel_by_tone["darker"]
check("the same signal selects the same regions at any luminance",
      a[0] == b[0], f"{sorted(a[0])} vs {sorted(b[0])}")
check("and recovers the same rate",
      a[1] and b[1] and abs(a[1] - b[1]) < 2, f"{a[1]:.2f} vs {b[1]:.2f}")

print("\n8. Patch definitions are sane")
check("no landmark index outside the 478-point mesh",
      max(i for idx in PATCHES.values() for i in idx) < 478)
check("more candidates than the three fixed regions it replaces",
      len(PATCHES) > 3, f"{len(PATCHES)} patches")

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — adaptive ROI selects on behaviour, not colour.")
