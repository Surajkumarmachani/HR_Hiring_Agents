"""Adaptive ROI selection for rPPG.

THE PROBLEM WITH THREE FIXED REGIONS
------------------------------------
v1.1 averaged a fixed forehead and two fixed cheek polygons. Those cheek
polygons assume bare skin. On a face with a beard they sample hair, which has
no blood volume; behind glasses they sample lens reflection, which moves with
the head. Neither contributes signal, both contribute structured noise, and
the quality-weighted fusion then averaged them into the answer -- measured
live at 44 BPM of disagreement between three regions of one face.

Beards and glasses are not edge cases. Fixed regions cannot be right for
everyone, so the region set has to be chosen per person.

WHY SELECTION IS BY BEHAVIOUR, NOT BY COLOUR
--------------------------------------------
The obvious approach is a skin classifier: test each pixel's colour and keep
the skin-like ones. Do not do this. A colour-based skin test is calibrated on
some distribution of skin tones and fails on the rest, so it would drop more
of a darker-skinned subject's face than a lighter-skinned one -- silently
handing them a worse measurement. That is the exact failure WP8b exists to
catch, built in on purpose.

Instead a patch earns its place by what it DOES over time: does it produce a
periodic signal in the cardiac band, and is its brightness stable? Beard has
no pulsatility whatever its colour. A lens reflection swings in brightness
when the head moves. Both are rejected on behaviour, and the test is identical
for every skin tone because it never looks at absolute colour at all.

KEEPING THE HONESTY CHECK
-------------------------
Dropping to a single region would lose `roi_spread`, which is the only signal
that catches a confident wrong answer: SQI says "this is periodic", not "this
is a heartbeat", and flicker, AGC and head-bob are all periodic. So the patch
set is deliberately larger than three, and agreement is measured between
whichever patches survive. Fewer than two survivors is reported as such,
because an estimate with no cross-check is a different kind of number.
"""

from collections import deque

import numpy as np

from config import CONFIG
from signals.rppg import POSEstimator, skin_mask_rgb_mean

# Candidate patches, as MediaPipe face-mesh landmark polygons.
#
# Chosen to spread across regions that fail INDEPENDENTLY: a beard takes the
# lower face, glasses take the orbital rim, a fringe takes the upper forehead,
# and a side-lit room takes one lateral side. Something survives most
# combinations. The lower cheek and jaw of v1.1 are gone entirely -- that is
# where a beard always is, so it was never a sensible fixed choice.
PATCHES = {
    # Forehead, split three ways: a fringe usually takes one side, not all.
    "forehead_l":  [67, 69, 108, 109, 104, 105],
    "forehead_c":  [109, 108, 151, 337, 338, 10],
    "forehead_r":  [297, 299, 337, 338, 333, 334],
    # Glabella, between the brows. Small, but bare on almost everyone.
    "glabella":    [9, 107, 66, 105, 63, 8, 293, 296, 336],
    # Malar, high on the cheekbone: below the eye, lateral to the nose, and
    # above where a beard reaches. Where glasses reflect, this fails and the
    # selector drops it.
    "malar_l":     [117, 118, 119, 100, 47, 114, 121],
    "malar_r":     [346, 347, 348, 329, 277, 343, 350],
    # Temples, outside the lens on most frames but often under hair.
    "temple_l":    [21, 54, 103, 67, 109],
    "temple_r":    [251, 284, 332, 297, 338],
    # Nasal dorsum: bare on everyone, but a spectacle bridge sits right on it.
    #
    # WAS [6, 197, 195, 5, 4], WHICH COULD NEVER WORK. Every one of those five
    # landmarks lies on the face's MIDLINE -- 6, 197, 195, 5 and 4 run straight
    # down the centre of the nose -- so the polygon was a vertical line with a
    # bounding box zero pixels wide. fillConvexPoly filled 49 pixels of line
    # against min_roi_pixels of 200, skin_mask_rgb_mean returned None, and this
    # patch reported 0.0% coverage on every frame of all seven recordings on
    # disk. It was not a marginal region; it was geometrically incapable of
    # producing a sample, and the failure was invisible because a rejected
    # patch is a normal event.
    #
    # Replaced with a quadrilateral that has area: 193 and 417 are the left and
    # right sides of the sellion, 196 and 419 the sides lower down the dorsum,
    # ordered so the quad is convex. 679 px on a 640x480 frame. Kept above the
    # nostrils deliberately -- extending down to the nose wings would gather
    # 1046 px and most of the extra would be nostril shadow.
    "nose_bridge": [193, 417, 419, 196],
}


class PatchState:
    """One candidate region: its estimator and the evidence about it."""

    def __init__(self, name, fps, cfg):
        self.name = name
        self.est = POSEstimator(fps=fps, cfg=cfg)
        self.frames = 0
        self.valid = 0
        self.brightness = deque(maxlen=int(fps * 10))
        self.bpm = None
        self.sqi = 0.0
        self.harmonic = None
        self.reason = None

    def update(self, rgb_mean, t=None):
        self.frames += 1
        if rgb_mean is None:
            return
        self.valid += 1
        self.est.update(rgb_mean, t)
        self.brightness.append(float(np.mean(rgb_mean)))

    @property
    def coverage(self):
        return self.valid / self.frames if self.frames else 0.0

    @property
    def brightness_cv(self):
        """Relative brightness swing. A lens reflection entering and leaving
        the patch as the head turns shows up here and nowhere else."""
        if len(self.brightness) < 10:
            return 0.0
        b = np.asarray(self.brightness)
        m = float(b.mean())
        return float(b.std() / m) if m > 1e-6 else 0.0

    def score(self):
        bpm, sqi, _ = self.est.estimate()
        self.bpm, self.sqi = bpm, (sqi or 0.0)
        self.harmonic = self.est.last_harmonic_ratio
        return self.bpm, self.sqi


class PulseTracker:
    """Holds a rate over time and refuses implausible jumps.

    The estimator treats every window independently, so nothing stopped it
    reporting 72 BPM and then 50 BPM from the same seated subject minutes
    apart -- both with excellent SQI, tight IQR and three regions agreeing.
    Internal consistency cannot catch that: three regions agreeing means they
    see the same thing, not that the thing is a heartbeat.

    Physiology can. A resting rate changes slowly, so a large step is evidence
    the estimator re-locked onto something else. A new value has to persist
    across several windows before it displaces the tracked one -- which lets a
    genuine change through within a few seconds while rejecting a single
    window that wandered.
    """

    def __init__(self, cfg=None):
        self.c = (cfg or CONFIG).rppg
        self.bpm = None
        self.last_t = None
        self.pending = None
        self.pending_n = 0

    def update(self, bpm, t):
        """Returns (accepted_bpm, status)."""
        if bpm is None:
            return self.bpm, "no estimate this window"
        if self.bpm is None:
            self.bpm, self.last_t = bpm, t
            return self.bpm, "acquiring"

        dt = max(1e-3, t - (self.last_t if self.last_t is not None else t))
        allowed = max(self.c.pulse_max_change_bpm_per_s * dt, 2.0)
        if abs(bpm - self.bpm) <= allowed:
            self.bpm, self.last_t = bpm, t
            self.pending, self.pending_n = None, 0
            return self.bpm, "tracking"

        # Out of range. Accept only if it keeps saying the same thing.
        if self.pending is not None and abs(bpm - self.pending) <= allowed:
            self.pending_n += 1
        else:
            self.pending, self.pending_n = bpm, 1
        self.pending = bpm
        if self.pending_n >= self.c.pulse_relock_windows:
            self.bpm, self.last_t = bpm, t
            self.pending, self.pending_n = None, 0
            return self.bpm, "re-locked"
        return self.bpm, (f"holding {self.bpm:.0f}; {bpm:.0f} seen "
                          f"{self.pending_n}/{self.c.pulse_relock_windows}")

    def reset(self):
        self.bpm = self.last_t = self.pending = None
        self.pending_n = 0


class AdaptiveROI:
    """Chooses which face regions to trust, per subject, from their behaviour."""

    def __init__(self, fps=30.0, cfg=None, patches=None):
        self.cfg = cfg or CONFIG
        self.c = self.cfg.rppg
        self.fps = fps
        self.names = list(patches or PATCHES)
        self.patches = {n: PatchState(n, fps, self.cfg) for n in self.names}
        self.selected = []
        self.rejected = {}
        self.frames = 0
        self.tracker = PulseTracker(self.cfg)

    def polygons(self, landmarks_px):
        """Landmark polygons in pixels for every candidate patch."""
        lm = np.asarray(landmarks_px)
        out = {}
        for name, idx in PATCHES.items():
            if name not in self.patches:
                continue
            try:
                out[name] = lm[idx].astype(np.int32)
            except IndexError:
                continue
        return out

    def update(self, frame_bgr, landmarks_px, t=None):
        """Feed one frame. Returns nothing; call estimate() at your own rate.

        `t` is the sample time in seconds. Without it the estimators assume
        the nominal rate, and any shortfall scales every reported rate.
        """
        self.update_means(
            {name: skin_mask_rgb_mean(frame_bgr, poly, cfg=self.cfg)
             for name, poly in self.polygons(landmarks_px).items()}, t)

    def update_means(self, means, t=None):
        """Feed one frame's ALREADY-EXTRACTED patch means. {name: rgb | None}.

        WHY THIS IS A PUBLIC ENTRY POINT

        Tuning is a search over parameters, so the pipeline has to run tens of
        thousands of times. Almost all of the cost of a run is upstream of the
        parameters being tuned: MediaPipe landmarking every frame, then a
        polygon fill and a masked average per patch. The DSP that the
        thresholds actually govern is microseconds by comparison.

        So rppg_eval.py extracts the patch means from a clip ONCE, caches
        them, and replays them through this method for every candidate
        parameter set. A sweep that would take days on video takes seconds,
        and -- the part that matters more -- it is the same selection,
        agreement and tracking code that runs in the interview, not a
        reimplementation of it that could drift.

        Patches absent from `means` are not counted against their own
        coverage, matching `update`: a landmark set that did not yield a
        polygon is a missing observation, not a failed one.
        """
        self.frames += 1
        for name, rgb in means.items():
            p = self.patches.get(name)
            if p is not None:
                p.update(rgb, t)

    # ------------------------------------------------------------ select
    def _select(self):
        """Keep the patches whose behaviour looks cardiac. Colour-blind.

        Three passes, in order: can the patch be measured at all, is it an
        outlier against its neighbours, and do the survivors agree.
        """
        for p in self.patches.values():
            p.score()

        keep, rejected = [], {}

        # Pass 1 -- measurable at all.
        for name in self.names:
            p = self.patches[name]
            if p.coverage < self.c.patch_min_coverage:
                rejected[name] = (f"only {p.coverage:.0%} of frames yielded "
                                  f"enough pixels")
            elif p.bpm is None:
                rejected[name] = "no rate recoverable"
            elif p.sqi < self.c.patch_min_sqi:
                rejected[name] = f"weak periodicity (sqi {p.sqi:.2f})"
            # NOTE: the harmonic ratio is REPORTED, not gated on. Gating
            # rejected all five real recordings available, two of them at a
            # ratio of 0.000. That could mean the peaks were never cardiac --
            # or that at webcam SNR the second harmonic, already weaker than
            # the fundamental, is simply buried in noise. Those two readings
            # have opposite consequences and nothing here can distinguish
            # them, so turning it into a gate would be enforcing a guess.
            # WP8a's contact ground truth settles it; until then the number is
            # shown as evidence for a human to weigh.
            else:
                keep.append(name)

        # Pass 2 -- brightness instability, judged RELATIVE to the other
        # patches. An absolute threshold rejects every patch at once when the
        # room light is simply unsteady, which is a fact about the room and
        # not about any region: measured on a real recording it discarded
        # seven of nine patches, including three parts of one flat forehead.
        # A reflection is a patch that swings MORE than its neighbours.
        if len(keep) >= 3:
            cvs = np.array([self.patches[n].brightness_cv for n in keep])
            baseline = float(np.median(cvs))
            limit = max(baseline * self.c.patch_brightness_outlier_ratio,
                        self.c.patch_max_brightness_cv)
            survivors = []
            for name in keep:
                cv = self.patches[name].brightness_cv
                if cv > limit:
                    rejected[name] = (f"brightness swings {cv:.2f} against a "
                                      f"face median of {baseline:.2f} "
                                      f"(reflection or shadow, not room light)")
                else:
                    survivors.append(name)
            if survivors:
                keep = survivors

        # Pass 3 -- mutual agreement. Not circular: with three or more
        # independent estimates the median is robust, and a patch tracking a
        # different frequency from every other region of the same face is
        # tracking something that is not that face's pulse.
        if len(keep) >= 3:
            bpms = np.array([self.patches[n].bpm for n in keep])
            med = float(np.median(bpms))
            survivors = [n for n in keep
                         if abs(self.patches[n].bpm - med)
                         <= self.c.patch_agreement_tolerance_bpm]
            for name in set(keep) - set(survivors):
                rejected[name] = (f"{self.patches[name].bpm:.0f} bpm against a "
                                  f"consensus of {med:.0f} bpm")
            if len(survivors) >= 2:
                keep = survivors

        keep.sort(key=lambda n: -self.patches[n].sqi)
        keep = keep[:self.c.patch_max_selected]

        # Pass 4 -- is the agreement better than chance?
        #
        # Agreement among a SUBSET proves nothing on its own. Nine patches
        # drawing random rates across a 138 BPM band will often leave three
        # within 12 BPM of each other; measured on pure white noise this
        # reported a confident 67.8 BPM from patches whose true rates were
        # scattered 51-130. So weak support has to buy stronger agreement:
        # a majority of the measurable patches may agree loosely, a minority
        # must agree tightly.
        # A single surviving region has no cross-check by definition, and SQI
        # alone does not distinguish a heartbeat from any other periodic
        # thing. Reporting one is how noise becomes a confident number.
        if len(keep) == 1:
            rejected[keep[0]] = ("only region left; an estimate with no "
                                 "cross-check is not asserted")
            keep = []

        if len(keep) > 1:
            # Every patch that produced a rate is part of the denominator.
            # Counting only those rejected for disagreement inflated the
            # majority fraction and let a 9.9 BPM spread through the loose
            # limit meant for well-supported estimates.
            measurable = sum(1 for p in self.patches.values()
                             if p.bpm is not None)
            bpms = np.array([self.patches[n].bpm for n in keep])
            spread = float(np.ptp(bpms))
            majority = len(keep) >= max(2, self.c.patch_majority_fraction
                                        * max(measurable, 1))
            limit = (self.c.patch_agreement_tolerance_bpm if majority
                     else self.c.patch_minority_max_spread_bpm)
            # A minority also has to be more than two: two random rates
            # landing within 6 BPM of each other across a 138 BPM band happens
            # about one time in twelve, which is not evidence of anything.
            if not majority and len(keep) < self.c.patch_min_minority_regions:
                for n in keep:
                    rejected[n] = (f"only {len(keep)} of {measurable} patches "
                                   f"agreed; too few to rule out coincidence")
                keep = []
            elif spread > limit:
                for n in keep:
                    rejected[n] = (f"{len(keep)} of {measurable} patches agreed "
                                   f"only to {spread:.0f} bpm; too weak to "
                                   f"assert a rate")
                keep = []

        self.selected, self.rejected = keep, rejected
        return keep

    # ---------------------------------------------------------- estimate
    def estimate(self):
        """Fused pulse from the surviving patches, with its own evidence.

        Returns a dict, never a bare number: an estimate with one surviving
        patch and an estimate with five are different claims and the caller
        has to be able to tell them apart.
        """
        if self.frames < self.c.patch_warmup_sec * self.fps:
            return {"bpm": None, "status": "warming up",
                    "warmup_remaining_s": round(
                        self.c.patch_warmup_sec - self.frames / self.fps, 1)}

        keep = self._select()
        if not keep:
            return {"bpm": None, "status": "no usable region",
                    "rejected": dict(self.rejected),
                    "detail": "every candidate patch failed its quality test"}

        bpms = np.array([self.patches[n].bpm for n in keep])
        sqis = np.array([self.patches[n].sqi for n in keep])
        spread = float(np.ptp(bpms)) if len(bpms) > 1 else None
        support = len(keep) / max(len(self.names), 1)

        raw = float(np.average(bpms, weights=sqis))
        tracked, track_status = self.tracker.update(raw, self.frames / self.fps)
        return {
            "bpm": tracked if tracked is not None else raw,
            "bpm_raw": round(raw, 1),
            "tracking": track_status,
            "harmonic": round(float(np.mean(
                [self.patches[n].harmonic for n in keep
                 if self.patches[n].harmonic is not None] or [0.0])), 3),
            "sqi": float(sqis.mean()),
            "roi_spread_bpm": spread,
            "n_regions": len(keep),
            "support": round(support, 2),
            "regions": {n: {"bpm": round(self.patches[n].bpm, 1),
                            "sqi": round(self.patches[n].sqi, 2)}
                        for n in keep},
            "rejected": dict(self.rejected),
            # An estimate from one region has no cross-check. Say so, rather
            # than letting it look like the same measurement as one from five.
            "cross_checked": len(keep) > 1,
            "status": "ok" if len(keep) > 1 else "single region: no cross-check",
        }

    def reset(self):
        for p in self.patches.values():
            p.est.reset()
            p.brightness.clear()
            p.frames = p.valid = 0
        self.frames = 0
        self.selected, self.rejected = [], {}
        self.tracker.reset()
