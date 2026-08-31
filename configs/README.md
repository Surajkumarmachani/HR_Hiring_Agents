# Configuration overlays

Each file here is a **partial** override of the defaults in `config.py`.
Anything absent keeps its default; an unknown key is rejected rather than
ignored, so a typo cannot silently leave a threshold unchanged.

    python3 run_live.py --config configs/strict-pulse.json --consent out/consent.json
    python3 run_live.py --config configs/strict-pulse.json --show-config

Every run prints its config digest and writes a `.config.json` sidecar next to
the feature file. Two recordings with the same digest were measured with the
same instrument; two with different digests were not, and any comparison
between them has to account for it.

WP8b will add one overlay per stratum here as calibration produces them.
