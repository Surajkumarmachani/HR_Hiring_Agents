"""
FACS Action Unit proxies derived from MediaPipe FaceLandmarker blendshapes.

IMPORTANT SCIENTIFIC NOTE
-------------------------
These are *proxies*, not certified FACS coding. MediaPipe blendshapes are
trained for avatar retargeting, not FACS intensity. They correlate with AU
activation but are not calibrated to the 0-5 FACS intensity scale.

Use this module for REAL-TIME work (it runs at 30+ fps on CPU).
For offline/gold-standard AU intensities, swap in py-feat or OpenFace 3.0
via signals/face.py's `backend` argument. Always report which backend
produced a given number.

Each AU entry: (au_code, human_name, region, [blendshape components])
Bilateral AUs list left and right separately so asymmetry can be computed.
"""

# region: "upper" = above the nasal root, "mid" = eyes/cheeks/nose,
#         "lower" = mouth/jaw, "head" = rigid head motion, "eye" = gaze
AU_DEFINITIONS = {
    # ---- Upper face -------------------------------------------------
    "AU01": ("Inner brow raiser", "upper", ["browInnerUp"], None),
    "AU02": ("Outer brow raiser", "upper", ["browOuterUpLeft"], ["browOuterUpRight"]),
    "AU04": ("Brow lowerer", "upper", ["browDownLeft"], ["browDownRight"]),
    # ---- Mid face / periocular --------------------------------------
    "AU05": ("Upper lid raiser", "mid", ["eyeWideLeft"], ["eyeWideRight"]),
    "AU06": ("Cheek raiser (Duchenne)", "mid", ["cheekSquintLeft"], ["cheekSquintRight"]),
    "AU07": ("Lid tightener", "mid", ["eyeSquintLeft"], ["eyeSquintRight"]),
    "AU09": ("Nose wrinkler", "mid", ["noseSneerLeft"], ["noseSneerRight"]),
    "AU45": ("Blink", "mid", ["eyeBlinkLeft"], ["eyeBlinkRight"]),
    # ---- Lower face --------------------------------------------------
    "AU10": ("Upper lip raiser", "lower", ["mouthShrugUpper"], None),
    "AU12": ("Lip corner puller (smile)", "lower", ["mouthSmileLeft"], ["mouthSmileRight"]),
    "AU14": ("Dimpler", "lower", ["mouthDimpleLeft"], ["mouthDimpleRight"]),
    "AU15": ("Lip corner depressor", "lower", ["mouthFrownLeft"], ["mouthFrownRight"]),
    "AU16": ("Lower lip depressor", "lower", ["mouthLowerDownLeft"], ["mouthLowerDownRight"]),
    "AU17": ("Chin raiser", "lower", ["mouthShrugLower"], None),
    "AU18": ("Lip pucker", "lower", ["mouthPucker"], None),
    "AU20": ("Lip stretcher", "lower", ["mouthStretchLeft"], ["mouthStretchRight"]),
    "AU22": ("Lip funneler", "lower", ["mouthFunnel"], None),
    "AU24": ("Lip pressor", "lower", ["mouthPressLeft"], ["mouthPressRight"]),
    "AU25": ("Lips part", "lower", ["mouthClose"], None),  # inverted downstream
    "AU26": ("Jaw drop", "lower", ["jawOpen"], None),
    "AU28": ("Lip suck", "lower", ["mouthRollUpper"], ["mouthRollLower"]),
    "AU29": ("Jaw thrust", "lower", ["jawForward"], None),
    "AU30": ("Jaw sideways", "lower", ["jawLeft"], ["jawRight"]),
    "AU33": ("Cheek puff", "lower", ["cheekPuff"], None),
    "AU36": ("Tongue show", "lower", ["tongueOut"], None),
    "AU38": ("Mouth sideways", "lower", ["mouthLeft"], ["mouthRight"]),
}

# Rigid head AUs (FACS 51-58) come from the head-pose transform, not blendshapes.
HEAD_AUS = {
    "AU51": ("Head turn left", "head"),
    "AU52": ("Head turn right", "head"),
    "AU53": ("Head up", "head"),
    "AU54": ("Head down", "head"),
    "AU55": ("Head tilt left", "head"),
    "AU56": ("Head tilt right", "head"),
    "AU57": ("Head forward", "head"),
    "AU58": ("Head back", "head"),
}

# Gaze AUs (FACS 61-64) come from the eyeLook* blendshapes.
GAZE_AUS = {
    "AU61": ("Eyes turn left", "eye", ["eyeLookOutLeft", "eyeLookInRight"]),
    "AU62": ("Eyes turn right", "eye", ["eyeLookInLeft", "eyeLookOutRight"]),
    "AU63": ("Eyes up", "eye", ["eyeLookUpLeft", "eyeLookUpRight"]),
    "AU64": ("Eyes down", "eye", ["eyeLookDownLeft", "eyeLookDownRight"]),
}


def au_catalogue():
    """Flat list of every AU this pipeline emits, for docs and schema checks."""
    rows = []
    for code, (name, region, _l, _r) in AU_DEFINITIONS.items():
        rows.append((code, name, region))
    for code, (name, region) in HEAD_AUS.items():
        rows.append((code, name, region))
    for code, (name, region, _c) in GAZE_AUS.items():
        rows.append((code, name, region))
    return sorted(rows)


if __name__ == "__main__":
    cat = au_catalogue()
    print(f"{len(cat)} action units emitted by this pipeline\n")
    for code, name, region in cat:
        print(f"  {code:<6} {region:<6} {name}")
