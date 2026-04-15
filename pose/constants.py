FACEKEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
}

FACESKELETONPAIRS = [
    ("left_eye", "right_eye"),
    ("left_eye", "nose"),
    ("left_eye", "left_ear"),
    ("right_eye", "nose"),
    ("right_eye", "right_ear"),
]

FACEUPPERCONNECTORS = [
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
]

UPPERKEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
}

UPPERSKELETONPAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

UPPERLOWERCONNECTORS = [
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]

LOWERKEYPOINTS = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

LOWERSKELETONPAIRS = [
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

ALLKEYPOINTS = {
    kp: idx
    for kpdict in (FACEKEYPOINTS, UPPERKEYPOINTS, LOWERKEYPOINTS)
    for kp, idx in kpdict.items()
}

ALLSKELETONPAIRS = (
    FACESKELETONPAIRS
    + FACEUPPERCONNECTORS
    + UPPERSKELETONPAIRS
    + UPPERLOWERCONNECTORS
    + LOWERSKELETONPAIRS
)

BODYKEYPOINTS = {
    kp: idx for kpdict in (UPPERKEYPOINTS, LOWERKEYPOINTS) for kp, idx in kpdict.items()
}

BODYSKELETONPAIRS = UPPERSKELETONPAIRS + UPPERLOWERCONNECTORS + LOWERSKELETONPAIRS
