from pathlib import Path

CONFTHRESHOLD = 0.5

LOWERCOLOURS = {
    "left_hip": (255, 100, 0),
    "right_hip": (255, 100, 0),
    "left_knee": (0, 200, 255),
    "right_knee": (0, 200, 255),
    "left_ankle": (0, 255, 100),
    "right_ankle": (0, 255, 100),
}

COMPSCOREWEIGHTS = {
    "detection_rate": 0.5,
    "left_ankle_conf": 0.15,
    "right_ankle_conf": 0.15,
    "left_knee_conf": 0.1,
    "right_knee_conf": 0.1,
}

RAWTRAININGVIDEOSDIR = Path("data/raw/training_drills_videos")
ANNOTATEDPOSEVIDEOSDIR = Path("data/sessions/annotated_videos")
