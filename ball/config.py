from pathlib import Path

from utils.config import CVATEXPORTSDIR, MODELSPATH, SESSIONSDIR

CONFTHRESHOLD = 0.25

ANNOTATEDBALLVIDEOSDIR = SESSIONSDIR / Path("ball/annotated_videos")
CVATEXPORTSBALLDIR = CVATEXPORTSDIR / Path("ball")

# current best is yolo11n finetuned on broadcast match footage from roboflow
BESTBALLMODELPATH = MODELSPATH / Path("yolo11n_ball/weights/best.pt")
