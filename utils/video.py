from pathlib import Path
from typing import Any, Dict, List

import cv2


def getVideoInfo(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def getAllVideoPaths(
    videoDir: Path, mp4: bool = True, mov: bool = True, prefix: str | None = None
) -> List[Path]:
    assert mp4 or mov, "mp4 or mov must be set to True"
    assert videoDir.is_dir(), "videoDir must be a valid directory"
    suffixes = []
    if mp4:
        suffixes.append("*.mp4")
    if mov:
        suffixes.append("*.mov")

    videoPaths = []
    for suf in suffixes:
        videoPaths.extend(videoDir.glob(suf))
    if prefix is not None:
        videoPaths = [p for p in videoPaths if p.stem.startswith(prefix)]
    return videoPaths
