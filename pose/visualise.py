from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics.engine.results import Results

from .config import CONFTHRESHOLD, LOWERCOLOURS
from .constants import LOWERKEYPOINTS, LOWERSKELETONPAIRS


def drawKeypoints(
    frame: np.ndarray,
    results: List[Results],
    keypointIndexes: Dict[str, int] = LOWERKEYPOINTS,
    skeletonPairs: List[Tuple[str, str]] = LOWERSKELETONPAIRS,
    colours: Dict[str, Tuple[int, int, int]] = LOWERCOLOURS,
    confThreshold: float = CONFTHRESHOLD,
) -> np.ndarray:
    """
    Draw keypoints and skeleton lines on a frame.

    Args:
        frame: RGB image as a numpy array (H, W, 3). Must be RGB not BGR.
               (convert with cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if sourcing
               from OpenCV or Ultralytics result.orig_img)
        results: Ultralytics Results list from model inference.
        keypointIndexes: mapping of joint name ot COCO keypoint index.
        skeletonPairs: list of (jointA, jointB) pairs to draw lines between.
        colours: mapping of joint name to RGB colour tuple.
        confThreshold: minimum keypoint confidence to draw. Range [0, 1).

    Returns:
        Annotated copy of the input frame as an RGB numpy array.
    """
    assert (
        colours.keys() == keypointIndexes.keys()
    ), "colours must contain all the same keys as the keypointIndexes"
    assert 0 <= confThreshold < 1, "confThreshold must be in the range [0, 1)"
    allPairNames = {name for pair in skeletonPairs for name in pair}
    assert (
        allPairNames <= keypointIndexes.keys()
    ), "skeletonPairs can only contain strings that are also keys in keypointIndexes"

    img = frame.copy()
    if results[0].keypoints is None:
        return img

    kps = results[0].keypoints.data

    for person in kps:
        for part, idx in keypointIndexes.items():
            x, y, conf = person[idx]
            if conf < confThreshold:
                continue
            cv2.circle(img, (int(x), int(y)), 5, colours[part], -1)
            cv2.putText(
                img,
                f"{conf:.2f}",
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                colours[part],
                1,
            )

        for a, b in skeletonPairs:
            xa, ya, ca = person[keypointIndexes[a]]
            xb, yb, cb = person[keypointIndexes[b]]
            if ca < confThreshold or cb < confThreshold:
                continue
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (200, 200, 200), 2)
    return img
