from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO

from utils.io import loadJSON, saveJSON

from .constants import LOWERKEYPOINTS


def _keypointConfidenceOneVideo(
    model: YOLO,
    path: Path,
    name: str,
    keypointIndexes: Dict[str, int] = LOWERKEYPOINTS,
    logging: bool = True,
    cache: Dict | None = None,
    cachePath: Path | None = None,
) -> Dict[str, str | float] | None:
    cacheKey = f"{name}_{path.name}"
    if cache is not None and cacheKey in cache:
        if logging:
            print(f"[{name}] Skipping {path.name} (cached)")
        return cache[cacheKey]

    if logging:
        print(f"[{name}] Processing {path.name}...")
    results = {f"{key}_conf": 0 for key in keypointIndexes.keys()}

    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    detections = 0

    for result in model(str(path), stream=True, verbose=False):
        if result.keypoints is None:
            continue
        kps = result.keypoints.data
        if len(kps) == 0:
            continue

        for person in kps:
            detections += 1
            for key, idx in keypointIndexes.items():
                conf = float(person[idx][2])
                results[f"{key}_conf"] += conf

    if detections == 0:
        return None

    for key in results.keys():
        results[key] /= detections

    results["model"] = name
    results["detection_rate"] = detections / frames

    if cache is not None:
        cache[cacheKey] = results
        if cachePath is not None:
            saveJSON(cache, cachePath)

    return results


def keypointConfidenceFromVideos(
    model: YOLO,
    paths: List[Path],
    name: str,
    keypointIndexes: Dict[str, int] = LOWERKEYPOINTS,
    logging: bool = True,
    cache: Dict | None = None,
    cachePath: Path | None = None,
) -> Dict[str, str | float] | None:
    if cache is None and cachePath is not None:
        cache = loadJSON(path=cachePath) or {}
    nsamples = 0
    results = {f"{key}_conf": 0 for key in keypointIndexes.keys()}
    results["detection_rate"] = 0

    for path in paths:
        videoRes = _keypointConfidenceOneVideo(
            model=model,
            path=path,
            name=name,
            keypointIndexes=keypointIndexes,
            logging=logging,
            cache=cache,
            cachePath=cachePath,
        )
        if videoRes is None:
            continue
        nsamples += 1

        for key in results.keys():
            if key not in videoRes:
                continue
            results[key] += videoRes[key]

    if nsamples == 0:
        return None

    for key in results.keys():
        results[key] /= nsamples

    results["model"] = name
    return results
