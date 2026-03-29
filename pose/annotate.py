from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd
from ultralytics import YOLO

from .config import CONFTHRESHOLD, LOWERCOLOURS
from .constants import LOWERKEYPOINTS, LOWERSKELETONPAIRS
from .visualise import drawKeypoints


def annotateVideo(
    model: YOLO,
    sourcePath: Path,
    outputPath: Path,
    keypointIndexes: Dict[str, int] = LOWERKEYPOINTS,
    skeletonPairs: List[Tuple[str, str]] = LOWERSKELETONPAIRS,
    colours: Dict[str, Tuple[int, int, int]] = LOWERCOLOURS,
    confThreshold: float = CONFTHRESHOLD,
    logging: bool = True,
) -> None:
    cap = cv2.VideoCapture(str(sourcePath))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(outputPath), fourcc, fps, (width, height))

    for i, result in enumerate(model(str(sourcePath), stream=True, verbose=False)):
        if logging:
            print(f"Frame {i + 1}/{totalFrames}", end="\r")
        bgr = result.orig_img
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        annotated = drawKeypoints(
            frame=rgb,
            results=[result],
            keypointIndexes=keypointIndexes,
            skeletonPairs=skeletonPairs,
            colours=colours,
            confThreshold=confThreshold,
        )
        bgrAnnotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(bgrAnnotated)

    writer.release()
    if logging:
        print(f"\nSaved to {outputPath}")


def _getOutputPaths(
    name: str,
    sourcePaths: List[Path],
    outputDir: Path,
) -> List[Path]:
    return [outputDir / name / path.with_suffix(".mp4").name for path in sourcePaths]


def annotateVideos(
    model: YOLO,
    name: str,
    sourcePaths: List[Path],
    outputDir: Path,
    keypointIndexes: Dict[str, int] = LOWERKEYPOINTS,
    skeletonPairs: List[Tuple[str, str]] = LOWERSKELETONPAIRS,
    colours: Dict[str, Tuple[int, int, int]] = LOWERCOLOURS,
    confThreshold: float = CONFTHRESHOLD,
    logging: bool = True,
    overwrite: bool = False,
) -> List[Path]:
    outputPaths = _getOutputPaths(
        name=name,
        sourcePaths=sourcePaths,
        outputDir=outputDir,
    )

    total = len(sourcePaths)
    skipped = 0

    for i, (outputPath, sourcePath) in enumerate(zip(outputPaths, sourcePaths)):
        if outputPath.exists() and not overwrite:
            if logging:
                print(
                    f"[{name}] Skipping {sourcePath.name}"
                    " ({i + 1}/{total}) - already exists"
                )
            skipped += 1
            continue
        if logging:
            print(f"[{name}] Annotating {sourcePath.name} ({i+1}/{total})...")
        annotateVideo(
            model=model,
            sourcePath=sourcePath,
            outputPath=outputPath,
            keypointIndexes=keypointIndexes,
            skeletonPairs=skeletonPairs,
            colours=colours,
            confThreshold=confThreshold,
            logging=logging,
        )

    if logging:
        completed = total - skipped
        print(f"\n[{name}] Done - {completed} annotated, {skipped} skipped.")
    return outputPaths


def _getModelByName(
    name: str,
    models: List[Tuple[str, YOLO]],
    logging: bool = True,
) -> YOLO | None:
    for n, model in models:
        if name == n:
            if logging:
                print(f"[{name}] Found model.")
            return model
    if logging:
        print(f"[{name}] Could not find model in models")
    return None


def annotateVideosForTopNModels(
    df: pd.DataFrame,
    models: List[Tuple[str, YOLO]],
    sourcePaths: List[Path],
    outputDir: Path,
    keypointIndexes: Dict[str, int] = LOWERKEYPOINTS,
    skeletonPairs: List[Tuple[str, str]] = LOWERSKELETONPAIRS,
    colours: Dict[str, Tuple[int, int, int]] = LOWERCOLOURS,
    confThreshold: float = CONFTHRESHOLD,
    logging: bool = True,
    overwrite: bool = False,
    n: int = 2,
) -> Dict[str, List[Path]]:
    modelNames = df["model"]
    assert n <= len(modelNames), f"n must be less than length of df: {len(modelNames)}"
    outputPaths = {}

    for i in range(n):
        name = modelNames.iloc[i]
        model = _getModelByName(name=name, models=models, logging=logging)
        if model is None:
            continue
        outputPaths[name] = annotateVideos(
            model=model,
            name=name,
            sourcePaths=sourcePaths,
            outputDir=outputDir,
            keypointIndexes=keypointIndexes,
            skeletonPairs=skeletonPairs,
            colours=colours,
            confThreshold=confThreshold,
            logging=logging,
            overwrite=overwrite,
        )
    return outputPaths
