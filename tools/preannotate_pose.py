import argparse
from pathlib import Path
from typing import Dict

from pose.config import BESTPOSEMODELPATH, CVATEXPORTSPOSEDIR
from pose.constants import (
    ALLKEYPOINTS,
    BODYKEYPOINTS,
    FACEKEYPOINTS,
    LOWERKEYPOINTS,
    UPPERKEYPOINTS,
)
from pose.preannotate import batchCVATYOLOPosePreannotation
from utils.config import RAWTRAININGVIDEOSDIR
from utils.io import resolvePath
from utils.video import getAllVideoPaths
from utils.yolo import loadYOLOModel


def chooseKeypoints(arg: str) -> Dict[str, int]:
    match arg:
        case "face":
            return FACEKEYPOINTS
        case "upper":
            return UPPERKEYPOINTS
        case "lower":
            return LOWERKEYPOINTS
        case "body":
            return BODYKEYPOINTS
        case "all":
            return ALLKEYPOINTS
        case _:
            raise ValueError(
                "invalid --keypoints argument.\nAllowed values for --keypoints args:"
                '\n  "face"\n  "upper"\n  "lower"\n  "body"\n  "all"'
            )


def main():
    parser = argparse.ArgumentParser(description="YOLO Pose to CVAT XML")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(RAWTRAININGVIDEOSDIR),
        help="Path to input video",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(BESTPOSEMODELPATH),
        help="Path to model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(CVATEXPORTSPOSEDIR),
        help="Output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite XML files of same name",
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        default="body",
        help="COCO keypoints to annotate",
    )
    args = parser.parse_args()

    root = Path(args.root)
    videoPath = resolvePath(root, args.video)
    modelPath = resolvePath(root, args.model)
    outputDir = resolvePath(root, args.output_dir)
    overwrite = args.overwrite
    keypointIndexes = chooseKeypoints(arg=args.keypoints)

    if videoPath.is_dir():
        videoFiles = getAllVideoPaths(
            videoDir=videoPath,
        )
        if not videoFiles:
            print(f"[Error] No .mp4 or .mov files found in {videoPath}")
            return
    else:
        videoFiles = [videoPath]

    model = loadYOLOModel(modelPath=modelPath)
    batchCVATYOLOPosePreannotation(
        model=model,
        videoPaths=videoFiles,
        outputDir=outputDir,
        keypointIndexes=keypointIndexes,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
