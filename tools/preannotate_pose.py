import argparse
from pathlib import Path

from pose.config import BESTPOSEMODELPATH, CVATEXPORTSPOSEDIR
from pose.constants import BODYKEYPOINTS
from pose.preannotate import batchCVATYOLOPosePreannotation
from utils.config import RAWTRAININGVIDEOSDIR
from utils.io import resolvePath
from utils.video import getAllVideoPaths
from utils.yolo import loadYOLOModel


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
    args = parser.parse_args()

    root = Path(args.root)
    videoPath = resolvePath(root, args.video)
    modelPath = resolvePath(root, args.model)
    outputDir = resolvePath(root, args.output_dir)
    overwrite = args.overwrite

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
        keypointIndexes=BODYKEYPOINTS,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
