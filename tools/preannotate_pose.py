import argparse
from pathlib import Path

from pose.config import BESTPOSEMODELPATH, CVATEXPORTSPOSEDIR
from pose.constants import BODYKEYPOINTS
from pose.cvat import toCVATVideoXML
from pose.inference import getRawVideoYOLOPose
from utils.io import saveText
from utils.video import getVideoInfo
from utils.yolo import loadYOLOModel


def main():
    parser = argparse.ArgumentParser(description="YOLO Pose to CVAT XML")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--model", type=str, default=str(BESTPOSEMODELPATH), help="Path to model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(CVATEXPORTSPOSEDIR),
        help="Output Directory",
    )
    args = parser.parse_args()

    videoPath = Path(args.video)
    modelPath = Path(args.model)
    outputPath = Path(args.output_dir) / f"{videoPath.stem}_pose.xml"

    videoInfo = getVideoInfo(videoPath)
    model = loadYOLOModel(modelPath)

    raw = getRawVideoYOLOPose(model, videoPath, keypointIndexes=BODYKEYPOINTS)
    xml = toCVATVideoXML(
        frames=raw, totalFrames=videoInfo["frame_count"], taskName=videoPath.stem
    )
    saveText(xml, outputPath)
    print(f"[Done] Exported: {outputPath}")


if __name__ == "__main__":
    main()
