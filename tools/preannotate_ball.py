import argparse
from pathlib import Path

from ball.config import BESTBALLMODELPATH, CVATEXPORTSBALLDIR
from ball.cvat import toCVATVideoXML
from ball.inference import getRawVideoYOLOBall
from utils.io import saveText
from utils.video import getVideoInfo
from utils.yolo import loadYOLOModel


def main():
    parser = argparse.ArgumentParser(description="YOLO Ball to CVAT XML")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--model",
        type=str,
        default=str(BESTBALLMODELPATH),
        help="Path to model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(CVATEXPORTSBALLDIR),
        help="Output Directory",
    )
    args = parser.parse_args()

    videoPath = Path(args.video)
    modelPath = Path(args.model)
    outputPath = Path(args.output_dir) / f"{videoPath.stem}_ball.xml"

    videoInfo = getVideoInfo(videoPath)
    model = loadYOLOModel(modelPath)

    raw = getRawVideoYOLOBall(model, videoPath)
    xml = toCVATVideoXML(
        frames=raw, totalFrames=videoInfo["frame_count"], taskName=videoPath.stem
    )
    saveText(xml, outputPath)
    print(f"[Done] Exported: {outputPath}")


if __name__ == "__main__":
    main()
