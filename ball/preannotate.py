from pathlib import Path
from typing import List

from ultralytics import YOLO

from ball.config import CONFTHRESHOLD, CVATEXPORTSBALLDIR
from ball.cvat import toCVATVideoXML
from ball.inference import getRawVideoYOLOBall
from utils.io import saveText
from utils.video import getVideoInfo


def batchCVATYOLOBallPreannotation(
    model: YOLO,
    videoPaths: List[Path],
    outputDir: Path = CVATEXPORTSBALLDIR,
    name: str = "YOLO-ball",
    logging: bool = True,
    overwrite: bool = False,
) -> None:
    for video in videoPaths:
        if logging:
            print(f"[Started] Processing: {video.name}")
        try:
            outputPath = outputDir / f"{video.stem}_ball.xml"
            if outputPath.exists() and not overwrite:
                if logging:
                    print(f"[Skipped] Output path already exists: {outputPath}\n")
                continue
            videoInfo = getVideoInfo(video)
            raw = getRawVideoYOLOBall(
                model=model,
                path=video,
                name=name,
                logging=logging,
            )
            xml = toCVATVideoXML(
                frames=raw,
                totalFrames=videoInfo["frame_count"],
                taskName=video.stem,
                confThreshold=CONFTHRESHOLD,
            )
            saveText(xml, outputPath)
            if logging:
                print(f"[Exported] CVAT pre-annotation XML: {outputPath.name}\n")
        except Exception as e:
            if logging:
                print(f"[Failed] {video.name}: {e}")

    if logging:
        print("[Pre-annotation Complete]")
