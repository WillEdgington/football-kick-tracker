from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO

from pose.config import CONFTHRESHOLD, CVATEXPORTSPOSEDIR
from pose.constants import BODYKEYPOINTS
from pose.cvat import toCVATVideoXML
from pose.inference import getRawVideoYOLOPose
from utils.io import saveText
from utils.video import getVideoInfo


def batchCVATYOLOPosePreannotation(
    model: YOLO,
    videoPaths: List[Path],
    outputDir: Path = CVATEXPORTSPOSEDIR,
    name: str = "YOLO-pose",
    keypointIndexes: Dict[str, int] = BODYKEYPOINTS,
    logging: bool = True,
    overwrite: bool = False,
) -> None:
    for video in videoPaths:
        if logging:
            print(f"[Started] Processing: {video.name}")
        try:
            outputPath = outputDir / f"{video.stem}_pose.xml"
            if outputPath.exists() and not overwrite:
                if logging:
                    print(f"[Skipped] Output path already exists: {outputPath}\n")
                continue
            videoInfo = getVideoInfo(video)
            raw = getRawVideoYOLOPose(
                model=model,
                path=video,
                keypointIndexes=keypointIndexes,
                name=name,
                logging=logging,
            )
            xml = toCVATVideoXML(
                frames=raw,
                totalFrames=videoInfo["frame_count"],
                keypointIndexes=keypointIndexes,
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
