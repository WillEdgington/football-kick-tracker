from pathlib import Path

from ultralytics import YOLO


def loadYOLOModel(modelPath: Path) -> YOLO:
    return YOLO(str(modelPath))
