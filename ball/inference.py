from pathlib import Path
from typing import Any, Dict, List

import cv2
from ultralytics import YOLO


def getRawVideoYOLOBall(
    model: YOLO,
    path: Path,
    name: str = "YOLO-ball",
    logging: bool = True,
) -> List[Dict[str, Any]]:
    if logging:
        print(f"[{name}] Processing {path.name}...")
    results = []

    cap = cv2.VideoCapture(str(path))
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    for frameIdx, result in enumerate(model(str(path), stream=True, verbose=False)):
        if logging:
            print(f"  [{name}] frame {frameIdx + 1}/{totalFrames}", end="\r")
        frameData = {
            "frame": frameIdx,
            "detections": [],
        }
        if result.boxes is None or len(result.boxes.data) == 0:
            results.append(frameData)
            continue

        boxes = result.boxes.data

        for ballIdx in range(len(boxes)):
            frameData["detections"].append(
                {
                    "box": boxes[ballIdx, :4].tolist(),
                    "conf": float(boxes[ballIdx, 4]),
                }
            )
        results.append(frameData)

    if logging:
        print(f"\n[{name}] Processing Complete.")
    return results
