from typing import Any, Dict, List
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

from ball.config import CONFTHRESHOLD


def _buildMeta(
    root: Element,
    totalFrames: int,
    taskName: str,
) -> None:
    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    SubElement(task, "id").text = "0"
    SubElement(task, "name").text = taskName
    SubElement(task, "size").text = str(totalFrames)
    SubElement(task, "mode").text = "interpolation"
    SubElement(task, "overlap").text = "0"
    SubElement(task, "flipped").text = "False"
    labels = SubElement(task, "labels")
    label = SubElement(labels, "label")
    SubElement(label, "name").text = "ball"
    SubElement(label, "type").text = "bbox"


def toCVATVideoXML(
    frames: List[Dict[str, Any]],
    totalFrames: int,
    taskName: str = "ball-preannotation",
    confThreshold: float = CONFTHRESHOLD,
) -> str:
    root = Element("annotations")
    SubElement(root, "version").text = "1.1"
    _buildMeta(root, totalFrames, taskName)

    trackId = 0
    for frameData in frames:
        frameIdx = frameData.get("frame", None)
        if frameIdx is None:
            continue
        for det in frameData.get("detections", []):
            x1, y1, x2, y2 = det["box"]
            track = SubElement(root, "track", id=str(trackId), source="auto")

            SubElement(
                track,
                "box",
                frame=str(frameIdx),
                xtl=f"{x1:.2f}",
                ytl=f"{y1:.2f}",
                xbr=f"{x2:.2f}",
                ybr=f"{y2:.2f}",
                outside="0",
                occluded="0" if det["conf"] >= confThreshold else "1",
                keyframe="1",
            )

            # Mark track as outside on the very next frame
            # to prevent CVAT interpolating forward
            if frameIdx + 1 < totalFrames:
                SubElement(
                    track,
                    "box",
                    frame=str(frameIdx + 1),
                    xtl=f"{x1:.2f}",
                    ytl=f"{y1:.2f}",
                    xbr=f"{x2:.2f}",
                    ybr=f"{y2:.2f}",
                    outside="1",
                    occluded="0",
                    keyframe="1",
                )
            trackId += 1
    return parseString(tostring(root, encoding="unicode")).toprettyxml(indent="  ")
