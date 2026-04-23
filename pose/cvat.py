from typing import Any, Dict, List
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

from pose.config import CONFTHRESHOLD
from pose.constants import BODYKEYPOINTS


def _buildMeta(
    root: Element,
    totalFrames: int,
    taskName: str,
    keypointIndexes: Dict[str, int] = BODYKEYPOINTS,
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

    skelLabel = SubElement(labels, "label")
    SubElement(skelLabel, "name").text = "person"
    SubElement(skelLabel, "type").text = "skeleton"

    for kpName in keypointIndexes:
        kplabel = SubElement(labels, "label")
        SubElement(kplabel, "name").text = kpName
        SubElement(kplabel, "type").text = "points"
        SubElement(kplabel, "parent").text = "person"


def toCVATVideoXML(
    frames: List[Dict[str, Any]],
    totalFrames: int,
    keypointIndexes: Dict[str, int] = BODYKEYPOINTS,
    taskName: str = "pose-preannotation",
    confThreshold: float = CONFTHRESHOLD,
) -> str:
    root = Element("annotations")
    SubElement(root, "version").text = "1.1"
    _buildMeta(root, totalFrames, taskName, keypointIndexes)

    trackId = 0
    for frameData in frames:
        frameIdx = frameData.get("frame", None)
        if frameIdx is None:
            continue
        for det in frameData.get("detections", []):
            x1, y1, x2, y2 = det["box"]
            track = SubElement(
                root, "track", id=str(trackId), label="person", source="auto"
            )

            skel = SubElement(
                track,
                "skeleton",
                frame=str(frameIdx),
                outside="0",
                occluded="0",
                keyframe="1",
            )
            # Mark track as outside on the very next frame
            # to prevent CVAT interpolating forward
            outskel = (
                SubElement(
                    track,
                    "skeleton",
                    frame=str(frameIdx + 1),
                    outside="1",
                    occluded="0",
                    keyframe="1",
                )
                if frameIdx + 1 < totalFrames
                else None
            )

            for kpName in keypointIndexes:
                kpData = det["keypoints"].get(kpName)  # [x, y, conf]
                if kpData is None:
                    continue
                kx, ky, kconf = det["keypoints"][kpName]
                SubElement(
                    skel,
                    "points",
                    label=kpName,
                    points=f"{kx:.2f},{ky:.2f}",
                    outside="0",
                    occluded="0" if kconf >= confThreshold else "1",
                    keyframe="1",
                )
                # mark keypoint as outside in next frame
                if outskel is not None:
                    SubElement(
                        outskel,
                        "points",
                        label=kpName,
                        points=f"{kx:.2f},{ky:.2f}",
                        outside="1",
                        occluded="0" if kconf >= confThreshold else "1",
                        keyframe="1",
                    )
            trackId += 1
    return parseString(tostring(root, encoding="unicode")).toprettyxml(indent="  ")
