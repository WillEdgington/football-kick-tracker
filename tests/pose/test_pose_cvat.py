from xml.etree.ElementTree import fromstring

import pytest

from pose.cvat import toCVATVideoXML


@pytest.fixture
def poseIndexes():
    return {"left_ankle": 15, "right_ankle": 16}


@pytest.fixture
def samplePoseData():
    return [
        {
            "frame": 0,
            "detections": [
                {
                    "box": [10, 20, 30, 40],
                    "keypoints": {
                        "left_ankle": [105.0, 205.0, 0.9],
                        "right_ankle": [110.0, 210.0, 0.1],  # Low confidence
                    },
                }
            ],
        }
    ]


@pytest.fixture
def samplePoseDataMissingAndExtraKeypoint():
    return [
        {
            "frame": 0,
            "detections": [
                {
                    "box": [10, 20, 30, 40],
                    "keypoints": {
                        "left_knee": [102.0, 195.0, 0.9],
                        "left_ankle": [105.0, 205.0, 0.9],
                    },
                }
            ],
        }
    ]


def test_toCVATVideoXML_fullCoverage_pose(samplePoseData, poseIndexes):
    xmlOut = toCVATVideoXML(
        samplePoseData, 10, keypointIndexes=poseIndexes, confThreshold=0.5
    )
    root = fromstring(xmlOut)

    # find active skeleton
    skeleton = root.find(".//skeleton[@outside='0']")
    assert skeleton is not None

    # check for termination skeleton
    outskeleton = root.find(".//skeleton[@outside='1']")
    assert outskeleton is not None

    # occlusion Check on the active frame
    pts = skeleton.findall("points")
    rightAnkle = next(p for p in pts if p.get("label") == "right_ankle")
    assert rightAnkle.get("occluded") == "1"


def test_toCVATVideoXML_handlesMissingAndExtraKeypoints(
    samplePoseDataMissingAndExtraKeypoint, poseIndexes
):
    xmlOut = toCVATVideoXML(
        samplePoseDataMissingAndExtraKeypoint,
        totalFrames=10,
        keypointIndexes=poseIndexes,
    )
    root = fromstring(xmlOut)

    # Target only the active skeleton for point counting
    activeSkel = root.find(".//skeleton[@outside='0']")
    pointsTags = activeSkel.findall("points")
    pointLabels = [p.get("label") for p in pointsTags]

    # assertions for the active frame
    assert "left_ankle" in pointLabels
    assert "right_ankle" not in pointLabels
    assert "left_knee" not in pointLabels
    assert len(pointsTags) == 1


def test_toCVATVideoXML_preventsOutOfBoundsFrame(samplePoseData):
    xmlOut = toCVATVideoXML(samplePoseData, totalFrames=1)  # frame = 10 + 1 is oob
    root = fromstring(xmlOut)

    skeletons = root.findall(".//track[@id='0']/skeleton")
    assert len(skeletons) == 1


def test_toCVATVideoXML_handlesEmptyInput():
    xmlOut = toCVATVideoXML([], totalFrames=10)
    root = fromstring(xmlOut)

    # Meta should exist, tracks should be empty
    assert root.find(".//size").text == "10"
    assert len(root.findall("track")) == 0


def test_toCVATVideoXML_skipsMalformedFrame():
    malformedData = [{"error": 0}]
    xmlOut = toCVATVideoXML(malformedData, 10)
    root = fromstring(xmlOut)
    assert len(root.findall("track")) == 0
