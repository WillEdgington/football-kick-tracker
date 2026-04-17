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

    # Skeleton and Box existence
    skeleton = root.find(".//skeleton")
    assert skeleton is not None
    assert skeleton.find("box") is not None

    # Occlusion
    pts = root.findall(".//points")
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
    pointsTags = root.findall(".//points")
    pointLabels = [p.get("label") for p in pointsTags]

    # should contain left_ankle, but NOT right_ankle
    assert "left_ankle" in pointLabels
    assert "right_ankle" not in pointLabels

    # left_knee should not be in point labels, even though it is in the pose data
    assert "left_knee" not in pointLabels

    # only expect one tag (left_ankle)
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
