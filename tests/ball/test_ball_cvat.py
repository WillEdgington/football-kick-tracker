from xml.etree.ElementTree import fromstring

import pytest

from ball.cvat import toCVATVideoXML


@pytest.fixture
def sampleBallData():
    """one frame (frame=10), two balls (high confidence)"""
    return [
        {
            "frame": 10,
            "detections": [
                {"box": [100, 200, 150, 250], "conf": 0.9},
                {"box": [300, 400, 350, 450], "conf": 0.8},
            ],
        }
    ]


def test_toCVATVideoXML_incrementsTrackIds(sampleBallData):
    xmlOut = toCVATVideoXML(sampleBallData, totalFrames=100)
    root = fromstring(xmlOut)

    tracks = root.findall("track")
    assert len(tracks) == 2
    assert tracks[0].get("id") == "0"
    assert tracks[1].get("id") == "1"


def test_toCVATVideoXML_preventsOutOfBoundsFrame(sampleBallData):
    xmlOut = toCVATVideoXML(sampleBallData, totalFrames=11)  # frame = 10 + 1 is oob
    root = fromstring(xmlOut)

    boxes = root.findall(".//track[@id='0']/box")
    assert len(boxes) == 1


def test_toCVATVideoXML_handlesEmptyInput():
    xmlOut = toCVATVideoXML([], totalFrames=10)
    root = fromstring(xmlOut)

    # Meta should exist, tracks should be empty
    assert root.find(".//size").text == "10"
    assert len(root.findall("track")) == 0


def test_toCVATVideoXML_skipsMalformedFrame():
    malformedData = [{"not_a_frame": 0}]
    xmlOut = toCVATVideoXML(malformedData, 10)
    root = fromstring(xmlOut)
    assert len(root.findall("track")) == 0
