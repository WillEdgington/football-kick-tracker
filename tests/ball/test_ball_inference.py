from unittest.mock import MagicMock, patch

import pytest

from ball.inference import getRawVideoYOLOBall


@pytest.fixture
def videoPath(tmp_path):
    p = tmp_path / "ball_clip.mp4"
    p.touch()
    return p


def test_getRawVideoYOLOBall_extractsDataCorrectly(videoPath):
    result = MagicMock()
    mockBoxes = MagicMock()
    mockBoxes.__len__.return_value = 2
    mockBoxes.__getitem__.return_value.tolist.return_value = [50, 60, 70, 80]
    mockBoxes.__getitem__.return_value.__float__.return_value = 0.88

    result.boxes.data = mockBoxes
    model = MagicMock()
    model.side_effect = lambda *a, **kw: iter([result])

    with patch("ball.inference.cv2.VideoCapture") as mockCap:
        mockCap.return_value.get.return_value = 1
        out = getRawVideoYOLOBall(
            model=model,
            path=videoPath,
            logging=False,
        )

    assert len(out) == 1
    assert len(out[0]["detections"]) == 2
    assert out[0]["detections"][0]["box"] == [50, 60, 70, 80]
    assert out[0]["detections"][0]["conf"] == 0.88


def test_getRawVideoYOLOBall_emptyFrame_returnsFrameWithNoDetections(videoPath):
    result = MagicMock()
    result.boxes.data = []  # no balls found
    model = MagicMock()
    model.side_effect = lambda *a, **kw: iter([result])

    with patch("ball.inference.cv2.VideoCapture") as mockCap:
        mockCap.return_value.get.return_value = 1
        out = getRawVideoYOLOBall(
            model=model,
            path=videoPath,
            logging=False,
        )

    assert out[0]["frame"] == 0
    assert out[0]["detections"] == []
