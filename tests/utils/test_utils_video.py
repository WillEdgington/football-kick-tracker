from unittest.mock import MagicMock, patch

import cv2
import pytest

from utils.video import getVideoInfo


@patch("utils.video.cv2.VideoCapture")
def test_getVideoInfo_returnsCorrectMetadata(mock_cap_class, tmp_path):
    mockCap = MagicMock()
    mockCap.isOpened.return_value = True

    def mockGet(prop):
        mapping = {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }
        return mapping.get(prop, 0)

    mockCap.get.side_effect = mockGet
    mock_cap_class.return_value = mockCap

    result = getVideoInfo(tmp_path / "fake_video.mp4")

    assert result["frame_count"] == 100
    assert result["fps"] == 30.0
    assert result["width"] == 1920
    mockCap.release.assert_called_once()


def test_getVideoInfo_raisesErrorOnInvalidPath(tmp_path):
    with patch("utils.video.cv2.VideoCapture") as mockVC:
        mockVC.return_value.isOpened.return_value = False
        with pytest.raises(ValueError, match="Could not open video"):
            getVideoInfo(tmp_path / "nonexistent.mp4")
