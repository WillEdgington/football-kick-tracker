from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import pytest

from utils.video import getAllVideoPaths, getVideoInfo


@pytest.fixture
def mockVideoDir(tmp_path):
    """Creates a fake directory structure with various files."""
    (tmp_path / "clip1.mp4").touch()
    (tmp_path / "clip2.mov").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "test_clip3.mp4").touch()
    return tmp_path


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


def test_getAllVideoPaths_findsAllVideos(mockVideoDir):
    paths = getAllVideoPaths(mockVideoDir)
    assert len(paths) == 3
    assert all(p.suffix in [".mp4", ".mov"] for p in paths)


def test_getAllVideoPaths_respectsPrefix(mockVideoDir):
    paths = getAllVideoPaths(mockVideoDir, prefix="test_")
    assert len(paths) == 1
    assert paths[0].name == "test_clip3.mp4"


def test_getAllVideoPaths_togglesSuffixes(mockVideoDir):
    only_mp4 = getAllVideoPaths(mockVideoDir, mov=False)
    assert all(p.suffix == ".mp4" for p in only_mp4)
    assert len(only_mp4) == 2


def test_getAllVideoPaths_raisesOnInvalidDir():
    with pytest.raises(AssertionError):
        getAllVideoPaths(Path("not/a/dir"))
