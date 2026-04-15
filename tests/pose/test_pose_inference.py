import json
from unittest.mock import MagicMock, patch

import pytest

from pose.inference import (
    _keypointConfidenceOneVideo,
    keypointConfidenceFromVideos,
)

KEYPOINTINDEXES = {"left_ankle": 15, "right_ankle": 16}


@pytest.fixture
def videoPath(tmp_path):
    p = tmp_path / "pose_clip.mp4"
    p.touch()
    return p


@pytest.fixture
def cachePath(tmp_path):
    return tmp_path / "cache.json"


def test_keypointConfidenceOneVideo_noDetections_returnsNone(videoPath):
    result = MagicMock()
    result.keypoints = None
    model = MagicMock()
    model.side_effect = lambda *a, **kw: iter([result])

    with patch("pose.inference.cv2.VideoCapture") as mockCap:
        mockCap.return_value.get.return_value = 10
        out = _keypointConfidenceOneVideo(
            model=model,
            path=videoPath,
            name="yolo11l-pose",
            keypointIndexes=KEYPOINTINDEXES,
            logging=False,
        )
    assert out is None


def test_keypointConfidenceOneVideo_cacheHit_skipsProcessing(videoPath):
    cache = {
        f"yolo11l-pose_{videoPath.name}": {
            "left_ankle_conf": 0.9,
            "model": "yolo11l-pose",
        }
    }
    model = MagicMock()

    out = _keypointConfidenceOneVideo(
        model=model,
        path=videoPath,
        name="yolo11l-pose",
        keypointIndexes=KEYPOINTINDEXES,
        logging=False,
        cache=cache,
    )
    model.assert_not_called()
    assert out == cache[f"yolo11l-pose_{videoPath.name}"]


def test_keypointConfidenceOneVideo_detectionRate_isDetectionsDividedByFrames(
    videoPath,
):
    person = MagicMock()
    person.__getitem__ = lambda self, idx: MagicMock(__getitem__=lambda s, i: 0.8)
    result = MagicMock()
    result.keypoints = MagicMock()
    result.keypoints.data = [person]

    model = MagicMock()
    model.side_effect = lambda *a, **kw: iter(
        [result, result]
    )  # 2 frames, 1 detection each

    with patch("pose.inference.cv2.VideoCapture") as mockCap:
        mockCap.return_value.get.return_value = 2
        out = _keypointConfidenceOneVideo(
            model=model,
            path=videoPath,
            name="yolo11l-pose",
            keypointIndexes=KEYPOINTINDEXES,
            logging=False,
        )
    assert out["detection_rate"] == pytest.approx(1.0)


def test_keypointConfidenceOneVideo_meanConfidence_isCorrect(videoPath):
    confs = [0.6, 0.8]
    results = []
    for c in confs:
        person = MagicMock()
        person.__getitem__ = lambda self, idx, c=c: MagicMock(
            __getitem__=lambda s, i: c
        )
        r = MagicMock()
        r.keypoints = MagicMock()
        r.keypoints.data = [person]
        results.append(r)

    model = MagicMock()
    model.side_effect = lambda *a, **kw: iter(results)

    with patch("pose.inference.cv2.VideoCapture") as mockCap:
        mockCap.return_value.get.return_value = 2
        out = _keypointConfidenceOneVideo(
            model=model,
            path=videoPath,
            name="yolo11l-pose",
            keypointIndexes=KEYPOINTINDEXES,
            logging=False,
        )
    assert out["left_ankle_conf"] == pytest.approx(0.7)


def test_keypointConfidenceOneVideo_cacheMiss_writesToCache(videoPath, cachePath):
    person = MagicMock()
    person.__getitem__ = lambda self, idx: MagicMock(__getitem__=lambda s, i: 0.9)
    result = MagicMock()
    result.keypoints = MagicMock()
    result.keypoints.data = [person]

    model = MagicMock()
    model.side_effect = lambda *a, **kw: iter([result])

    cache = {}
    with patch("pose.inference.cv2.VideoCapture") as mockCap:
        mockCap.return_value.get.return_value = 1
        _keypointConfidenceOneVideo(
            model=model,
            path=videoPath,
            name="yolo11l-pose",
            keypointIndexes=KEYPOINTINDEXES,
            logging=False,
            cache=cache,
            cachePath=cachePath,
        )
    assert f"yolo11l-pose_{videoPath.name}" in cache
    assert cachePath.exists()


def test_keypointConfidenceFromVideos_allVideosNoDetections_returnsNone(tmp_path):
    paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for p in paths:
        p.touch()

    with patch("pose.inference._keypointConfidenceOneVideo", return_value=None):
        out = keypointConfidenceFromVideos(
            model=MagicMock(),
            paths=paths,
            name="yolo11l-pose",
            keypointIndexes=KEYPOINTINDEXES,
            logging=False,
        )
    assert out is None


def test_keypointConfidenceFromVideos_loadsCacheFromDiskWhenCacheIsNone(
    tmp_path, cachePath
):
    cached = {
        "yolo11l-pose_a.mp4": {
            "left_ankle_conf": 0.9,
            "right_ankle_conf": 0.8,
            "detection_rate": 1.0,
            "model": "yolo11l-pose",
        }
    }
    cachePath.write_text(json.dumps(cached))

    paths = [tmp_path / "a.mp4"]
    for p in paths:
        p.touch()

    model = MagicMock()
    with patch(
        "pose.inference._keypointConfidenceOneVideo",
        wraps=lambda **kw: kw["cache"].get("yolo11l-pose_a.mp4"),
    ):
        out = keypointConfidenceFromVideos(
            model=model,
            paths=paths,
            name="yolo11l-pose",
            keypointIndexes=KEYPOINTINDEXES,
            logging=False,
            cache=None,
            cachePath=cachePath,
        )
    assert out is not None
    model.assert_not_called()
