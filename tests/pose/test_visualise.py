from unittest.mock import MagicMock

import numpy as np
import pytest

from pose.visualise import drawKeypoints


@pytest.fixture
def blankFrame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mockResults():
    results = MagicMock()
    results[0].keypoints = None
    return results


def test_drawKeypoints_returnsNumpyArray(blankFrame, mockResults):
    output = drawKeypoints(blankFrame, mockResults)
    assert isinstance(output, np.ndarray)


def test_drawKeypoints_outputShapeMatchesInput(blankFrame, mockResults):
    output = drawKeypoints(blankFrame, mockResults)
    assert output.shape == blankFrame.shape


def test_drawKeypoints_missingColourKey_raisesAssertionError(blankFrame, mockResults):
    with pytest.raises(AssertionError):
        drawKeypoints(blankFrame, mockResults, colours={"test": (255, 255, 0)})


def test_drawKeypoints_negativeConfThreshold_raisesAssertionError(
    blankFrame, mockResults
):
    with pytest.raises(AssertionError):
        drawKeypoints(blankFrame, mockResults, confThreshold=-1)


def test_drawKeypoints_invalidSkeletonPairKey_raisesAssertionError(
    blankFrame, mockResults
):
    with pytest.raises(AssertionError):
        drawKeypoints(blankFrame, mockResults, skeletonPairs=[("test", "123")])


def test_drawKeypoints_inputIsNotMutated(mockResults):
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    frameCopy = np.ones((100, 100, 3), dtype=np.uint8) * 128
    drawKeypoints(frame, mockResults)
    assert np.array_equal(frameCopy, frame)
