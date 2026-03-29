from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pose.annotate import (
    _getModelByName,
    _getOutputPaths,
    annotateVideos,
    annotateVideosForTopNModels,
)


@pytest.fixture
def sourcePaths(tmp_path):
    paths = [tmp_path / "clip1.mov", tmp_path / "clip2.mp4"]
    for p in paths:
        p.touch()
    return paths


@pytest.fixture
def outputDir(tmp_path):
    return tmp_path / "output"


@pytest.fixture
def mockModel():
    return MagicMock()


@pytest.fixture
def modelList(mockModel):
    return [("yolo11l-pose", mockModel), ("yolo11m-pose", MagicMock())]


def test_getOutputPaths_movInput_returnsMp4Extension(sourcePaths, outputDir):
    paths = _getOutputPaths("yolo11l-pose", sourcePaths, outputDir)
    assert all(p.suffix == ".mp4" for p in paths)


def test_getOutputPaths_mp4Input_returnsMp4Extension(sourcePaths, outputDir):
    paths = _getOutputPaths("yolo11l-pose", sourcePaths, outputDir)
    assert paths[1].suffix == ".mp4"


def test_getOutputPaths_outputsAreUnderModelSubdir(sourcePaths, outputDir):
    paths = _getOutputPaths("yolo11l-pose", sourcePaths, outputDir)
    assert all(p.parent == outputDir / "yolo11l-pose" for p in paths)


def test_getModelByName_validName_returnsModel(modelList, mockModel):
    result = _getModelByName("yolo11l-pose", modelList, logging=False)
    assert result is mockModel


def test_getModelByName_unknownName_returnsNone(modelList):
    result = _getModelByName("yolo26m-pose", modelList, logging=False)
    assert result is None


def test_annotateVideos_existingOutput_skipsWhenOverwriteFalse(
    sourcePaths, outputDir, mockModel
):
    outputPaths = _getOutputPaths("yolo11l-pose", sourcePaths, outputDir)
    for p in outputPaths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    with patch("pose.annotate.annotateVideo") as mockAnnotate:
        annotateVideos(
            model=mockModel,
            name="yolo11l-pose",
            sourcePaths=sourcePaths,
            outputDir=outputDir,
            logging=False,
            overwrite=False,
        )
    mockAnnotate.assert_not_called()


def test_annotateVideos_existingOutput_overwritesWhenOverwriteTrue(
    sourcePaths, outputDir, mockModel
):
    outputPaths = _getOutputPaths("yolo11l-pose", sourcePaths, outputDir)
    for p in outputPaths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    with patch("pose.annotate.annotateVideo") as mockAnnotate:
        annotateVideos(
            model=mockModel,
            name="yolo11l-pose",
            sourcePaths=sourcePaths,
            outputDir=outputDir,
            logging=False,
            overwrite=True,
        )
    assert mockAnnotate.call_count == len(sourcePaths)


def test_annotateVideosForTopNModels_nExceedsModelCount_raisesAssertionError(
    sourcePaths, outputDir, modelList
):
    df = pd.DataFrame({"model": ["yolo11l-pose", "yolo11m-pose"]})
    with pytest.raises(AssertionError):
        annotateVideosForTopNModels(
            df=df,
            models=modelList,
            sourcePaths=sourcePaths,
            outputDir=outputDir,
            logging=False,
            n=5,
        )
