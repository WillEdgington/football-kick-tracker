from unittest.mock import MagicMock, patch

import pytest

from ball.preannotate import batchCVATYOLOBallPreannotation


@pytest.fixture
def mockDependencies():
    with (
        patch("ball.preannotate.getVideoInfo") as mInfo,
        patch("ball.preannotate.getRawVideoYOLOBall") as mRaw,
        patch("ball.preannotate.toCVATVideoXML") as mXml,
        patch("ball.preannotate.saveText") as mSave,
    ):

        mInfo.return_value = {"frame_count": 100}
        mRaw.return_value = [{"frame": 0, "detections": []}]
        mXml.return_value = "<xml></xml>"

        yield (mInfo, mRaw, mXml, mSave)


def test_batchCVATYOLOBallPreannotation_skipsExistingFiles(tmp_path, mockDependencies):
    mInfo, mRaw, mXml, mSave = mockDependencies

    videoPath = tmp_path / "vid1.mp4"
    xmlPath = tmp_path / "vid1_ball.xml"
    xmlPath.touch()

    batchCVATYOLOBallPreannotation(
        model=MagicMock(),
        videoPaths=[videoPath],
        outputDir=tmp_path,
        overwrite=False,
        logging=False,
    )

    # Should have skipped processing
    mRaw.assert_not_called()
    mSave.assert_not_called()


def test_batchCVATYOLOBallPreannotation_overwritesWhenRequested(
    tmp_path, mockDependencies
):
    mInfo, mRaw, mXml, mSave = mockDependencies
    videoPath = tmp_path / "vid1.mp4"
    (tmp_path / "vid1_ball.xml").touch()

    batchCVATYOLOBallPreannotation(
        model=MagicMock(),
        videoPaths=[videoPath],
        outputDir=tmp_path,
        overwrite=True,
        logging=False,
    )

    # Should have processed despite the file existing
    mRaw.assert_called_once()
    mSave.assert_called_once()


def test_batchCVATYOLOBallPreannotation_continuesOnIndividualFailure(
    tmp_path, mockDependencies
):
    mInfo, mRaw, mXml, mSave = mockDependencies

    vid1, vid2 = tmp_path / "fail.mp4", tmp_path / "pass.mp4"

    # Force failure on the first video
    mRaw.side_effect = [Exception("GPU OOM"), [{"frame": 0}]]

    batchCVATYOLOBallPreannotation(
        model=MagicMock(), videoPaths=[vid1, vid2], outputDir=tmp_path, logging=False
    )

    # Verify it attempted both and saved the successful one
    assert mRaw.call_count == 2
    mSave.assert_called_once()
