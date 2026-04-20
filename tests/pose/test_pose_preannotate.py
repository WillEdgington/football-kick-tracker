from unittest.mock import MagicMock, patch

import pytest

from pose.preannotate import batchCVATYOLOPosePreannotation


@pytest.fixture
def mockDependencies():
    with (
        patch("pose.preannotate.getVideoInfo") as mInfo,
        patch("pose.preannotate.getRawVideoYOLOPose") as mRaw,
        patch("pose.preannotate.toCVATVideoXML") as mXml,
        patch("pose.preannotate.saveText") as mSave,
    ):

        mInfo.return_value = {"frame_count": 100}
        mRaw.return_value = [{"frame": 0, "detections": []}]
        mXml.return_value = "<xml></xml>"

        yield (mInfo, mRaw, mXml, mSave)


def test_batchCVATYOLOPosePreannotation_skipsExistingFiles(tmp_path, mockDependencies):
    mInfo, mRaw, mXml, mSave = mockDependencies

    videoPath = tmp_path / "vid1.mp4"
    xmlPath = tmp_path / "vid1_pose.xml"
    xmlPath.touch()

    batchCVATYOLOPosePreannotation(
        model=MagicMock(),
        videoPaths=[videoPath],
        outputDir=tmp_path,
        overwrite=False,
        logging=False,
    )

    # Should have skipped processing
    mRaw.assert_not_called()
    mSave.assert_not_called()


def test_batchCVATYOLOPosePreannotation_overwritesWhenRequested(
    tmp_path, mockDependencies
):
    mInfo, mRaw, mXml, mSave = mockDependencies
    videoPath = tmp_path / "vid1.mp4"
    (tmp_path / "vid1_pose.xml").touch()

    batchCVATYOLOPosePreannotation(
        model=MagicMock(),
        videoPaths=[videoPath],
        outputDir=tmp_path,
        overwrite=True,
        logging=False,
    )

    # Should have processed despite the file existing
    mRaw.assert_called_once()
    mSave.assert_called_once()


def test_batchCVATYOLOPosePreannotation_continuesOnIndividualFailure(
    tmp_path, mockDependencies
):
    mInfo, mRaw, mXml, mSave = mockDependencies

    vid1, vid2 = tmp_path / "fail.mp4", tmp_path / "pass.mp4"

    # Force failure on the first video
    mRaw.side_effect = [Exception("GPU OOM"), [{"frame": 0}]]

    batchCVATYOLOPosePreannotation(
        model=MagicMock(), videoPaths=[vid1, vid2], outputDir=tmp_path, logging=False
    )

    # Verify it attempted both and saved the successful one
    assert mRaw.call_count == 2
    mSave.assert_called_once()
