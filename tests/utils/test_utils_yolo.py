from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.yolo import loadYOLOModel


@patch("utils.yolo.YOLO")
def test_loadYOLOModel_callsYOLOWithCorrectPath(mock_yolo_class):
    testPath = Path("models/test_model.pt")
    loadYOLOModel(testPath)
    mock_yolo_class.assert_called_once_with(str(testPath))


def test_loadYOLOModel_returnsModelInstance():
    with patch("utils.yolo.YOLO") as mockyolo:
        mockInstance = MagicMock()
        mockyolo.return_value = mockInstance

        model = loadYOLOModel(Path("any.pt"))
        assert model == mockInstance
