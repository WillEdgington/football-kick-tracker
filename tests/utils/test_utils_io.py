import json

import pytest

from utils.io import loadJSON, saveJSON


@pytest.fixture
def tmpPath(tmp_path):
    return tmp_path / "test.json"


def test_saveJSON_writesValidJSONToDisk(tmpPath):
    saveJSON({"key": "value"}, tmpPath)
    with open(tmpPath, "r") as f:
        data = json.load(f)
    assert data == {"key": "value"}


def test_saveJSON_cleansTmpFileOnSuccess(tmpPath):
    saveJSON({"key": "value"}, tmpPath)
    assert not tmpPath.with_suffix(".tmp").exists()


def test_saveJSON_cleansTmpFileOnFailure(tmp_path):
    path = tmp_path / "test.json"
    with pytest.raises(Exception):
        saveJSON(object(), path)
    assert not path.with_suffix(".tmp").exists()


def test_loadJSON_returnsNoneForNonexistentPath(tmp_path):
    result = loadJSON(tmp_path / "missing.json")
    assert result is None


def test_loadJSON_roundTripsWithSaveJSON(tmpPath):
    data = {"a": 1, "b": [1, 2, 3]}
    saveJSON(data, tmpPath)
    assert loadJSON(tmpPath) == data
