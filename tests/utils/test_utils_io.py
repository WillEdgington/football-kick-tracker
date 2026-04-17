import json

import pytest

from utils.io import loadJSON, saveJSON, saveText


@pytest.fixture
def jsonPath(tmp_path):
    return tmp_path / "test.json"


@pytest.fixture
def txtPath(tmp_path):
    return tmp_path / "test.txt"


def test_saveJSON_writesValidJSONToDisk(jsonPath):
    saveJSON({"key": "value"}, jsonPath)
    with open(jsonPath, "r") as f:
        data = json.load(f)
    assert data == {"key": "value"}


def test_saveJSON_cleansTmpFileOnSuccess(jsonPath):
    saveJSON({"key": "value"}, jsonPath)
    assert not jsonPath.with_suffix(".tmp").exists()


def test_saveJSON_cleansTmpFileOnFailure(tmp_path):
    path = tmp_path / "test.json"
    with pytest.raises(Exception):
        saveJSON(object(), path)
    assert not path.with_suffix(".tmp").exists()


def test_loadJSON_returnsNoneForNonexistentPath(tmp_path):
    result = loadJSON(tmp_path / "missing.json")
    assert result is None


def test_loadJSON_roundTripsWithSaveJSON(jsonPath):
    data = {"a": 1, "b": [1, 2, 3]}
    saveJSON(data, jsonPath)
    assert loadJSON(jsonPath) == data


def test_saveText_writesCorrectContent(txtPath):
    content = "testing, testing, 123..."
    saveText(content, txtPath)
    assert txtPath.read_text(encoding="utf-8") == content


def test_saveText_cleansTmpFileOnFailure(tmp_path):
    path = tmp_path / "fail.txt"
    with pytest.raises(Exception):
        saveText(None, path)
    assert not path.with_suffix(".tmp").exists()
