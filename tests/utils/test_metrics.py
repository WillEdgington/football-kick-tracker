import pandas as pd
import pytest

from utils.metrics import addCompositeScore, normaliseMetric


@pytest.fixture
def simpleDf():
    return pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.fixture
def multiColDf():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
        }
    )


def test_normaliseMetric_standard_outputMeanIsNearZero(simpleDf):
    result = normaliseMetric(simpleDf, "score", method="standard")
    assert abs(result.mean()) < 1e-6


def test_normaliseMetric_standard_outputStdIsNearOne(simpleDf):
    result = normaliseMetric(simpleDf, "score", method="standard")
    assert abs(result.std() - 1.0) < 1e-6


def test_normaliseMetric_minmax_outputMinIsZero(simpleDf):
    result = normaliseMetric(simpleDf, "score", method="minmax")
    assert result.min() == pytest.approx(0.0)


def test_normaliseMetric_minmax_outputMaxIsOne(simpleDf):
    result = normaliseMetric(simpleDf, "score", method="minmax")
    assert result.max() == pytest.approx(1.0)


def test_normaliseMetric_missingColumn_raisesValueError(simpleDf):
    with pytest.raises(ValueError):
        normaliseMetric(simpleDf, "missing")


def test_addCompositeScore_mismatchedKeys_raisesValueError(multiColDf):
    with pytest.raises(ValueError):
        addCompositeScore(
            multiColDf,
            weights={"a": 1.0},
            ascending={"a": False, "b": False},
        )


def test_addCompositeScore_outputColumnPresentInDf(multiColDf):
    result = addCompositeScore(multiColDf, weights={"a": 1.0}, ascending={"a": False})
    assert "composite_score" in result.columns


def test_addCompositeScore_customColName_isPresentInDf(multiColDf):
    result = addCompositeScore(
        multiColDf,
        weights={"a": 1.0},
        ascending={"a": False},
        colName="my_score",
    )
    assert "my_score" in result.columns


def test_addCompositeScore_higherRawValue_scoresHigherWhenAscendingFalse(multiColDf):
    result = addCompositeScore(multiColDf, weights={"a": 1.0}, ascending={"a": False})
    assert result["composite_score"].iloc[2] > result["composite_score"].iloc[0]


def test_addCompositeScore_doesNotMutateInputDf(multiColDf):
    colsBefore = list(multiColDf.columns)
    addCompositeScore(multiColDf, weights={"a": 1.0}, ascending={"a": False})
    assert list(multiColDf.columns) == colsBefore
