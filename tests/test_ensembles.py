import pytest

from src.trading.ensembles import parse_weight_spec, simple_average, weighted_average


def test_simple_average():
    values = [0.2, 0.4, 0.6]
    assert simple_average(values) == pytest.approx(0.4)


def test_weighted_average_with_partial_weights():
    values = {"xgb": 0.4, "lstm": 0.6, "transformer": 0.8}
    weights = {"xgb": 1.0, "transformer": 3.0}
    result = weighted_average(values, weights)
    expected = (0.4 * 1.0 + 0.8 * 3.0) / (1.0 + 3.0)
    assert result == pytest.approx(expected)


def test_parse_weight_spec_lowercases_keys():
    spec = "Transformer:2,lStM:1,xgb:0.5"
    weights = parse_weight_spec(spec)
    assert weights == {"transformer": 2.0, "lstm": 1.0, "xgb": 0.5}


def test_weighted_average_zero_weight_raises():
    values = {"xgb": 0.4}
    weights = {"xgb": 0.0}
    with pytest.raises(ValueError):
        weighted_average(values, weights)
