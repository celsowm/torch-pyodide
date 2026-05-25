from __future__ import annotations

from torch.distributions import Bernoulli, Categorical, Normal, Uniform


class _FakeTensor:
    def __init__(self, value: str) -> None:
        self.value = value

    def __add__(self, other: object) -> "_FakeTensor":
        return _FakeTensor(f"({self.value}+{_value(other)})")

    def __sub__(self, other: object) -> "_FakeTensor":
        return _FakeTensor(f"({self.value}-{_value(other)})")

    def __rsub__(self, other: object) -> "_FakeTensor":
        return _FakeTensor(f"({_value(other)}-{self.value})")

    def __mul__(self, other: object) -> "_FakeTensor":
        return _FakeTensor(f"({self.value}*{_value(other)})")

    def __truediv__(self, other: object) -> "_FakeTensor":
        return _FakeTensor(f"({self.value}/{_value(other)})")


def _value(value: object) -> str:
    return value.value if isinstance(value, _FakeTensor) else str(value)


def test_distribution_properties_match_stored_tensors():
    normal = Normal.__new__(Normal)
    normal.loc = _FakeTensor("loc")
    normal.scale = _FakeTensor("scale")

    assert normal.mean is normal.loc
    assert normal.stddev is normal.scale
    assert normal.variance.value == "(scale*scale)"

    categorical = Categorical.__new__(Categorical)
    categorical._probs = _FakeTensor("probs")

    assert categorical.probs is categorical._probs


def test_distribution_moment_properties_are_available():
    uniform = Uniform.__new__(Uniform)
    uniform.low = _FakeTensor("low")
    uniform.high = _FakeTensor("high")

    assert uniform.mean.value == "((low+high)*0.5)"
    assert uniform.variance.value == "(((high-low)*(high-low))/12.0)"

    bernoulli = Bernoulli.__new__(Bernoulli)
    bernoulli.probs = _FakeTensor("probs")

    assert bernoulli.mean is bernoulli.probs
    assert bernoulli.variance.value == "(probs*(1.0-probs))"
