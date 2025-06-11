import pytest

from unmute.stt.exponential_moving_average import ExponentialMovingAverage


def test_ema():
    ema = ExponentialMovingAverage(attack_time=0.1, release_time=0.5)
    ema.update(dt=1.0, new_value=0.0)
    assert ema.value == 0.0

    ema.update(dt=0.1, new_value=1.0)
    assert ema.value == pytest.approx(0.5)

    ema.update(dt=0.25, new_value=0.0)
    ema.update(dt=0.25, new_value=0.0)
    assert ema.value == pytest.approx(0.25)

    # Should work even with values different than 0 and 1
    ema.update(dt=0.1, new_value=0.75)
    assert ema.value == pytest.approx(0.5)

    ema.update(dt=1e9, new_value=1.0)
    assert ema.value == pytest.approx(1.0)
