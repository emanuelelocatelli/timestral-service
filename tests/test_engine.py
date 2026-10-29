import numpy as np

from engine import TimestralPredictor, direction_and_strength


def test_flat_series_is_neutral():
    context = np.full(20, 100.0)
    direction, strength = direction_and_strength(100.0, 100.0, context)
    assert direction == "NEUTRAL"
    assert strength == "none"


def test_jump_above_the_return_threshold_is_up():
    context = np.linspace(100.0, 101.0, 30)
    direction, strength = direction_and_strength(101.0, 110.0, context)
    assert direction == "UP"
    assert strength == "strong"


def test_batch_keeps_input_order_when_lengths_differ():
    predictor = TimestralPredictor()
    levels = (0.1, 0.5, 0.9)

    def fake_batch(batch, horizon):
        # Echo the last close as the median so each row is identifiable.
        rows = []
        for series in batch:
            median = np.full(horizon, series[-1])
            low = median - 1
            high = median + 1
            rows.append(np.stack([low, median, high], axis=1))
        return np.stack(rows), levels

    predictor._forecast_batch = fake_batch
    short = [1.0] * 8
    long = [2.0] * 12
    results = predictor.predict_many([short, long, short], prediction_horizon=4, interval_seconds=60)
    assert [item.forecast[-1] for item in results] == [1.0, 2.0, 1.0]
    assert results[1].label is None
    assert results[0].prediction_direction == "NEUTRAL"
    assert "q50" in results[0].quantiles
