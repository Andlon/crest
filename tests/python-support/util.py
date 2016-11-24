import numpy as np


def estimate_slope(x, y):
    poly_coeff = np.polyfit(np.log(x), np.log(y), 1)
    return poly_coeff[0]


def is_strictly_decreasing(vector):
    differences = np.diff(vector)
    return np.all(differences < 0)


def errors_within_relative_tolerance(actual, expected, tolerance=1e-6):
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    assert np.all(actual > 0)
    assert np.all(expected > 0)

    diff = np.abs(actual - expected)
    return np.all(diff < tolerance * expected)
