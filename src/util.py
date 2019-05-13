import numpy as np


def exponential_decay(value: float, min_value: float,
                      decay_rate: float) -> float:
    return min_value + (value - min_value) * np.exp(-decay_rate)