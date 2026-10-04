import numpy as np


def mse(pred, true):
    return float(np.mean((pred - true) ** 2))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def metric(pred, true, names=None):
    names = names or ['mse', 'mae']
    mapping = {
        'mse': mse(pred, true),
        'mae': mae(pred, true),
    }
    return {name: mapping[name] for name in names}
