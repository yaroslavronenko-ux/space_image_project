"""Ручна реалізація гістограмного вирівнювання та морфологічної ерозії."""

import numpy as np


def histogram(image, levels=256):
    """Підраховує гістограму яскравостей (кількість пікселів кожного рівня)."""
    flat = np.asarray(image, dtype=np.int64).ravel()
    hist = np.zeros(levels, dtype=np.int64)
    # np.add.at — накопичення лічильників без готових функцій гістограми.
    np.add.at(hist, np.clip(flat, 0, levels - 1), 1)
    return hist


def cumulative_distribution(hist):
    """Кумулятивна функція розподілу (CDF): cdf[k] = sum_{i<=k} hist[i]."""
    cdf = np.zeros_like(hist)
    running = 0
    for index, count in enumerate(hist):
        running += count
        cdf[index] = running
    return cdf
