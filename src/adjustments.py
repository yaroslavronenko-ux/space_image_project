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


def histogram_equalization(image, levels=256):
    """Гістограмне вирівнювання через CDF.

    Формула перетворення рівня ``v``:
    ``h(v) = round((cdf(v) - cdf_min) / (N - cdf_min) * (L - 1))``,
    де ``N`` — кількість пікселів, ``cdf_min`` — мінімальне ненульове
    значення CDF, ``L`` — кількість рівнів.

    Для однотонного зображення (N == cdf_min) повертається копія входу.
    """
    image = np.asarray(image, dtype=np.uint8)
    hist = histogram(image, levels)
    cdf = cumulative_distribution(hist)
    total = image.size
    cdf_min = cdf[cdf > 0].min()
    if total == cdf_min:
        return image.copy()
    lut = np.rint((cdf - cdf_min) / (total - cdf_min) * (levels - 1))
    lut = np.clip(lut, 0, levels - 1).astype(np.uint8)
    return lut[image]
