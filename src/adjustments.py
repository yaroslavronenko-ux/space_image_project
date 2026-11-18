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


def erosion(image, kernel_size=3, iterations=1):
    """Морфологічна ерозія квадратним структурним елементом.

    Для напівтонових зображень ерозія — це мінімум у вікні
    ``kernel_size x kernel_size``; для бінарних (0/255) результат
    збігається з класичною бінарною ерозією. Краї доповнюються
    максимальним значенням (255), щоб не створювати хибне «зношування»
    біля меж кадру.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size має бути непарним додатним числом")
    result = np.asarray(image, dtype=np.uint8).copy()
    pad = kernel_size // 2
    height, width = result.shape
    for _ in range(max(1, int(iterations))):
        padded = np.pad(result, pad, mode="constant", constant_values=255)
        eroded = np.full((height, width), 255, dtype=np.uint8)
        for row in range(kernel_size):
            for col in range(kernel_size):
                window = padded[row:row + height, col:col + width]
                eroded = np.minimum(eroded, window)
        result = eroded
    return result
