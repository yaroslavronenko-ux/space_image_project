"""Статистичні характеристики та метрики якості зображень (MSE, PSNR, SSIM)."""

import math

import numpy as np

from src.filters import convolve_separable, gaussian_kernel_1d
from src.adjustments import histogram


def mean_intensity(image):
    """Середня інтенсивність: mu = (1/N) * sum I(i)."""
    values = np.asarray(image, dtype=np.float64)
    return float(values.sum() / values.size)


def variance(image):
    """Дисперсія: sigma^2 = (1/N) * sum (I(i) - mu)^2."""
    values = np.asarray(image, dtype=np.float64)
    mu = mean_intensity(values)
    return float(((values - mu) ** 2).sum() / values.size)


def contrast(image):
    """Контраст як стандартне відхилення інтенсивностей (RMS-контраст)."""
    return math.sqrt(variance(image))


def entropy(image, bins=256):
    """Ентропія Шеннона: H = -sum p_k * log2(p_k) по ненульових p_k."""
    hist = histogram(image, bins).astype(np.float64)
    probabilities = hist / hist.sum()
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def image_statistics(image, bins=256):
    """Повертає словник статистичних характеристик зображення."""
    return {
        "mean": mean_intensity(image),
        "variance": variance(image),
        "contrast": contrast(image),
        "entropy": entropy(image, bins),
    }


def mse(reference, processed):
    """Середньоквадратична помилка: MSE = (1/N) * sum (I - I')^2."""
    ref = np.asarray(reference, dtype=np.float64)
    proc = np.asarray(processed, dtype=np.float64)
    if ref.shape != proc.shape:
        raise ValueError(f"Розміри зображень не збігаються: {ref.shape} vs {proc.shape}")
    return float(((ref - proc) ** 2).sum() / ref.size)


def psnr(reference, processed, max_value=255.0):
    """Пікове відношення сигнал/шум: PSNR = 10 * log10(MAX^2 / MSE).

    Для ідентичних зображень (MSE = 0) повертає ``float('inf')``.
    """
    error = mse(reference, processed)
    if error == 0:
        return float("inf")
    return float(10.0 * math.log10(max_value ** 2 / error))


SSIM_DEFAULTS = {"window_size": 11, "sigma": 1.5, "k1": 0.01, "k2": 0.03,
                 "max_value": 255.0}


def _local_moments(x, y, window_1d):
    """Локальні середні, дисперсії та коваріація у гаусовому вікні.

    Гаусове вікно сепарабельне, тож кожне усереднення виконується двома
    1D-проходами (``convolve_separable``) замість 2D-згортки 11x11.
    """
    mu_x = convolve_separable(x, window_1d)
    mu_y = convolve_separable(y, window_1d)
    var_x = convolve_separable(x * x, window_1d) - mu_x ** 2
    var_y = convolve_separable(y * y, window_1d) - mu_y ** 2
    cov_xy = convolve_separable(x * y, window_1d) - mu_x * mu_y
    return mu_x, mu_y, var_x, var_y, cov_xy


def _window_size(requested, shape):
    """Непарний розмір вікна, що не перевищує меншу сторону зображення."""
    size = min(int(requested), *shape)
    if size % 2 == 0:
        size -= 1
    return max(size, 1)


def ssim(reference, processed, params=None):
    """Індекс структурної подібності (SSIM, Wang et al., 2004).

    Локальні середні, дисперсії та коваріація обчислюються гаусовим вікном
    за допомогою власної сепарабельної згортки ``convolve_separable``:

    ``SSIM = ((2*mu_x*mu_y + C1)(2*sigma_xy + C2)) /
    ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))``,

    де ``C1 = (k1*L)^2``, ``C2 = (k2*L)^2``. Повертається середнє значення
    карти SSIM. Для зображень, менших за вікно, розмір вікна зменшується.

    Args:
        reference, processed: зображення однакового розміру.
        params: словник з ключами window_size, sigma, k1, k2, max_value
            (відсутні ключі беруться з ``SSIM_DEFAULTS``).
    """
    cfg = {**SSIM_DEFAULTS, **(params or {})}
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(processed, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Розміри зображень не збігаються: {x.shape} vs {y.shape}")

    window = gaussian_kernel_1d(_window_size(cfg["window_size"], x.shape),
                                float(cfg["sigma"]))

    c1 = (cfg["k1"] * cfg["max_value"]) ** 2
    c2 = (cfg["k2"] * cfg["max_value"]) ** 2
    mu_x, mu_y, var_x, var_y, cov_xy = _local_moments(x, y, window)

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
        ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim_map.mean())


def quality_metrics(reference, processed, ssim_params=None):
    """Обчислює MSE, PSNR та SSIM однією функцією."""
    params = ssim_params or {}
    return {
        "mse": mse(reference, processed),
        "psnr": psnr(reference, processed),
        "ssim": ssim(reference, processed, params),
    }
