"""Статистичні характеристики та метрики якості зображень (MSE, PSNR, SSIM)."""

import math

import numpy as np

from src.filters import convolve2d, gaussian_kernel
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
