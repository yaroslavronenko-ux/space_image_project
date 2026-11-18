"""Unit-тести на малих масивах (3x3), значення обчислені вручну (критерій C3).

Запуск: python -m pytest -q
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from src.adjustments import erosion, histogram_equalization
from src.filters import (convolve2d, convolve_separable, gaussian_blur, gaussian_kernel,
                         gaussian_kernel_1d, sobel_gradients)
from src.metrics import contrast, entropy, mean_intensity, mse, psnr, ssim


def test_gaussian_kernel_3x3():
    """Ядро 3x3 (sigma=1): нормоване, симетричне, центр — максимум."""
    kernel = gaussian_kernel(3, 1.0)
    edge = math.exp(-0.5)
    corner = math.exp(-1.0)
    total = 1 + 4 * edge + 4 * corner
    assert kernel.sum() == pytest.approx(1.0)
    assert kernel[1, 1] == pytest.approx(1 / total)
    assert kernel[0, 1] == pytest.approx(edge / total)
    assert kernel[0, 0] == pytest.approx(corner / total)


def test_gaussian_blur_constant_image_unchanged():
    """Розмиття однотонного зображення не змінює його."""
    image = np.full((3, 3), 100, dtype=np.uint8)
    assert np.array_equal(gaussian_blur(image, 3, 1.0), image)


def test_gaussian_blur_center_impulse():
    """Імпульс 255 у центрі 3x3: центр стає 255 * K[1,1]."""
    image = np.zeros((3, 3), dtype=np.uint8)
    image[1, 1] = 255
    kernel = gaussian_kernel(3, 1.0)
    result = gaussian_blur(image, 3, 1.0)
    assert result[1, 1] == round(255 * kernel[1, 1])


def test_convolution_box_kernel():
    """Згортка з ядром 3x3 з одиниць у центрі = сума 9 сусідів."""
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    result = convolve2d(image, np.ones((3, 3)))
    assert result[1, 1] == pytest.approx(36.0)


def test_separable_matches_2d_convolution():
    """Оптимізація: два 1D-проходи дають той самий результат, що й 2D-згортка."""
    image = np.array([[10, 50, 90],
                      [20, 200, 30],
                      [70, 0, 255]], dtype=np.float64)
    kernel_1d = gaussian_kernel_1d(3, 1.0)
    expected = convolve2d(image, gaussian_kernel(3, 1.0))
    assert np.allclose(convolve_separable(image, kernel_1d), expected)


def test_sobel_vertical_edge():
    """Вертикальна межа [0,0,255]: Gx у центрі = 4*255, Gy = 0."""
    image = np.array([[0, 0, 255],
                      [0, 0, 255],
                      [0, 0, 255]], dtype=np.float64)
    gx, gy, magnitude = sobel_gradients(image)
    assert gx[1, 1] == pytest.approx(1020.0)
    assert gy[1, 1] == pytest.approx(0.0)
    assert magnitude[1, 1] == pytest.approx(1020.0)


def test_sobel_horizontal_edge():
    """Горизонтальна межа: Gy у центрі = 4*255, Gx = 0."""
    image = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [255, 255, 255]], dtype=np.float64)
    gx, gy, _ = sobel_gradients(image)
    assert gx[1, 1] == pytest.approx(0.0)
    assert gy[1, 1] == pytest.approx(1020.0)


def test_histogram_equalization_3x3():
    """Рівні 10,20,30 по 3 пікселі: CDF = 3,6,9 -> 0,128,255."""
    image = np.array([[10, 10, 10],
                      [20, 20, 20],
                      [30, 30, 30]], dtype=np.uint8)
    expected = np.array([[0, 0, 0],
                         [128, 128, 128],
                         [255, 255, 255]], dtype=np.uint8)
    assert np.array_equal(histogram_equalization(image), expected)


def test_histogram_equalization_constant_image():
    """Однотонне зображення не змінюється (немає ділення на нуль)."""
    image = np.full((3, 3), 77, dtype=np.uint8)
    assert np.array_equal(histogram_equalization(image), image)


def test_erosion_removes_single_bright_pixel():
    """Ерозія 3x3 видаляє ізольований яскравий піксель («зірку»)."""
    image = np.zeros((3, 3), dtype=np.uint8)
    image[1, 1] = 255
    assert np.array_equal(erosion(image, 3), np.zeros((3, 3), dtype=np.uint8))


def test_erosion_grayscale_minimum():
    """Центр після ерозії = мінімум усього вікна 3x3."""
    image = np.array([[9, 8, 7],
                      [6, 5, 4],
                      [3, 2, 1]], dtype=np.uint8)
    result = erosion(image, 3)
    assert result[1, 1] == 1
    assert result[0, 0] == 5


def test_statistics_3x3():
    """Середнє, контраст і ентропія для [0..8] обчислені вручну."""
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    assert mean_intensity(image) == pytest.approx(4.0)
    assert contrast(image) == pytest.approx(math.sqrt(60 / 9))
    assert entropy(image) == pytest.approx(math.log2(9))


def test_mse_and_psnr_3x3():
    """Один піксель відрізняється на 3: MSE = 9/9 = 1, PSNR = 20*log10(255)."""
    first = np.zeros((3, 3), dtype=np.uint8)
    second = first.copy()
    second[0, 0] = 3
    assert mse(first, second) == pytest.approx(1.0)
    assert psnr(first, second) == pytest.approx(20 * math.log10(255))
    assert psnr(first, first) == float("inf")


def test_ssim_identical_and_different():
    """SSIM(x, x) = 1; для інвертованого зображення SSIM < 1."""
    image = np.arange(9, dtype=np.uint8).reshape(3, 3) * 30
    assert ssim(image, image) == pytest.approx(1.0)
    assert ssim(image, 255 - image) < 0.5
