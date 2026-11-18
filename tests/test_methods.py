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
from src.filters import convolve2d, gaussian_blur, gaussian_kernel, sobel_gradients
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
