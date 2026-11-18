"""Допоміжні функції: конфіг, читання/збереження зображень, шум.

OpenCV використовується тут ВИКЛЮЧНО для декодування/кодування файлів
(cv2.imdecode / cv2.imencode). Перетворення в відтінки сірого та
нормалізація бітності реалізовані вручну.
"""

import json
import os

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MIN_SIDE = 3


def load_config(path):
    """Читає JSON-конфігурацію."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_images(directory):
    """Повертає відсортований список шляхів до зображень у каталозі."""
    if not os.path.isdir(directory):
        return []
    names = sorted(os.listdir(directory))
    return [os.path.join(directory, name) for name in names
            if name.lower().endswith(SUPPORTED_EXTENSIONS)]


def to_uint8(image):
    """Приводить масив довільної бітності до uint8 [0, 255]."""
    if image.dtype == np.uint8:
        return image
    data = image.astype(np.float64)
    if image.dtype == np.uint16:
        data = data / 257.0
    elif np.issubdtype(image.dtype, np.floating) and data.max() <= 1.0:
        data = data * 255.0
    return np.clip(np.rint(data), 0, 255).astype(np.uint8)


def to_grayscale(image):
    """Ручне перетворення в відтінки сірого (ITU-R BT.601).

    Y = 0.299 R + 0.587 G + 0.114 B. OpenCV зберігає канали як BGR.
    Альфа-канал (якщо є) відкидається.
    """
    if image.ndim == 2:
        return to_uint8(image)
    if image.ndim == 3 and image.shape[2] == 1:
        return to_uint8(image[:, :, 0])
    if image.ndim == 3 and image.shape[2] >= 3:
        data = to_uint8(image).astype(np.float64)
        blue, green, red = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        return np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    raise ValueError(f"Непідтримувана форма зображення: {image.shape}")


def read_image(path):
    """Читає зображення та повертає його у відтінках сірого (uint8).

    Підтримує шляхи з кирилицею (через np.fromfile + cv2.imdecode).

    Raises:
        ValueError: файл порожній, пошкоджений або замалий.
    """
    if os.path.getsize(path) == 0:
        raise ValueError("порожній файл")
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("файл пошкоджений або формат не підтримується")
    gray = to_grayscale(image)
    if min(gray.shape) < MIN_SIDE:
        raise ValueError(f"зображення замале: {gray.shape}")
    return gray


def save_image(path, image):
    """Зберігає зображення (підтримує шляхи з кирилицею)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extension = os.path.splitext(path)[1] or ".png"
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise IOError(f"Не вдалося закодувати {path}")
    buffer.tofile(path)


def add_gaussian_noise(image, sigma, rng):
    """Додає адитивний гаусів шум N(0, sigma^2) з фіксованим генератором."""
    noise = rng.normal(0.0, sigma, size=image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
