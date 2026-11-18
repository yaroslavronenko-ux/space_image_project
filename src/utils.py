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
