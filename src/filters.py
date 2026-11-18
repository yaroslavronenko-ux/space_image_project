"""Ручна реалізація фільтрів: згортка, фільтр Гауса, оператор Собеля.

Усі алгоритми написані на NumPy без використання функцій комп'ютерного
зору з OpenCV / scikit-image.
"""

import numpy as np

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float64)


def convolve2d(image, kernel):
    """Двовимірна згортка зображення з ядром (ручна реалізація).

    Замість подвійного циклу по пікселях використовується цикл по
    елементах ядра: для кожного зсуву (dy, dx) до результату додається
    зсунутий фрагмент доповненого зображення, помножений на коефіцієнт.
    Це математично еквівалентно класичній формулі
    ``G(x, y) = sum_i sum_j K(i, j) * I(x - i, y - j)``.

    Args:
        image: 2D масив (відтінки сірого).
        kernel: 2D ядро непарного розміру.

    Returns:
        Масив float64 того ж розміру, що й ``image``.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolve2d очікує 2D зображення та 2D ядро")
    k_h, k_w = kernel.shape
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError("Розміри ядра мають бути непарними")

    # Справжня згортка: ядро відображається на 180 градусів.
    flipped = kernel[::-1, ::-1]
    pad_h, pad_w = k_h // 2, k_w // 2
    # Дзеркальне доповнення країв; для зображень, менших за ядро, — повтор краю.
    mode = "reflect" if min(image.shape) > max(pad_h, pad_w) else "edge"
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)

    height, width = image.shape
    result = np.zeros((height, width), dtype=np.float64)
    for row in range(k_h):
        for col in range(k_w):
            coeff = flipped[row, col]
            if coeff != 0.0:
                result += coeff * padded[row:row + height, col:col + width]
    return result


def gaussian_kernel(size, sigma):
    """Будує нормоване ядро Гауса розміру ``size x size``.

    Формула: ``G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))``,
    після чого ядро ділиться на суму елементів (сума = 1).
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("Розмір ядра Гауса має бути непарним додатним числом")
    if sigma <= 0:
        raise ValueError("sigma має бути > 0")
    half = size // 2
    axis = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(image, size=5, sigma=1.0):
    """Фільтр Гауса: згортка з нормованим гаусовим ядром.

    Returns:
        Масив uint8 (значення округлені та обмежені [0, 255]).
    """
    blurred = convolve2d(image, gaussian_kernel(size, sigma))
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def sobel_gradients(image):
    """Обчислює градієнти Собеля Gx, Gy та модуль градієнта.

    Модуль: ``|G| = sqrt(Gx^2 + Gy^2)``.

    Returns:
        Кортеж ``(gx, gy, magnitude)`` у float64.
    """
    # Для Собеля використовуємо кореляцію (як у класичних підручниках):
    # тому заздалегідь відображаємо ядро, щоб convolve2d повернула його назад.
    gx = convolve2d(image, SOBEL_X[::-1, ::-1])
    gy = convolve2d(image, SOBEL_Y[::-1, ::-1])
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, magnitude


def sobel_edges(image, threshold=None):
    """Карта контурів Собеля.

    Модуль градієнта лінійно нормується до [0, 255]. Якщо задано
    ``threshold``, додатково повертається бінарна карта (255 — контур).

    Returns:
        Кортеж ``(magnitude_u8, binary_u8 | None)``.
    """
    _, _, magnitude = sobel_gradients(image)
    max_value = magnitude.max()
    if max_value > 0:
        normalized = magnitude * (255.0 / max_value)
    else:
        normalized = np.zeros_like(magnitude)
    magnitude_u8 = np.clip(np.rint(normalized), 0, 255).astype(np.uint8)
    binary = None
    if threshold is not None:
        binary = np.where(magnitude_u8 >= threshold, 255, 0).astype(np.uint8)
    return magnitude_u8, binary
