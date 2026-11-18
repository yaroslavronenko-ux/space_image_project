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
