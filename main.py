"""Головний скрипт: обробка вибірки зображень космічних апаратів.

Приклади запуску:
    python main.py
    python main.py --method gaussian sobel
    python main.py --config configs/config.json --input data/raw
"""

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np

from src.adjustments import erosion, histogram_equalization
from src.filters import gaussian_blur, sobel_edges
from src.metrics import image_statistics, mse, quality_metrics
from src.utils import (add_gaussian_noise, list_images, load_config,
                       read_image, save_image)

ALL_METHODS = ("gaussian", "sobel", "histogram_equalization", "erosion", "statistics")

CSV_FIELDS = (
    "image", "method", "input", "time_ms",
    "mse", "psnr", "ssim", "mse_noisy_baseline",
    "mean", "variance", "contrast", "entropy",
)

LOGGER = logging.getLogger("space_image_project")


def parse_args(argv=None):
    """Розбирає аргументи командного рядка."""
    parser = argparse.ArgumentParser(
        description="Ручна обробка зображень космічних апаратів (5 класичних методів).")
    parser.add_argument("--config", default="configs/config.json",
                        help="шлях до JSON-конфігурації")
    parser.add_argument("--method", nargs="+", choices=ALL_METHODS,
                        help="які методи запускати (за замовчуванням — з конфігу)")
    parser.add_argument("--input", help="каталог з вихідними зображеннями")
    parser.add_argument("--output", help="каталог для оброблених зображень")
    parser.add_argument("--report", help="шлях до CSV з метриками")
    return parser.parse_args(argv)


def fmt(value):
    """Форматує число для CSV (inf зберігається як рядок 'inf')."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def make_row(context, method, elapsed_ms, measures):
    """Формує рядок CSV для одного запуску методу.

    Args:
        context: словник з ключами ``name`` та ``source`` (вхід методу).
        measures: словник з необов'язковими ключами ``quality`` (MSE/PSNR/SSIM),
            ``stats`` (статистики) та ``baseline`` (MSE зашумленого входу).
    """
    quality = measures.get("quality") or {}
    stats = measures.get("stats") or {}
    baseline = measures.get("baseline")
    return {
        "image": context["name"],
        "method": method,
        "input": context.get("source", "original"),
        "time_ms": f"{elapsed_ms:.2f}",
        "mse": fmt(quality.get("mse")),
        "psnr": fmt(quality.get("psnr")),
        "ssim": fmt(quality.get("ssim")),
        "mse_noisy_baseline": fmt(baseline),
        "mean": fmt(stats.get("mean")),
        "variance": fmt(stats.get("variance")),
        "contrast": fmt(stats.get("contrast")),
        "entropy": fmt(stats.get("entropy")),
    }
