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


def run_gaussian(ctx, config):
    """Фільтр Гауса застосовується до зашумленої копії (перевірка шумоподавлення)."""
    params = config.get("gaussian", {})
    source = ctx["noisy"] if ctx["noisy"] is not None else ctx["original"]
    result = gaussian_blur(source, int(params.get("kernel_size", 5)),
                           float(params.get("sigma", 1.0)))
    extra = {}
    if ctx["noisy"] is not None:
        extra["baseline"] = mse(ctx["original"], source)
    return result, "noisy" if ctx["noisy"] is not None else "original", extra


def run_sobel(ctx, config):
    """Оператор Собеля; бінарна карта контурів зберігається окремо."""
    params = config.get("sobel", {})
    result, binary = sobel_edges(ctx["original"], params.get("threshold"))
    extra = {}
    if binary is not None and params.get("save_binary", True):
        extra["binary"] = binary
    return result, "original", extra


def run_histogram(ctx, config):
    """Гістограмне вирівнювання через CDF."""
    levels = int(config.get("histogram_equalization", {}).get("levels", 256))
    return histogram_equalization(ctx["original"], levels), "original", {}


def run_erosion(ctx, config):
    """Морфологічна ерозія квадратним структурним елементом."""
    params = config.get("erosion", {})
    result = erosion(ctx["original"], int(params.get("kernel_size", 3)),
                     int(params.get("iterations", 1)))
    return result, "original", {}


IMAGE_METHODS = {
    "gaussian": run_gaussian,
    "sobel": run_sobel,
    "histogram_equalization": run_histogram,
    "erosion": run_erosion,
}


def prepare_context(path, config, rng, output_dir):
    """Читає зображення, створює зашумлену копію та зберігає обидва."""
    original = read_image(path)
    name = os.path.splitext(os.path.basename(path))[0]
    save_image(os.path.join(output_dir, "original", f"{name}.png"), original)
    noisy = None
    noise_cfg = config.get("noise", {})
    if noise_cfg.get("enabled", True):
        noisy = add_gaussian_noise(original, float(noise_cfg.get("sigma", 20.0)), rng)
        save_image(os.path.join(output_dir, "noisy", f"{name}.png"), noisy)
    return {"name": name, "original": original, "noisy": noisy}


def run_method(ctx, method, config, output_dir):
    """Виконує один метод, зберігає результат і повертає рядок CSV."""
    bins = config.get("statistics", {}).get("entropy_bins", 256)
    start = time.perf_counter()
    if method == "statistics":
        stats = image_statistics(ctx["original"], bins)
        elapsed = (time.perf_counter() - start) * 1000
        return make_row(ctx, method, elapsed, {"stats": stats})

    result, source, extra = IMAGE_METHODS[method](ctx, config)
    elapsed = (time.perf_counter() - start) * 1000
    save_image(os.path.join(output_dir, method, f"{ctx['name']}.png"), result)
    if "binary" in extra:
        save_image(os.path.join(output_dir, f"{method}_binary", f"{ctx['name']}.png"),
                   extra["binary"])
    measures = {
        "quality": quality_metrics(ctx["original"], result, config.get("ssim", {})),
        "stats": image_statistics(result, bins),
        "baseline": extra.get("baseline"),
    }
    return make_row({"name": ctx["name"], "source": source}, method, elapsed, measures)


def process_image(path, methods, config, rng, output_dir):
    """Застосовує обрані методи до одного зображення.

    Кожен метод отримує на вхід оригінал (а Гаус — зашумлену копію,
    щоб кількісно перевірити шумоподавлення), результат порівнюється
    з оригіналом за MSE / PSNR / SSIM.

    Returns:
        Список рядків CSV.
    """
    ctx = prepare_context(path, config, rng, output_dir)
    return [run_method(ctx, method, config, output_dir) for method in methods]


def write_csv(path, rows):
    """Записує детальні результати в CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    """Середні значення метрик по кожному методу (для порівняння методів)."""
    summary = []
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        entry = {"method": method, "images": len(subset)}
        for key in ("time_ms", "mse", "psnr", "ssim", "mse_noisy_baseline",
                    "mean", "contrast", "entropy"):
            values = [float(row[key]) for row in subset
                      if row[key] not in ("", "inf")]
            entry[key] = f"{sum(values) / len(values):.4f}" if values else ""
        summary.append(entry)
    return summary


def write_summary(path, summary):
    """Записує зведену таблицю по методах."""
    if not summary:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def plot_summary(path, summary):
    """Будує стовпчикову діаграму середніх PSNR і SSIM по методах."""
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        LOGGER.warning("matplotlib не встановлено — графік не побудовано")
        return
    data = [item for item in summary if item["ssim"]]
    if not data:
        return
    names = [item["method"] for item in data]
    fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(11, 4))
    ax_psnr.bar(names, [float(item["psnr"] or 0) for item in data], color="#4C72B0")
    ax_psnr.set_title("Середній PSNR, дБ")
    ax_ssim.bar(names, [float(item["ssim"]) for item in data], color="#55A868")
    ax_ssim.set_title("Середній SSIM")
    ax_ssim.set_ylim(min(0.0, min(float(item["ssim"]) for item in data)), 1.0)
    for axis in (ax_psnr, ax_ssim):
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def resolve_paths(args, config):
    """Обирає шляхи: аргументи командного рядка мають пріоритет над конфігом."""
    paths = config.get("paths", {})
    return (args.input or paths.get("raw_dir", "data/raw"),
            args.output or paths.get("processed_dir", "data/processed"),
            args.report or paths.get("report_csv", "reports/metrics_results.csv"))


def save_reports(report_csv, rows):
    """Зберігає детальний CSV, зведену таблицю та графік."""
    write_csv(report_csv, rows)
    summary = summarize(rows)
    report_dir = os.path.dirname(report_csv) or "."
    write_summary(os.path.join(report_dir, "metrics_summary.csv"), summary)
    plot_summary(os.path.join(report_dir, "metrics_summary.png"), summary)


def main(argv=None):
    """Точка входу: читає конфіг, обробляє вибірку, зберігає метрики."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        LOGGER.error("Не вдалося прочитати конфіг %s: %s", args.config, error)
        return 1

    input_dir, output_dir, report_csv = resolve_paths(args, config)
    methods = args.method or config.get("methods", list(ALL_METHODS))
    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        LOGGER.error("Невідомі методи в конфігу: %s", ", ".join(unknown))
        return 1

    images = list_images(input_dir)
    if not images:
        LOGGER.error("У каталозі %s немає зображень (.jpg/.png/.bmp/.tif)", input_dir)
        return 1

    rng = np.random.default_rng(int(config.get("random_seed", 42)))
    rows, skipped = [], 0
    for index, path in enumerate(images, start=1):
        try:
            rows.extend(process_image(path, methods, config, rng, output_dir))
            LOGGER.info("[%d/%d] %s — OK", index, len(images), os.path.basename(path))
        except (ValueError, OSError) as error:
            skipped += 1
            LOGGER.warning("[%d/%d] %s — пропущено: %s", index, len(images),
                           os.path.basename(path), error)

    if not rows:
        LOGGER.error("Жодне зображення не оброблено")
        return 1

    save_reports(report_csv, rows)
    LOGGER.info("Оброблено: %d, пропущено: %d", len(images) - skipped, skipped)
    LOGGER.info("Метрики: %s", report_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
