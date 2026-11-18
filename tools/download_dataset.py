"""Завантаження вибірки з 30 зображень космічних апаратів з NASA Image Library.

Використовує відкритий API https://images-api.nasa.gov (ключ не потрібен).
Знімки NASA є суспільним надбанням. Для кожного запиту береться перший
результат, ще не використаний у вибірці. Список джерел зберігається у
``data/sources.csv`` (для розділу «Опис вибірки» у звіті).

Запуск з кореня проєкту:
    python tools/download_dataset.py
    python tools/download_dataset.py --dry-run     # лише показати, що буде завантажено

Повторний запуск докачує лише відсутні файли.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://images-api.nasa.gov/search"
DETAILS_URL = "https://images.nasa.gov/details/{}"
USER_AGENT = "space-image-project/1.0 (student STEM project)"
SIZE_PREFERENCE = ("~medium.jpg", "~large.jpg", "~small.jpg", "~orig.jpg",
                   "~medium.png", "~large.png", "~orig.png")
TIMEOUT = 30

# (ім'я файлу без розширення, пошуковий запит)
QUERIES = (
    # Супутники та телескопи на орбіті (8)
    ("sat_01_hubble_release", "Hubble Space Telescope released from Space Shuttle"),
    ("sat_02_hubble_sm4", "Hubble Space Telescope backdropped by Earth STS-125 Atlantis"),
    ("sat_03_hubble_sm1", "Hubble Space Telescope STS-61 grapple"),
    ("sat_04_tdrs_deploy", "TDRS satellite deploy payload bay"),
    ("sat_05_chandra", "Chandra X-ray Observatory deploy STS-93"),
    ("sat_06_compton", "Compton Gamma Ray Observatory release Atlantis"),
    ("sat_07_intelsat", "Intelsat VI capture STS-49"),
    ("sat_08_shuttle_deploy", "satellite deployed from Space Shuttle orbit"),
    # Космічні станції (7)
    ("iss_01_earth", "International Space Station backdropped by Earth"),
    ("iss_02_space", "International Space Station backdropped by blackness of space"),
    ("iss_03_soyuz_view", "International Space Station photographed from Soyuz"),
    ("iss_04_flyaround", "International Space Station fly around after undocking"),
    ("iss_05_sun_transit", "International Space Station transit Sun"),
    ("mir_01_atlantis", "Mir space station photographed from Space Shuttle Atlantis"),
    ("skylab_01_orbit", "Skylab space station in Earth orbit"),
    # Ракети-носії при запуску (8)
    ("rocket_01_saturn_v", "Apollo 11 launch Saturn V Launch Complex 39A"),
    ("rocket_02_shuttle_night", "Space Shuttle night launch"),
    ("rocket_03_shuttle_day", "Space Shuttle Columbia STS-1 launch"),
    ("rocket_04_falcon9", "SpaceX Falcon 9 Crew Dragon launch"),
    ("rocket_05_falcon_heavy", "Falcon Heavy launch"),
    ("rocket_06_sls_artemis", "Artemis I Space Launch System launch"),
    ("rocket_07_atlas_v", "United Launch Alliance Atlas V rocket lifts off"),
    ("rocket_08_delta_iv", "Delta IV Heavy launch"),
    # Марсоходи (7)
    ("rover_01_curiosity_selfie", "Curiosity rover selfie Mars"),
    ("rover_02_perseverance_selfie", "Perseverance rover selfie"),
    ("rover_03_sojourner", "Sojourner rover Mars Pathfinder"),
    ("rover_04_opportunity", "Opportunity rover Mars"),
    ("rover_05_spirit", "Spirit rover Mars"),
    ("rover_06_ingenuity", "Ingenuity helicopter Mars Perseverance"),
    ("rover_07_curiosity_test", "Curiosity rover clean room test"),
)

SOURCE_FIELDS = ("file", "query", "nasa_id", "title", "date_created", "center",
                 "page_url", "image_url")


def http_get(url):
    """GET-запит з таймаутом; повертає байти відповіді."""
    safe_url = urllib.parse.quote(url, safe=":/?&=%~+")
    request = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def search(query, fetch=http_get):
    """Пошук зображень у NASA Image Library; повертає список елементів."""
    params = urllib.parse.urlencode({"q": query, "media_type": "image"})
    payload = json.loads(fetch(f"{API_URL}?{params}").decode("utf-8"))
    return payload.get("collection", {}).get("items", [])


def choose_asset(asset_urls):
    """Обирає URL зображення за пріоритетом розміру (medium -> large -> ...)."""
    for suffix in SIZE_PREFERENCE:
        for url in asset_urls:
            if url.lower().endswith(suffix):
                return url.replace("http://", "https://", 1)
    return None


def resolve_item(item, fetch=http_get):
    """Повертає (метадані, URL зображення) для елемента пошуку або None."""
    data = (item.get("data") or [{}])[0]
    nasa_id = data.get("nasa_id")
    href = item.get("href")
    if not nasa_id or not href:
        return None
    assets = json.loads(fetch(href).decode("utf-8"))
    image_url = choose_asset(assets)
    if image_url is None:
        return None
    meta = {
        "nasa_id": nasa_id,
        "title": data.get("title", ""),
        "date_created": data.get("date_created", "")[:10],
        "center": data.get("center", ""),
        "page_url": DETAILS_URL.format(nasa_id),
    }
    return meta, image_url


def is_image(content):
    """Перевіряє сигнатуру JPEG/PNG, щоб не зберегти HTML-сторінку помилки."""
    return content.startswith(b"\xff\xd8\xff") or content.startswith(b"\x89PNG")


def pick_for_query(query, used_ids, fetch=http_get, max_candidates=8):
    """Перший результат запиту, якого ще немає у вибірці."""
    for item in search(query, fetch)[:max_candidates]:
        data = (item.get("data") or [{}])[0]
        if data.get("nasa_id") in used_ids:
            continue
        resolved = resolve_item(item, fetch)
        if resolved is not None:
            return resolved
    return None


def load_sources(path):
    """Читає вже збережені джерела (для докачування)."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def save_sources(path, sources):
    """Записує data/sources.csv."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        for name, _ in QUERIES:
            for row in sources.values():
                if os.path.splitext(row["file"])[0] == name:
                    writer.writerow(row)


def download_one(name, query, context, fetch=http_get):
    """Знаходить і завантажує одне зображення; повертає рядок джерела або None."""
    picked = pick_for_query(query, context["used_ids"], fetch)
    if picked is None:
        print(f"  ✗ {name}: нічого не знайдено за запитом «{query}»")
        return None
    meta, image_url = picked
    extension = os.path.splitext(image_url)[1].lower()
    filename = f"{name}{extension}"
    print(f"  → {filename}: {meta['title'][:70]}")
    if context["dry_run"]:
        return None
    content = fetch(image_url)
    if not is_image(content):
        print(f"  ✗ {name}: отримано не зображення, пропущено")
        return None
    with open(os.path.join(context["out_dir"], filename), "wb") as handle:
        handle.write(content)
    return {"file": filename, "query": query, "image_url": image_url, **meta}


def parse_args(argv=None):
    """Аргументи командного рядка."""
    parser = argparse.ArgumentParser(description="Завантаження вибірки з NASA Image Library")
    parser.add_argument("--out", default="data/raw", help="каталог для зображень")
    parser.add_argument("--sources", default="data/sources.csv", help="CSV зі списком джерел")
    parser.add_argument("--dry-run", action="store_true", help="лише показати вибір")
    return parser.parse_args(argv)


def main(argv=None, fetch=http_get):
    """Завантажує всі зображення зі списку QUERIES."""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    sources = load_sources(args.sources)
    context = {"out_dir": args.out, "dry_run": args.dry_run,
               "used_ids": {row["nasa_id"] for row in sources.values()}}
    existing = {os.path.splitext(name)[0] for name in os.listdir(args.out)}

    for index, (name, query) in enumerate(QUERIES, start=1):
        print(f"[{index}/{len(QUERIES)}] {query}")
        if name in existing:
            print(f"  ✓ {name}: вже є, пропускаю")
            continue
        try:
            row = download_one(name, query, context, fetch)
        except (urllib.error.URLError, OSError, ValueError) as error:
            print(f"  ✗ {name}: помилка мережі/даних: {error}")
            continue
        if row is not None:
            sources[row["file"]] = row
            context["used_ids"].add(row["nasa_id"])
            save_sources(args.sources, sources)
        time.sleep(0.3)  # не перевантажуємо API

    total = len([n for n in os.listdir(args.out) if not n.startswith(".")])
    print(f"\nУ {args.out}: {total} зображень. Джерела: {args.sources}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
