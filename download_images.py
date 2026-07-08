"""
Downloads each normalized product's image to data/images/{design_id}.jpg.
Standalone and resumable -- skips images already downloaded. Reusable by
any downstream step that needs product images (embeddings, visualization,
manual QA), not tied to any one consumer.
"""

import json
import os

import requests

import config

NORMALIZED_DIR = os.path.join(config.DATA_DIR, "normalized")
IMAGES_DIR = os.path.join(config.DATA_DIR, "images")
FAILED_LOG = os.path.join(config.LOG_DIR, "image_download_failures.txt")

os.makedirs(IMAGES_DIR, exist_ok=True)


def load_normalized_records():
    records = []
    for fname in os.listdir(NORMALIZED_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(NORMALIZED_DIR, fname), encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def download_image(design_id, image_url):
    image_path = os.path.join(IMAGES_DIR, f"{design_id}.jpg")
    if os.path.exists(image_path):
        return image_path, "skipped_existing"

    if not image_url:
        return None, "no_image_url"

    try:
        response = requests.get(image_url, headers=config.DEFAULT_HEADERS, timeout=config.REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        with open(image_path, "wb") as f:
            f.write(response.content)
        return image_path, "downloaded"
    except requests.RequestException as e:
        return None, f"error: {e}"


def main():
    records = load_normalized_records()
    print(f"[download_images] Found {len(records)} normalized record(s)")

    downloaded = 0
    skipped = 0
    failed = []

    for record in records:
        design_id = record["design_id"]
        path, status = download_image(design_id, record.get("image_url"))

        if status == "downloaded":
            downloaded += 1
            print(f"[download_images] designId={design_id}: downloaded")
        elif status == "skipped_existing":
            skipped += 1
        else:
            failed.append((design_id, status))
            print(f"[download_images] designId={design_id}: FAILED ({status})")

    if failed:
        with open(FAILED_LOG, "w") as f:
            for design_id, reason in failed:
                f.write(f"{design_id},{reason}\n")

    print(f"\n[download_images] Done. {downloaded} downloaded, {skipped} already existed, {len(failed)} failed.")
    if failed:
        print(f"[download_images] See {FAILED_LOG} for details.")


if __name__ == "__main__":
    main()