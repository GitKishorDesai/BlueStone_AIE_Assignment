"""
Fetches product details from the BlueStone Product Details API for each
sampled design ID, and caches the raw response locally.

Resumable: skips any designId that's already cached. Safe to re-run
anytime -- only fetches what's missing.
"""

import argparse
import csv
import json
import os
import time

import requests

import config


def load_sampled_ids(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def already_cached(design_id):
    return os.path.exists(os.path.join(config.RAW_DIR, f"{design_id}.json"))


def parse_image_schema(raw_json):
    """
    imageSchema is a JSON-encoded string nested inside the response.
    Parse it once here so every downstream script gets clean data
    without needing to remember this quirk.
    Returns None if missing or malformed -- doesn't crash the pipeline.
    """
    raw_schema = raw_json.get("imageSchema")
    if not raw_schema:
        return None
    try:
        return json.loads(raw_schema)
    except (json.JSONDecodeError, TypeError):
        return None


def resolve_image_url(raw_json):
    """
    Applies the fallback priority: imageUrl -> mediaItems.imageItems[0].urls.828
    -> carouselSeq[0].urlMap.828. Returns None if all are missing.
    """
    if raw_json.get("imageUrl"):
        return raw_json["imageUrl"]

    try:
        return raw_json["mediaItems"]["imageItems"][0]["urls"]["828"]
    except (KeyError, IndexError, TypeError):
        pass

    try:
        return raw_json["mediaItems"]["carouselSeq"][0]["urlMap"]["828"]
    except (KeyError, IndexError, TypeError):
        pass

    return None


def fetch_one(design_id):
    """
    Attempts to fetch a single design ID with retry/backoff.
    Returns (success: bool, error_reason: str or None).
    """
    url = config.PRODUCT_DETAILS_URL.format(design_id=design_id)

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=config.DEFAULT_HEADERS,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 404:
                return False, "404_not_found"

            response.raise_for_status()
            raw_json = response.json()

            # BlueStone sometimes returns HTTP 200 with a JSON envelope wrapping
            # an actual upstream error (e.g. 503) inside an "error" string field,
            # instead of a proper non-200 status code. Detect and retry these.
            if isinstance(raw_json, dict) and "error" in raw_json and "designId" not in raw_json:
                if attempt < config.MAX_RETRIES:
                    time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)
                    continue
                else:
                    return False, "upstream_503_error"

            # Enrich with parsed imageSchema + resolved image URL before caching
            raw_json["_parsed_image_schema"] = parse_image_schema(raw_json)
            raw_json["_resolved_image_url"] = resolve_image_url(raw_json)

            out_path = os.path.join(config.RAW_DIR, f"{design_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(raw_json, f, ensure_ascii=False, indent=2)

            return True, None

        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)
            else:
                return False, str(e)

    return False, "unknown_failure"


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache product details for sampled design IDs.")
    parser.add_argument(
        "--sampled-ids-file",
        default=os.path.join(config.DATA_DIR, "sampled_ids.txt"),
        help="Path to the sampled_ids.txt produced by sampler.py",
    )
    args = parser.parse_args()

    ids = load_sampled_ids(args.sampled_ids_file)
    print(f"[fetcher] Loaded {len(ids)} design ID(s) to process")

    to_fetch = [i for i in ids if not already_cached(i)]
    skipped = len(ids) - len(to_fetch)
    if skipped:
        print(f"[fetcher] Skipping {skipped} already-cached ID(s)")

    log_rows = []
    failed_ids = []

    for idx, design_id in enumerate(to_fetch, start=1):
        success, error_reason = fetch_one(design_id)
        status = "OK" if success else f"FAILED ({error_reason})"
        print(f"[fetcher] ({idx}/{len(to_fetch)}) designId={design_id}: {status}")

        log_rows.append({"designId": design_id, "success": success, "error": error_reason or ""})
        if not success:
            failed_ids.append(design_id)

        time.sleep(config.SLEEP_BETWEEN_REQUESTS)

    # Write/append fetch log
    write_header = not os.path.exists(config.FETCH_LOG_FILE)
    with open(config.FETCH_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["designId", "success", "error"])
        if write_header:
            writer.writeheader()
        writer.writerows(log_rows)

    if failed_ids:
        with open(config.FAILED_IDS_FILE, "a") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
        print(f"[fetcher] {len(failed_ids)} ID(s) failed. See {config.FAILED_IDS_FILE} and {config.FETCH_LOG_FILE}")

    fetched_count = len(to_fetch) - len(failed_ids)
    print(f"[fetcher] Done. {fetched_count} new record(s) cached, {len(failed_ids)} failed, {skipped} already existed.")


if __name__ == "__main__":
    main()