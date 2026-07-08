"""
Inspects all cached raw product-detail JSONs and produces a report on:
- category distribution
- field completeness (including nested fields, discovered dynamically)
- tags richness

No field or category list is hardcoded -- everything is discovered from
the actual cached data, so this script keeps working unmodified as the
dataset grows or reveals fields/categories not seen in the dev sample.
"""

import json
import os
from collections import defaultdict, Counter

import config


def load_all_records():
    records = []
    for fname in os.listdir(config.RAW_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(config.RAW_DIR, fname), encoding="utf-8") as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError:
                print(f"[inspect] WARNING: could not parse {fname}, skipping")
    return records


def flatten_keys(obj, prefix=""):
    """
    Recursively walks a JSON object and yields dotted key paths.
    Lists are handled by inspecting their first element only (as a
    representative sample), tagged with '[0]' in the path.
    This is discovery-only -- it doesn't hardcode any field names.
    """
    paths = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.append(new_prefix)
            paths.extend(flatten_keys(v, new_prefix))
    elif isinstance(obj, list) and obj:
        new_prefix = f"{prefix}[0]"
        paths.extend(flatten_keys(obj[0], new_prefix))
    return paths


def get_value_by_path(obj, path):
    """
    Given a dotted path like 'mediaItems.imageItems[0].urls.828',
    walks the object and returns the value, or None if not found.
    """
    parts = path.replace("[0]", ".0").split(".")
    current = obj
    for part in parts:
        try:
            if isinstance(current, list):
                current = current[int(part)]
            else:
                current = current[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return current


def is_empty(value):
    return value is None or value == "" or value == {} or value == []


def main():
    records = load_all_records()
    total = len(records)
    print(f"[inspect] Loaded {total} cached record(s)\n")

    if total == 0:
        print("[inspect] No records found -- run fetcher.py first.")
        return

    # ---- Category distribution ----
    category_counts = Counter(r.get("categoryType", "UNKNOWN") for r in records)
    print("=== Category Distribution ===")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")
    print()

    # ---- Discover all field paths across all records ----
    all_paths = set()
    for r in records:
        all_paths.update(flatten_keys(r))

    # ---- Field completeness ----
    print("=== Field Completeness (non-empty %) ===")
    completeness = {}
    for path in sorted(all_paths):
        non_empty = sum(1 for r in records if not is_empty(get_value_by_path(r, path)))
        pct = round(100 * non_empty / total, 1)
        completeness[path] = pct

    # Sort by completeness ascending -- the sparse/risky fields surface first
    for path, pct in sorted(completeness.items(), key=lambda x: x[1]):
        print(f"  {pct:5.1f}%  {path}")
    print()

    # ---- Tags richness ----
    print("=== Tags Analysis ===")
    tag_keys_seen = Counter()
    tags_per_product = []
    for r in records:
        tags = r.get("tags") or {}
        tags_per_product.append(len(tags))
        tag_keys_seen.update(tags.keys())

    avg_tags = round(sum(tags_per_product) / total, 1) if total else 0
    zero_tag_products = sum(1 for c in tags_per_product if c == 0)
    print(f"  Avg tags per product: {avg_tags}")
    print(f"  Products with zero tags: {zero_tag_products} / {total}")
    print(f"  Distinct tag keys seen across sample: {len(tag_keys_seen)}")
    print("  Most common tags:")
    for tag, count in tag_keys_seen.most_common(15):
        print(f"    {count:3d}x  {tag}")
    print()

    # ---- Save full report to file ----
    report_path = os.path.join(config.LOG_DIR, "inspection_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Total records: {total}\n\n")
        f.write("Category Distribution:\n")
        for cat, count in category_counts.most_common():
            f.write(f"  {cat}: {count}\n")
        f.write("\nField Completeness:\n")
        for path, pct in sorted(completeness.items(), key=lambda x: x[1]):
            f.write(f"  {pct:5.1f}%  {path}\n")
        f.write("\nTags Analysis:\n")
        f.write(f"  Avg tags per product: {avg_tags}\n")
        f.write(f"  Products with zero tags: {zero_tag_products} / {total}\n")
        f.write(f"  Distinct tag keys seen: {len(tag_keys_seen)}\n")
        f.write("  Most common tags:\n")
        for tag, count in tag_keys_seen.most_common(30):
            f.write(f"    {count:3d}x  {tag}\n")

    print(f"[inspect] Full report saved to {report_path}")


if __name__ == "__main__":
    main()