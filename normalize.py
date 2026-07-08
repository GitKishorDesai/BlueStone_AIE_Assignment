"""
Reads each raw cached product JSON from data/raw/ and produces a clean,
flat, matching-ready record in data/normalized/.

All logic here is derived directly from what inspect.py revealed about
the real data -- e.g. the gemstone fallback chain exists because we
observed three separate, largely non-overlapping field-sets for it.

No category list or tag vocabulary is hardcoded: category handling uses
string-pattern checks (e.g. "kids" substring) so it keeps working if the
full 8k catalog reveals categories the 96-sample didn't have.
"""

import json
import os
import re

import config

NORMALIZED_DIR = os.path.join(config.DATA_DIR, "normalized")
os.makedirs(NORMALIZED_DIR, exist_ok=True)


def clean_price(price_str):
    """Converts '₹ 95,980' style strings into a plain float. Returns None if unparseable."""
    if price_str is None:
        return None
    digits = re.sub(r"[^\d.]", "", str(price_str))
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def get_metal_color(record):
    """Prefer the reliable imageSchema color field; fall back to tag scanning."""
    parsed = record.get("_parsed_image_schema") or {}
    color = parsed.get("color")
    if color:
        return color
    return None  # will be backfilled by tag-based fallback if needed downstream


def get_gemstone_info(record):
    """
    Merges three mutually-exclusive-ish field sets (diamond / colored gemstone /
    solitaire) into one normalized structure. Falls back to 'plain_gold' when
    none are present -- a real, common product state, not missing data.
    """
    if record.get("diamondStoneType") or record.get("totalDiamonds"):
        return {
            "kind": "diamond",
            "stone_type": record.get("diamondStoneType"),
            "carat": record.get("diamondCarat"),
            "count": record.get("totalDiamonds"),
        }

    gemstones = record.get("gemstonesTypeAndCount")
    if gemstones:
        return {
            "kind": "colored_gemstone",
            "details": gemstones,
        }

    if record.get("solitaireClarity") or record.get("solitaireTotalCarat"):
        return {
            "kind": "solitaire",
            "clarity": record.get("solitaireClarity"),
            "color": record.get("solitaireColor"),
            "carat": record.get("solitaireTotalCarat"),
            "count": record.get("solitaireTotalNoOfStones"),
        }

    return {"kind": "plain_gold"}


def get_tag_bag(record):
    """
    Returns all tag keys (dict keys of 'tags') lowercased and joined,
    for cheap substring searching -- since tags is a large free-vocabulary
    dict, not a fixed schema, exact-key lookups aren't reliable.
    """
    tags = record.get("tags") or {}
    return " | ".join(k.lower() for k in tags.keys())


def detect_gender(tag_bag, design_name, short_desc):
    text = f"{tag_bag} | {design_name or ''} | {short_desc or ''}".lower()

    # Explicit, unambiguous phrasing overrides everything else
    if re.search(r"\bfor him\b|\bmen's\b|\bmens\b(?!\s*earring)", text):
        return "men"
    if re.search(r"\bfor her\b|\bwomen's\b|\bwomens\b", text):
        return "women"

    mentions_women = bool(re.search(r"\bwomen\b|\bwomens\b", text))
    mentions_men = bool(re.search(r"\bmen\b|\bmens\b", text))
    mentions_couple = "couple" in text or "unisex" in text

    if mentions_couple:
        return "unisex"
    if mentions_women and mentions_men:
        return "unisex"
    if mentions_women:
        return "women"
    if mentions_men:
        return "men"
    return "unknown"


def detect_is_kids(category_type, item_category_name, tag_bag):
    """Kids-detection via category name first (cheap, reliable), then tags."""
    combined = f"{category_type or ''} {item_category_name or ''} {tag_bag}".lower()
    return "kids" in combined


def detect_occasions(tag_bag):
    """
    Returns a list of matched occasion keywords found anywhere in the tag bag.
    Kept as a list (not a single value) since products can have multiple
    occasion tags (e.g. both 'Wedding' and 'Anniversary').
    """
    occasion_keywords = [
        "wedding", "anniversary", "engagement", "festive", "akshaya tritiya",
        "valentines day", "everyday", "office", "workwear", "party",
        "vacation", "weekend", "special occasion", "gift",
    ]
    found = [kw for kw in occasion_keywords if kw in tag_bag]
    return found


def normalize_record(record):
    missing_fields = []

    category_type = record.get("categoryType")
    item_category_name = record.get("itemCategoryName")
    design_name = record.get("designName")
    short_desc = record.get("shortDesc")
    tag_bag = get_tag_bag(record)

    metal_color = get_metal_color(record)
    if metal_color is None:
        missing_fields.append("metal_color")

    collection_name = record.get("collectionName")
    if not collection_name:
        missing_fields.append("collectionName")

    gemstone_info = get_gemstone_info(record)

    gender = detect_gender(tag_bag, design_name, short_desc)
    if gender == "unknown":
        missing_fields.append("gender")

    is_kids = detect_is_kids(category_type, item_category_name, tag_bag)

    occasions = detect_occasions(tag_bag)
    if not occasions:
        missing_fields.append("occasions")

    price = clean_price(record.get("discountedPrice") or record.get("price"))

    normalized = {
        "design_id": record.get("designId"),
        "design_name": design_name,
        "category_type": category_type,
        "item_category_name": item_category_name,
        "is_kids": is_kids,
        "gender": gender,
        "metal": record.get("metal"),
        "metal_type": record.get("metalType"),
        "gold_purity": record.get("goldPurity"),
        "metal_color": metal_color,
        "gemstone_info": gemstone_info,
        "collection_name": collection_name,
        "occasions": occasions,
        "price": price,
        "image_url": record.get("_resolved_image_url"),
        "product_page_url": (record.get("canonicalUrl") or "").replace("__BASE_URL__", config.BASE_URL),
        "raw_tag_count": len((record.get("tags") or {})),
        "missing_fields": missing_fields,
    }
    return normalized


def main():
    files = [f for f in os.listdir(config.RAW_DIR) if f.endswith(".json")]
    print(f"[normalize] Found {len(files)} raw record(s) to normalize")

    stats_missing = {}
    processed = 0

    for fname in files:
        with open(os.path.join(config.RAW_DIR, fname), encoding="utf-8") as f:
            record = json.load(f)

        normalized = normalize_record(record)

        out_path = os.path.join(NORMALIZED_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

        for field in normalized["missing_fields"]:
            stats_missing[field] = stats_missing.get(field, 0) + 1

        processed += 1

    print(f"[normalize] Normalized {processed} record(s) -> {NORMALIZED_DIR}")
    print("\n[normalize] Missing-field summary (needs fallback/LLM enrichment):")
    for field, count in sorted(stats_missing.items(), key=lambda x: -x[1]):
        print(f"  {field}: {count} / {processed}")


if __name__ == "__main__":
    main()