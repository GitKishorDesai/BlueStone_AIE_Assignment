"""
Central configuration for the BlueStone matching-set pipeline.
Keep every path, URL, and tunable setting here so nothing is buried
inside the actual logic scripts.
"""

import os
import re

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # one JSON file per designId
LOG_DIR = os.path.join(DATA_DIR, "logs")
FAILED_IDS_FILE = os.path.join(LOG_DIR, "failed_ids.txt")
FETCH_LOG_FILE = os.path.join(LOG_DIR, "fetch_log.csv")

# ---- API endpoints ----
PRODUCT_DETAILS_URL = "https://page.bluestone.com/page/product/{design_id}"
SIMILAR_DESIGNS_URL = "https://www.bluestone.com/similar-design/design-group/{design_id}"
BASE_URL = "https://www.bluestone.com"

# ---- Request settings ----
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2      # multiplied by attempt number
SLEEP_BETWEEN_REQUESTS = 0.3   # be polite to their API, avoid rate-limit/ban

# ---- Sampling ----
DEFAULT_SAMPLE_SIZE = 100
RANDOM_SEED = 42                # fixed seed -> reproducible sample across runs

# ---- Headers (some APIs reject requests with no user-agent) ----
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# ---- Category zone groups ----
# Categories in the same group occupy the same "body slot" and are excluded
# from each other's recommendations by default (e.g. a necklace anchor
# shouldn't default-recommend another neck-worn item as if it were a
# complementary piece). User can opt in via include_same_zone=True.
ZONE_GROUPS = [
    {"Necklaces", "Pendants", "Mangalsutra Chains"},
]

# ---- Category canonicalization ----
# Maps detailed category_type strings down to a core canonical category,
# so subtype variants (e.g. "PreSet Solitaire Rings", "Rings") are treated
# as the same category for compatibility/exclusion purposes, while the
# original detailed label is still kept for display/explanations.
CATEGORY_KEYWORDS = {
    "ring": r"\bring(s)?\b",
    "earring": r"\bearring(s)?\b",
    "necklace": r"\bnecklace(s)?\b",
    "pendant": r"\bpendant(s)?\b",
    "mangalsutra": r"\bmangalsutra\b",
    "bangle": r"\bbangle(s)?\b",
    "bracelet": r"\bbracelet(s)?\b",
    "nose_pin": r"\bnose pin(s)?\b|\bnose screw(s)?\b",
}


def canonicalize_category(category_type):
    """
    Returns the canonical core category for a detailed category_type string.
    Falls back to the lowercased original string if no keyword matches
    (keeps things working for categories not yet seen/anticipated).
    """
    text = (category_type or "").lower()
    for canonical, pattern in CATEGORY_KEYWORDS.items():
        if re.search(pattern, text):
            return canonical
    return text  # fallback: treat unrecognized categories as their own bucket


def get_zone_group(category_type):
    """Returns the zone-group set containing this category, or None if it's not in any group."""
    for group in ZONE_GROUPS:
        if category_type in group:
            return group
    return None

for _dir in (DATA_DIR, RAW_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)