"""
Core matching logic for the jewellery matching-set recommender.

Retrieval now goes through Chroma: the anchor's embedding is used to query
the vector collection for the top-N most visually similar candidates,
with is_kids pre-filtered at the Chroma query level (cheap, exact match).
Category-canonicalization and gender-compatibility hard filters, plus all
soft-criteria scoring, still run in Python on that shortlist -- this keeps
the nuanced logic (unisex-as-wildcard, ring/solitaire-ring grouping)
explicit and readable rather than baked into a query filter language.
"""

import numpy as np

import config
from data_loader import load_all, get_product_by_id

WEIGHTS = {
    "visual_similarity": 0.30,
    "metal_match": 0.25,
    "gemstone_match": 0.20,
    "price_closeness": 0.10,
    "collection_match": 0.10,
    "occasion_match": 0.05,
}

# How many visually-similar candidates to retrieve from Chroma before
# applying remaining hard filters + scoring. Larger than final top_k
# since some retrieved candidates will still be filtered out (same
# category, gender mismatch) after retrieval.
RETRIEVAL_POOL_SIZE = 40


# ---------------------------------------------------------------------
# Hard filters (applied to the Chroma-retrieved shortlist)
# ---------------------------------------------------------------------

def is_same_category(anchor, candidate):
    return config.canonicalize_category(anchor["category_type"]) == config.canonicalize_category(candidate["category_type"])


def is_same_zone_group(anchor, candidate):
    anchor_group = config.get_zone_group(anchor["category_type"])
    if anchor_group is None:
        return False
    return candidate["category_type"] in anchor_group


def gender_compatible(anchor, candidate):
    """unisex is compatible with anything; women/men must match exactly."""
    a, c = anchor["gender"], candidate["gender"]
    if a == "unisex" or c == "unisex":
        return True
    return a == c


def passes_remaining_filters(anchor, candidate, include_same_zone=False):
    """
    is_kids is already filtered at the Chroma query level -- this only
    re-checks category/zone/gender, which aren't expressed as simple
    Chroma metadata equality filters.
    """
    if is_same_category(anchor, candidate):
        return False
    if not include_same_zone and is_same_zone_group(anchor, candidate):
        return False
    if not gender_compatible(anchor, candidate):
        return False
    return True


# ---------------------------------------------------------------------
# Soft scoring criteria (each returns a float in [0, 1])
# ---------------------------------------------------------------------

def score_visual_similarity(anchor, candidate):
    sim = float(np.dot(anchor["embedding"], candidate["embedding"]))
    return max(0.0, min(1.0, sim))


def score_metal_match(anchor, candidate):
    type_match = 1.0 if anchor["metal_type"] == candidate["metal_type"] else 0.0
    color_match = 1.0 if (
        anchor["metal_color"] and candidate["metal_color"]
        and anchor["metal_color"] == candidate["metal_color"]
    ) else 0.0
    return (type_match + color_match) / 2


def score_gemstone_match(anchor, candidate):
    a_kind = anchor["gemstone_info"].get("kind")
    c_kind = candidate["gemstone_info"].get("kind")
    return 1.0 if a_kind == c_kind else 0.0


def score_price_closeness(anchor, candidate):
    a_price, c_price = anchor.get("price"), candidate.get("price")
    if not a_price or not c_price:
        return 0.0
    return min(a_price, c_price) / max(a_price, c_price)


def score_collection_match(anchor, candidate):
    a_col, c_col = anchor.get("collection_name"), candidate.get("collection_name")
    if not a_col or not c_col:
        return 0.0
    return 1.0 if a_col == c_col else 0.0


def score_occasion_match(anchor, candidate):
    a_occ, c_occ = set(anchor.get("occasions") or []), set(candidate.get("occasions") or [])
    if not a_occ or not c_occ:
        return 0.0
    overlap = a_occ & c_occ
    return len(overlap) / len(a_occ | c_occ)


SCORERS = {
    "visual_similarity": score_visual_similarity,
    "metal_match": score_metal_match,
    "gemstone_match": score_gemstone_match,
    "price_closeness": score_price_closeness,
    "collection_match": score_collection_match,
    "occasion_match": score_occasion_match,
}


def compute_match_score(anchor, candidate):
    breakdown = {name: scorer(anchor, candidate) for name, scorer in SCORERS.items()}
    final_score = sum(breakdown[name] * WEIGHTS[name] for name in WEIGHTS)
    return final_score, breakdown


# ---------------------------------------------------------------------
# Explanation generation (template-based, deterministic)
# ---------------------------------------------------------------------

def generate_explanation(anchor, candidate, breakdown):
    reasons = []

    if breakdown["metal_match"] >= 0.99:
        reasons.append(f"Same metal and finish ({candidate['metal_color']} {candidate['metal_type']})")
    elif breakdown["metal_match"] >= 0.5:
        reasons.append(f"Matching metal type ({candidate['metal_type']})")

    if breakdown["gemstone_match"] >= 0.99:
        kind = candidate["gemstone_info"].get("kind", "").replace("_", " ").title()
        reasons.append(f"Consistent gemstone style ({kind})")

    if breakdown["collection_match"] >= 0.99:
        reasons.append(f"Part of the same '{candidate['collection_name']}' collection")

    if breakdown["visual_similarity"] >= 0.8:
        reasons.append("Very similar visual design language")
    elif breakdown["visual_similarity"] >= 0.6:
        reasons.append("Similar visual style")

    if breakdown["price_closeness"] >= 0.8:
        reasons.append("Comparable price tier")

    if breakdown["occasion_match"] >= 0.5:
        reasons.append("Shared occasion and style tags")

    if not reasons:
        reasons.append("Moderate overall compatibility across style and metal")

    category_title = candidate["category_type"].title()
    return f"Recommended as a {category_title} match: " + "; ".join(reasons) + "."


# ---------------------------------------------------------------------
# Main entry point -- retrieval now via Chroma, scoring still in Python
# ---------------------------------------------------------------------

def get_matching_set(anchor_design_id, products, collection=None, top_k=5, max_per_category=1, include_same_zone=False):
    anchor = get_product_by_id(products, anchor_design_id)
    if anchor is None:
        raise ValueError(f"design_id {anchor_design_id} not found in loaded products")

    if collection is not None:
        # Chroma-backed retrieval: pre-filter on is_kids (cheap, exact),
        # retrieve top-N visually similar candidates by embedding distance.
        results = collection.query(
            query_embeddings=[anchor["embedding"].tolist()],
            n_results=min(RETRIEVAL_POOL_SIZE, len(products)),
            where={"is_kids": anchor["is_kids"]},
        )
        retrieved_ids = [int(id_str) for id_str in results["ids"][0]]
        candidate_pool = [get_product_by_id(products, did) for did in retrieved_ids if did != anchor_design_id]
    else:
        # Fallback: no collection provided, use full product list (e.g. for quick scripts/tests)
        candidate_pool = [p for p in products if p["design_id"] != anchor_design_id]

    candidates = [p for p in candidate_pool if passes_remaining_filters(anchor, p, include_same_zone)]

    scored = []
    for candidate in candidates:
        score, breakdown = compute_match_score(anchor, candidate)
        explanation = generate_explanation(anchor, candidate, breakdown)
        scored.append({
            "design_id": candidate["design_id"],
            "design_name": candidate["design_name"],
            "category_type": candidate["category_type"],
            "image_url": candidate["image_url"],
            "product_page_url": candidate["product_page_url"],
            "match_score": round(score, 4),
            "score_breakdown": {k: round(v, 3) for k, v in breakdown.items()},
            "explanation": explanation,
        })

    scored.sort(key=lambda x: x["match_score"], reverse=True)

    final_selection = []
    category_counts = {}

    for item in scored:
        cat = config.canonicalize_category(item["category_type"])
        if category_counts.get(cat, 0) >= max_per_category:
            continue
        final_selection.append(item)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if len(final_selection) >= top_k:
            break

    return anchor, final_selection


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Test the matching-set recommender for a given (or random) design ID.")
    parser.add_argument("--design-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-same-zone", action="store_true")
    args = parser.parse_args()

    products, collection, id_order = load_all()

    test_id = args.design_id if args.design_id is not None else random.choice(products)["design_id"]
    if args.design_id is None:
        print(f"[matching] No --design-id given, picked random anchor: {test_id}")

    anchor, top_k = get_matching_set(
        test_id, products, collection=collection, top_k=args.top_k, include_same_zone=args.include_same_zone
    )

    print(f"\nAnchor: {anchor['design_name']} (designId={anchor['design_id']}, {anchor['category_type']})\n")
    for i, item in enumerate(top_k, start=1):
        print(f"{i}. {item['design_name']} ({item['category_type']}) -- score: {item['match_score']}")
        print(f"   {item['explanation']}")
        print(f"   breakdown: {item['score_breakdown']}\n")