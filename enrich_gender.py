"""
Enriches gender for normalized records where normalize.py marked it 'unknown'.
Text-only, using designName + shortDesc + category + tag bag.

Provider-agnostic by design: the actual LLM call is isolated in
call_llm_groq(). To switch providers later, write a new call_llm_<provider>()
function with the same signature and point ACTIVE_PROVIDER at it.
"""

import json
import os
import re

from groq import Groq
from dotenv import load_dotenv
load_dotenv()  

import config

NORMALIZED_DIR = os.path.join(config.DATA_DIR, "normalized")
ENRICHMENT_LOG = os.path.join(config.LOG_DIR, "gender_enrichment.json")

# Non-ring categories default to this per catalog convention (BlueStone skews
# overwhelmingly women's fine jewellery outside of rings) rather than
# spending an LLM call on near-certain cases.
NON_RING_DEFAULT_GENDER = "women"

ACTIVE_PROVIDER = "groq"  # swap to "gemini" once that function exists


def load_records_needing_gender():
    records = []
    for fname in os.listdir(NORMALIZED_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(NORMALIZED_DIR, fname)
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        if "gender" in record.get("missing_fields", []):
            records.append((path, record))
    return records


def build_prompt(record):
    return f"""You are classifying the target gender for a jewellery product based on its metadata.

Product details:
- Category: {record.get("category_type")}
- Sub-category: {record.get("item_category_name")}
- Design name: {record.get("design_name")}
- Existing tag keywords: {record.get("raw_tag_count")} tags present (not gender-specific ones, already checked)

Classify the target gender as one of: "women", "men", "unisex".
If the product name/category gives no clear signal, default to "unisex".

Respond with ONLY valid JSON, no other text, in this exact format:
{{"gender": "women", "confidence": "high", "reasoning": "brief reason"}}
"""


def call_llm_groq(prompt):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content


def call_llm(prompt):
    if ACTIVE_PROVIDER == "groq":
        return call_llm_groq(prompt)
    raise ValueError(f"Unknown provider: {ACTIVE_PROVIDER}")


def parse_llm_response(raw_text):
    """Strips markdown code fences if present, then parses JSON."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def main():
    records = load_records_needing_gender()
    print(f"[enrich_gender] Found {len(records)} record(s) needing gender enrichment")

    enrichment_log = {}

    for path, record in records:
        category = (record.get("category_type") or "").lower()
        is_ring = bool(re.search(r"\bring\b", category))

        if not is_ring:
            # Cheap default, no LLM call needed -- documented assumption
            gender = NON_RING_DEFAULT_GENDER
            enrichment_log[record["design_id"]] = {
                "method": "category_default",
                "gender": gender,
            }
        else:
            prompt = build_prompt(record)
            raw_response = call_llm(prompt)
            parsed = parse_llm_response(raw_response)

            if parsed and parsed.get("gender") in ("women", "men", "unisex"):
                gender = parsed["gender"]
                enrichment_log[record["design_id"]] = {
                    "method": "llm",
                    "gender": gender,
                    "confidence": parsed.get("confidence"),
                    "reasoning": parsed.get("reasoning"),
                }
            else:
                # LLM failed to return usable JSON -- fall back safely rather than crash
                gender = "unisex"
                enrichment_log[record["design_id"]] = {
                    "method": "llm_failed_fallback",
                    "gender": gender,
                    "raw_response": raw_response,
                }

        # Update the record in place
        record["gender"] = gender
        record["missing_fields"] = [f for f in record["missing_fields"] if f != "gender"]
        record["gender_enrichment_method"] = enrichment_log[record["design_id"]]["method"]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"[enrich_gender] designId={record['design_id']} ({record['category_type']}) -> {gender} "
              f"[{enrichment_log[record['design_id']]['method']}]")

    with open(ENRICHMENT_LOG, "w", encoding="utf-8") as f:
        json.dump(enrichment_log, f, ensure_ascii=False, indent=2)

    print(f"\n[enrich_gender] Done. Log saved to {ENRICHMENT_LOG}")


if __name__ == "__main__":
    main()