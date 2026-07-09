"""
Enriches gender for records in SQLite where gender was marked 'unknown'.
Text-only, using designName + shortDesc + category + tag bag.
"""

import json
import os
import re
import sqlite3
import time

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

import config

DB_PATH = os.path.join(config.DATA_DIR, "products.db")
ENRICHMENT_LOG = os.path.join(config.LOG_DIR, "gender_enrichment.json")
NON_RING_DEFAULT_GENDER = "women"
ACTIVE_PROVIDER = "groq"


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
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT design_id, data FROM products")
    rows = cur.fetchall()

    records_needing_gender = []
    for design_id, data_str in rows:
        record = json.loads(data_str)
        if "gender" in record.get("missing_fields", []):
            records_needing_gender.append(record)

    print(f"[enrich_gender] Found {len(records_needing_gender)} record(s) needing gender enrichment")

    enrichment_log = {}

    for record in records_needing_gender:
        category = (record.get("category_type") or "").lower()
        is_ring = bool(re.search(r"\bring\b", category))

        if not is_ring:
            gender = NON_RING_DEFAULT_GENDER
            enrichment_log[record["design_id"]] = {"method": "category_default", "gender": gender}
        else:
            prompt = build_prompt(record)
            raw_response = call_llm(prompt)
            time.sleep(config.LLM_CALL_DELAY_SECONDS)
            parsed = parse_llm_response(raw_response)

            if parsed and parsed.get("gender") in ("women", "men", "unisex"):
                gender = parsed["gender"]
                enrichment_log[record["design_id"]] = {
                    "method": "llm", "gender": gender,
                    "confidence": parsed.get("confidence"), "reasoning": parsed.get("reasoning"),
                }
            else:
                gender = "unisex"
                enrichment_log[record["design_id"]] = {
                    "method": "llm_failed_fallback", "gender": gender, "raw_response": raw_response,
                }

        record["gender"] = gender
        record["missing_fields"] = [f for f in record["missing_fields"] if f != "gender"]
        record["gender_enrichment_method"] = enrichment_log[record["design_id"]]["method"]

        cur.execute("UPDATE products SET data = ? WHERE design_id = ?",
                     (json.dumps(record, ensure_ascii=False), record["design_id"]))

        print(f"[enrich_gender] designId={record['design_id']} ({record['category_type']}) -> {gender} "
              f"[{enrichment_log[record['design_id']]['method']}]")

    conn.commit()
    conn.close()

    with open(ENRICHMENT_LOG, "w", encoding="utf-8") as f:
        json.dump(enrichment_log, f, ensure_ascii=False, indent=2)

    print(f"\n[enrich_gender] Done. Log saved to {ENRICHMENT_LOG}")


if __name__ == "__main__":
    main()