"""
FastAPI backend for the jewellery matching-set UI.

Connects to SQLite (product records) and Chroma (vector index) once at
startup -- no bulk in-memory loading. Exposes:
  GET /products             -> list of {design_id, design_name, category_type}
  GET /recommend             -> anchor + top-K recommendations
  GET /about                 -> models/prompts used, for transparency
"""
import random
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
import json
import re
from groq import Groq
from google import genai as google_genai

from dotenv import load_dotenv
load_dotenv()

from data_loader import load_all, get_product_by_id, get_products_by_ids,list_all_products
from matching import get_matching_set

app = FastAPI(title="BlueStone Matching Set API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONN, COLLECTION, VALID_IDS = load_all()


@app.get("/products")
def list_products():
    return list_all_products(CONN)


@app.get("/recommend")
def recommend(
    design_id: int = Query(...),
    top_k: int = Query(5, ge=1, le=20),
    include_same_zone: bool = Query(False),
):
    anchor = get_product_by_id(CONN, design_id)
    if anchor is None:
        raise HTTPException(status_code=404, detail=f"design_id {design_id} not found")

    anchor_record, recommendations = get_matching_set(
        design_id, CONN, COLLECTION, top_k=top_k, include_same_zone=include_same_zone
    )

    return {
        "anchor": {
            "design_id": anchor_record["design_id"],
            "design_name": anchor_record["design_name"],
            "category_type": anchor_record["category_type"],
            "image_url": anchor_record["image_url"],
            "product_page_url": anchor_record["product_page_url"],
            "gender": anchor_record["gender"],
            "is_kids": anchor_record["is_kids"],
            "metal_type": anchor_record["metal_type"],
            "metal_color": anchor_record["metal_color"],
            "gemstone_kind": anchor_record["gemstone_info"].get("kind"),
            "collection_name": anchor_record.get("collection_name"),
            "occasions": anchor_record.get("occasions"),
            "price": anchor_record.get("price"),
        },
        "recommendations": recommendations,
    }

@app.get("/discover")
def discover():
    sample_ids = random.sample(list(VALID_IDS), 3)
    featured = []
    for design_id in sample_ids:
        anchor_record, recommendations = get_matching_set(design_id, CONN, COLLECTION, top_k=5)
        evaluation = run_evaluation(CONN,anchor_record, recommendations)

        featured.append({
            "anchor": {
                "design_id": anchor_record["design_id"],
                "design_name": anchor_record["design_name"],
                "category_type": anchor_record["category_type"],
                "image_url": anchor_record["image_url"],
                "product_page_url": anchor_record["product_page_url"],
            },
            "recommendations": recommendations,
            "evaluation": evaluation,
        })
    return {"featured": featured}

def build_evaluation_prompt(anchor, recommendations):
    def summarize(item):
        return (
            f"- {item['design_name']} ({item['category_type']}): "
            f"{item.get('metal_color', 'unknown')} {item.get('metal_type', 'unknown')}, "
            f"gemstone: {item.get('gemstone_kind') or item.get('gemstone_info', {}).get('kind', 'unknown')}, "
            f"price: ₹{item.get('price', 'unknown')}, "
            f"occasions: {', '.join(item.get('occasions') or []) or 'none listed'}"
        )

    anchor_summary = summarize(anchor)
    recs_summary = "\n".join(summarize(r) for r in recommendations)

    return f"""You are an expert jewellery stylist evaluating whether a recommended set of
        jewellery pieces genuinely works together as a matching set for a customer.

        You will be given one "anchor" product and a set of recommended products
        that were algorithmically selected to pair with it. Evaluate the set
        strictly using only the attributes provided below — do not assume or
        invent details about appearance, quality, or style beyond what is given.

        Anchor product:
        {anchor_summary}

        Recommended set:
        {recs_summary}

        Evaluate the set on these four dimensions, each scored 1-10:

        1. Metal & Finish Consistency — do the metal types and colors across all
        items look cohesive when worn together?
        2. Gemstone & Color Harmony — do the gemstone types (or absence of stones)
        create a coherent visual palette across the set?
        3. Style & Occasion Coherence — do the pieces suit the same general
        occasion and style sensibility (e.g. not mixing everyday minimal pieces
        with heavy bridal pieces)?
        4. Overall Set Quality — holistic judgment of whether a customer would
        perceive this as a deliberately curated matching set.

        For each dimension, give a score (1-10) and one concise sentence of
        reasoning grounded only in the attributes provided.

        Then give an "overall_verdict" (one of: "strong match", "acceptable match",
        "weak match") and one sentence of overall reasoning.

        Respond with ONLY valid JSON, no other text, in exactly this schema:
        {{
        "metal_finish": {{"score": 8, "reasoning": "..."}},
        "gemstone_color": {{"score": 7, "reasoning": "..."}},
        "style_occasion": {{"score": 9, "reasoning": "..."}},
        "overall_quality": {{"score": 8, "reasoning": "..."}},
        "overall_verdict": "strong match",
        "overall_reasoning": "..."
        }}"""

def build_multimodal_prompt(anchor, recommendations):
    def summarize(item):
        return (
            f"- {item['design_name']} ({item['category_type']}): "
            f"{item.get('metal_color', 'unknown')} {item.get('metal_type', 'unknown')}, "
            f"gemstone: {item.get('gemstone_kind') or 'unknown'}, "
            f"price: ₹{item.get('price', 'unknown')}, "
            f"occasions: {', '.join(item.get('occasions') or []) or 'none listed'}"
        )

    anchor_summary = summarize(anchor)
    recs_summary = "\n".join(summarize(r) for r in recommendations)

    return f"""You are an expert jewellery stylist helping a customer decide whether a
        recommended set of jewellery pieces is worth buying together as a matching set.
        You are shown the actual product images below, in this order: first the anchor
        piece the customer is already interested in, then each recommended piece.

        Anchor product attributes:
        {anchor_summary}

        Recommended set attributes:
        {recs_summary}

        Look at the images and reason using both what you see and the attributes above.
        Do not invent details not visible in the images or listed above.

        Write for the customer directly, in a warm, confident, concise tone —
        2-3 sentences maximum, no jewellery-analyst jargon, no bullet lists in the summary text.

        Also identify the single weakest-fitting item in the set, if any (there may be
        none, in which case say so) — name it explicitly and explain briefly why a
        customer might want to treat it as optional rather than essential.

        Score four dimensions 1-10 based on what you see in the images plus the
        attributes: metal_finish, gemstone_color, style_occasion, overall_quality.

        Give an overall_verdict: one of "strong match", "acceptable match", "weak match".

        Respond with ONLY valid JSON, no other text, in exactly this schema:
        {{
        "buyer_summary": "...",
        "weakest_item": "design name, or null if none",
        "weakest_item_reason": "... or null",
        "metal_finish": {{"score": 8, "reasoning": "..."}},
        "gemstone_color": {{"score": 7, "reasoning": "..."}},
        "style_occasion": {{"score": 9, "reasoning": "..."}},
        "overall_quality": {{"score": 8, "reasoning": "..."}},
        "overall_verdict": "strong match"
        }}"""

def run_multimodal_evaluation(conn, anchor, recommendations):
    full_records = get_products_by_ids(conn, [r["design_id"] for r in recommendations])

    enriched_recommendations = []
    for rec in recommendations:
        full = full_records.get(rec["design_id"], {})
        enriched_recommendations.append({
            **rec,
            "metal_type": full.get("metal_type"),
            "metal_color": full.get("metal_color"),
            "gemstone_kind": full.get("gemstone_info", {}).get("kind"),
            "price": full.get("price"),
            "occasions": full.get("occasions"),
        })

    prompt = build_multimodal_prompt(anchor, enriched_recommendations)

    # Build the input list: prompt text first, then one image entry per
    # product (anchor first, then each recommendation), passed directly
    # by public URL -- no download/base64 step needed.
    input_items = [{"type": "text", "text": prompt}]
    for item in [anchor] + enriched_recommendations:
        image_url = item.get("image_url")
        if image_url:
            input_items.append({
                "type": "image",
                "uri": image_url,
                "mime_type": "image/jpeg",
            })

    client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    interaction = client.interactions.create(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        input=input_items,
    )

    raw = interaction.output_text
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def run_evaluation(conn, anchor, recommendations):
    """Shared by /evaluate and /discover. Returns parsed evaluation dict or None on failure."""
    full_records = get_products_by_ids(conn, [r["design_id"] for r in recommendations])

    enriched_recommendations = []
    for rec in recommendations:
        full = full_records.get(rec["design_id"], {})
        enriched_recommendations.append({
            **rec,
            "metal_type": full.get("metal_type"),
            "metal_color": full.get("metal_color"),
            "gemstone_kind": full.get("gemstone_info", {}).get("kind"),
            "price": full.get("price"),
            "occasions": full.get("occasions"),
        })

    prompt = build_evaluation_prompt(anchor, enriched_recommendations)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    raw = response.choices[0].message.content
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

@app.get("/evaluate")
def evaluate(design_id: int = Query(...), top_k: int = Query(5)):
    anchor = get_product_by_id(CONN, design_id)
    if anchor is None:
        raise HTTPException(status_code=404, detail=f"design_id {design_id} not found")

    anchor_record, recommendations = get_matching_set(design_id, CONN, COLLECTION, top_k=top_k)
    evaluation = run_multimodal_evaluation(CONN, anchor_record, recommendations)

    if evaluation is None:
        raise HTTPException(status_code=502, detail="LLM returned an unparseable evaluation")

    return {"evaluation": evaluation}  

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")