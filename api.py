"""
FastAPI backend for the jewellery matching-set UI.

Connects to SQLite (product records) and Chroma (vector index) once at
startup -- no bulk in-memory loading. Exposes:
  GET /products             -> list of {design_id, design_name, category_type}
  GET /recommend             -> anchor + top-K recommendations
  GET /about                 -> models/prompts used, for transparency
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from data_loader import load_all, get_product_by_id, list_all_products
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

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")