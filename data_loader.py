"""
Loads all normalized records + their embeddings into memory, and connects
to the persistent Chroma vector collection (built by build_vector_index.py)
for similarity-based retrieval.

Single entry point: load_all(). Downstream scripts (matching logic, UI)
call this once and work off the returned in-memory structure -- no one
else re-implements file reading.
"""

import json
import os

import numpy as np
import chromadb

import config

NORMALIZED_DIR = os.path.join(config.DATA_DIR, "normalized")
EMBEDDINGS_DIR = os.path.join(config.DATA_DIR, "embeddings")
CHROMA_DIR = os.path.join(config.DATA_DIR, "chroma_db")
COLLECTION_NAME = "jewellery_products"


def load_all():
    """
    Returns:
        products: list of dicts, each = normalized fields + 'embedding' (np.array)
        collection: the Chroma collection object, ready for similarity queries
        id_order: kept for backward compatibility -- list of all design_ids loaded
    """
    products = []
    skipped_no_embedding = []

    for fname in os.listdir(NORMALIZED_DIR):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(NORMALIZED_DIR, fname), encoding="utf-8") as f:
            record = json.load(f)

        design_id = record["design_id"]
        embedding_path = os.path.join(EMBEDDINGS_DIR, f"{design_id}.npy")

        if not os.path.exists(embedding_path):
            skipped_no_embedding.append(design_id)
            continue

        embedding = np.load(embedding_path)
        record["embedding"] = embedding
        products.append(record)

    if skipped_no_embedding:
        print(f"[data_loader] Skipped {len(skipped_no_embedding)} product(s) with no embedding: "
              f"{skipped_no_embedding}")

    if not products:
        raise RuntimeError("No products with embeddings found -- run the pipeline scripts first.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    id_order = [p["design_id"] for p in products]

    print(f"[data_loader] Loaded {len(products)} product(s) into memory; "
          f"connected to Chroma collection with {collection.count()} vector(s)")
    return products, collection, id_order


def get_product_by_id(products, design_id):
    """Convenience lookup -- linear scan, fine at this scale."""
    for p in products:
        if p["design_id"] == design_id:
            return p
    return None


if __name__ == "__main__":
    products, collection, id_order = load_all()
    print(f"Collection contains {collection.count()} vectors")
    print(f"Sample product keys: {list(products[0].keys())}")