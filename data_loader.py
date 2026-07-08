"""
Loads all normalized records + their embeddings into memory, and builds a
FAISS flat index over the embeddings for similarity lookup.

Single entry point: load_all(). Downstream scripts (matching logic, UI)
call this once and work off the returned in-memory structure -- no one
else re-implements file reading.
"""

import json
import os

import numpy as np
import faiss

import config

NORMALIZED_DIR = os.path.join(config.DATA_DIR, "normalized")
EMBEDDINGS_DIR = os.path.join(config.DATA_DIR, "embeddings")


def load_all():
    """
    Returns:
        products: list of dicts, each = normalized fields + 'embedding' (np.array)
        index: a FAISS IndexFlatIP (inner product = cosine similarity, since
               embeddings are pre-normalized in generate_embeddings.py)
        id_order: list of design_ids in the same order as vectors in the index,
                  so index position -> design_id lookup is possible
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

    embedding_dim = products[0]["embedding"].shape[0]
    index = faiss.IndexFlatIP(embedding_dim)  # inner product == cosine similarity for normalized vectors

    vectors = np.stack([p["embedding"] for p in products]).astype("float32")
    index.add(vectors)

    id_order = [p["design_id"] for p in products]

    print(f"[data_loader] Loaded {len(products)} product(s) into memory and FAISS index")
    return products, index, id_order


def get_product_by_id(products, design_id):
    """Convenience lookup -- linear scan, fine at this scale."""
    for p in products:
        if p["design_id"] == design_id:
            return p
    return None


if __name__ == "__main__":
    # Quick smoke test when run directly
    products, index, id_order = load_all()
    print(f"Index contains {index.ntotal} vectors")
    print(f"Sample product keys: {list(products[0].keys())}")