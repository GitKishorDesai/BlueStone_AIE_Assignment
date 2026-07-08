"""
Builds a persistent Chroma vector collection from normalized records +
their embeddings. Run once after normalize.py / generate_embeddings.py
have produced their outputs (or any time embeddings change).

Stores metadata (category_type, gender, is_kids, design_id) alongside
each vector, so retrieval-time queries can filter directly in Chroma
rather than only in Python afterward.
"""

import json
import os

import chromadb

import config

NORMALIZED_DIR = os.path.join(config.DATA_DIR, "normalized")
EMBEDDINGS_DIR = os.path.join(config.DATA_DIR, "embeddings")
CHROMA_DIR = os.path.join(config.DATA_DIR, "chroma_db")

COLLECTION_NAME = "jewellery_products"


def load_normalized_records():
    records = []
    for fname in os.listdir(NORMALIZED_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(NORMALIZED_DIR, fname), encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def main():
    records = load_normalized_records()
    print(f"[build_vector_index] Found {len(records)} normalized record(s)")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh build each run -- drop and recreate to avoid stale/duplicate entries
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    ids, embeddings, metadatas = [], [], []
    skipped = []

    for record in records:
        design_id = record["design_id"]
        embedding_path = os.path.join(EMBEDDINGS_DIR, f"{design_id}.npy")

        if not os.path.exists(embedding_path):
            skipped.append(design_id)
            continue

        import numpy as np
        embedding = np.load(embedding_path)

        ids.append(str(design_id))
        embeddings.append(embedding.tolist())
        metadatas.append({
            "design_id": design_id,
            "category_type": record["category_type"],
            "gender": record["gender"],
            "is_kids": record["is_kids"],
        })

    if skipped:
        print(f"[build_vector_index] Skipped {len(skipped)} record(s) with no embedding: {skipped}")

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"[build_vector_index] Indexed {len(ids)} product(s) into Chroma collection '{COLLECTION_NAME}'")
    print(f"[build_vector_index] Persisted to {CHROMA_DIR}")


if __name__ == "__main__":
    main()