"""
Builds a persistent Chroma vector collection by reading embeddings
directly from SQLite (no intermediate .npy files).
"""

import json
import os
import sqlite3

import numpy as np
import chromadb

import config

DB_PATH = os.path.join(config.DATA_DIR, "products.db")
CHROMA_DIR = os.path.join(config.DATA_DIR, "chroma_db")
COLLECTION_NAME = "jewellery_products"
BATCH_SIZE = 5000


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT design_id, data, embedding FROM products WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    conn.close()

    print(f"[build_vector_index] Found {len(rows)} product(s) with embeddings")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    ids, embeddings, metadatas = [], [], []
    for design_id, data_str, embedding_bytes in rows:
        record = json.loads(data_str)
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        ids.append(str(design_id))
        embeddings.append(embedding.tolist())
        metadatas.append({
            "design_id": design_id,
            "category_type": record["category_type"],
            "canonical_category": config.canonicalize_category(record["category_type"]),
            "gender": record["gender"],
            "is_kids": record["is_kids"],
        })

    for i in range(0, len(ids), BATCH_SIZE):
        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            embeddings=embeddings[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
        )
        print(f"[build_vector_index] Added batch {i}-{i + BATCH_SIZE} of {len(ids)}")

    print(f"[build_vector_index] Indexed {len(ids)} product(s) into Chroma collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()