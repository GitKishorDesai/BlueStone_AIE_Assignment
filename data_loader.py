"""
Connects to the SQLite product database and the persistent Chroma vector
collection. No bulk in-memory loading -- product records are fetched
on-demand per query, keeping the app's memory footprint independent of
catalog size.
"""

import json
import os
import sqlite3

import chromadb

import config

CHROMA_DIR = os.path.join(config.DATA_DIR, "chroma_db")
DB_PATH = os.path.join(config.DATA_DIR, "products.db")
COLLECTION_NAME = "jewellery_products"


def load_all():
    """
    Returns:
        conn: sqlite3 connection for on-demand product lookups
        collection: Chroma collection for similarity queries
        valid_ids: set of design_ids present in the Chroma collection
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    all_chroma_ids = collection.get(include=[])["ids"]
    valid_ids = set(int(i) for i in all_chroma_ids)

    print(f"[data_loader] Connected to SQLite ({DB_PATH}) and Chroma "
          f"({len(valid_ids)} product(s) with embeddings)")
    return conn, collection, valid_ids


def get_product_by_id(conn, design_id):
    cur = conn.cursor()
    cur.execute("SELECT data FROM products WHERE design_id = ?", (design_id,))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None


def get_products_by_ids(conn, design_ids):
    """Batch fetch -- used for the Chroma-retrieved candidate shortlist."""
    if not design_ids:
        return {}
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in design_ids)
    cur.execute(f"SELECT design_id, data FROM products WHERE design_id IN ({placeholders})", design_ids)
    return {row[0]: json.loads(row[1]) for row in cur.fetchall()}


def list_all_products(conn):
    """Lightweight listing for the UI dropdown -- no JSON parsing needed."""
    cur = conn.cursor()
    cur.execute("SELECT design_id, design_name, category_type FROM products")
    return [
        {"design_id": row[0], "design_name": row[1], "category_type": row[2]}
        for row in cur.fetchall()
    ]


def count_products(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM products")
    return cur.fetchone()[0]


if __name__ == "__main__":
    conn, collection, valid_ids = load_all()
    print(f"Total products in DB: {count_products(conn)}")
    print(f"Products with embeddings (Chroma): {len(valid_ids)}")