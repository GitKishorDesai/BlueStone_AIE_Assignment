"""
Computes a CLIP embedding for each product's downloaded image and stores
it directly in SQLite (as raw bytes), rather than separate .npy files.
"""

import os
import sqlite3

import numpy as np
import torch
from PIL import Image
import open_clip

import config

DB_PATH = os.path.join(config.DATA_DIR, "products.db")
IMAGES_DIR = os.path.join(config.DATA_DIR, "images")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def load_model():
    print("[embeddings] Loading CLIP model (first run may take a minute to download weights)...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    return model, preprocess


def compute_embedding(model, preprocess, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[embeddings] Could not open image {image_path}: {e}")
        return None

    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.squeeze(0).numpy()


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT design_id FROM products WHERE embedding IS NULL")
    design_ids = [row[0] for row in cur.fetchall()]

    print(f"[embeddings] Found {len(design_ids)} product(s) needing an embedding")

    model, preprocess = load_model()

    processed = 0
    failed = []

    for design_id in design_ids:
        image_path = os.path.join(IMAGES_DIR, f"{design_id}.jpg")
        if not os.path.exists(image_path):
            failed.append((design_id, "image_not_found"))
            continue

        embedding = compute_embedding(model, preprocess, image_path)
        if embedding is None:
            failed.append((design_id, "embedding_computation_failed"))
            continue

        cur.execute("UPDATE products SET embedding = ? WHERE design_id = ?",
                     (embedding.astype(np.float32).tobytes(), design_id))
        processed += 1
        if processed % 200 == 0:
            conn.commit()
            print(f"[embeddings] ({processed}/{len(design_ids)}) checkpoint saved")

    conn.commit()
    conn.close()

    if failed:
        with open(os.path.join(config.LOG_DIR, "embedding_failures.txt"), "w") as f:
            for design_id, reason in failed:
                f.write(f"{design_id},{reason}\n")

    print(f"\n[embeddings] Done. {processed} embedding(s) computed, {len(failed)} failed.")


if __name__ == "__main__":
    main()