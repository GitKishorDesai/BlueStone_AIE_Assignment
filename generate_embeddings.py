"""
Computes a CLIP embedding for each downloaded product image and caches it
to data/embeddings/{design_id}.npy.

Assumes images already exist locally (see download_images.py). This keeps
this script's dependencies purely about the model/embedding computation,
not networking.
"""

import os

import numpy as np
import torch
from PIL import Image
import open_clip

import config

IMAGES_DIR = os.path.join(config.DATA_DIR, "images")
EMBEDDINGS_DIR = os.path.join(config.DATA_DIR, "embeddings")
FAILED_LOG = os.path.join(config.LOG_DIR, "embedding_failures.txt")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

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
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize for cosine similarity

    return embedding.squeeze(0).numpy()


def main():
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[embeddings] Found {len(image_files)} image(s) in {IMAGES_DIR}")

    model, preprocess = load_model()

    processed = 0
    skipped_existing = 0
    failed = []

    for fname in image_files:
        design_id = os.path.splitext(fname)[0]
        embedding_path = os.path.join(EMBEDDINGS_DIR, f"{design_id}.npy")

        if os.path.exists(embedding_path):
            skipped_existing += 1
            continue

        image_path = os.path.join(IMAGES_DIR, fname)
        embedding = compute_embedding(model, preprocess, image_path)

        if embedding is None:
            failed.append(design_id)
            continue

        np.save(embedding_path, embedding)
        processed += 1
        print(f"[embeddings] ({processed}) designId={design_id}: embedding saved")

    if failed:
        with open(FAILED_LOG, "w") as f:
            for design_id in failed:
                f.write(f"{design_id}\n")

    print(f"\n[embeddings] Done. {processed} new embedding(s) computed, "
          f"{skipped_existing} already existed, {len(failed)} failed.")
    if failed:
        print(f"[embeddings] See {FAILED_LOG} for failure details.")


if __name__ == "__main__":
    main()