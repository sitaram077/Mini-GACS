import os
import json
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import csv


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def load_clip_model(device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor, device


def compute_embeddings(metadata_csv: str, embeddings_dir: str, batch_size: int = 16) -> tuple[np.ndarray, list[dict]]:
    model, processor, device = load_clip_model()
    os.makedirs(embeddings_dir, exist_ok=True)

    with open(metadata_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    valid_rows, all_embeddings = [], []

    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        images = []

        for row in batch:
            try:
                img = Image.open(row["file_path"]).convert("RGB")
                images.append((row, img))
            except Exception as e:
                print(f"[embedder] Skipping {row['file_path']}: {e}")

        if not images:
            continue

        batch_rows, batch_imgs = zip(*images)
        inputs = processor(images=list(batch_imgs), return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            # L2 normalize for cosine similarity
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())
        valid_rows.extend(batch_rows)
        print(f"[embedder] Processed batch {i // batch_size + 1} | {len(valid_rows)} frames embedded")

    embeddings = np.vstack(all_embeddings)

    # Verification
    assert embeddings.shape[1] == 512, f"Expected 512-dim embeddings, got {embeddings.shape[1]}"
    assert not np.isnan(embeddings).any(), "NaN values detected in embeddings"

    npy_path = os.path.join(embeddings_dir, "frame_embeddings.npy")
    index_path = os.path.join(embeddings_dir, "frame_index.json")

    np.save(npy_path, embeddings)
    with open(index_path, "w") as f:
        json.dump([dict(idx=i, **row) for i, row in enumerate(valid_rows)], f, indent=2)

    print(f"[embedder] Embeddings shape: {embeddings.shape} | Saved to {npy_path}")
    return embeddings, valid_rows
