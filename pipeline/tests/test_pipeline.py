import numpy as np
import os
import json
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock


EMBEDDINGS_DIR = "embeddings"
FRAMES_DIR = "data/frames"


def test_metadata_csv_exists():
    path = os.path.join(FRAMES_DIR, "metadata.csv")
    assert os.path.exists(path), "metadata.csv not found"


def test_frame_files_exist():
    import csv
    meta_path = os.path.join(FRAMES_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        pytest.skip("metadata.csv not yet generated")
    with open(meta_path) as f:
        rows = list(csv.DictReader(f))
    missing = [r["file_path"] for r in rows if not os.path.exists(r["file_path"])]
    assert len(missing) == 0, f"Missing frame files: {missing[:5]}"


def test_embedding_shapes():
    npy_path = os.path.join(EMBEDDINGS_DIR, "frame_embeddings.npy")
    if not os.path.exists(npy_path):
        pytest.skip("Embeddings not yet generated")
    embeddings = np.load(npy_path)
    assert embeddings.ndim == 2, "Embeddings should be 2D"
    assert embeddings.shape[1] == 512, f"Expected 512 dims, got {embeddings.shape[1]}"


def test_no_nan_in_embeddings():
    npy_path = os.path.join(EMBEDDINGS_DIR, "frame_embeddings.npy")
    if not os.path.exists(npy_path):
        pytest.skip("Embeddings not yet generated")
    embeddings = np.load(npy_path)
    assert not np.isnan(embeddings).any(), "NaN values found in embeddings"


def test_identical_image_similarity():
    from pipeline.embedding_engine import load_clip_model
    import torch
    from transformers import CLIPProcessor

    model, processor, device = load_clip_model()
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    inputs = processor(images=[img, img], return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    sim = float((features[0] @ features[1]).item())
    assert sim >= 0.9999, f"Identical images should have sim~1.0, got {sim}"


def test_topk_returns_five():
    npy_path = os.path.join(EMBEDDINGS_DIR, "frame_embeddings.npy")
    index_path = os.path.join(EMBEDDINGS_DIR, "frame_index.json")
    if not os.path.exists(npy_path):
        pytest.skip("Embeddings not yet generated")

    from pipeline.similarity_engine import compute_similarity_matrix, get_top_k_similar
    embeddings = np.load(npy_path)
    with open(index_path) as f:
        index = json.load(f)

    sim_matrix = compute_similarity_matrix(embeddings)
    results = get_top_k_similar(0, sim_matrix, index, k=5)
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
