import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


def load_embeddings(embeddings_dir: str) -> tuple[np.ndarray, list[dict]]:
    npy_path = os.path.join(embeddings_dir, "frame_embeddings.npy")
    index_path = os.path.join(embeddings_dir, "frame_index.json")

    if not os.path.exists(npy_path) or not os.path.exists(index_path):
        raise FileNotFoundError("Embeddings or index file not found.")

    embeddings = np.load(npy_path)
    with open(index_path) as f:
        index = json.load(f)

    assert not np.isnan(embeddings).any(), "NaN detected in loaded embeddings"
    return embeddings, index


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    sim_matrix = cosine_similarity(embeddings)
    assert sim_matrix.shape == (len(embeddings), len(embeddings)), "Similarity matrix shape mismatch"
    return sim_matrix


def get_top_k_similar(query_idx: int, sim_matrix: np.ndarray, index: list[dict], k: int = 5) -> list[dict]:
    scores = sim_matrix[query_idx].copy()
    scores[query_idx] = -1  # exclude self
    top_indices = np.argsort(scores)[::-1][:k]
    return [{"rank": r + 1, "idx": int(i), "score": round(float(scores[i]), 4), **index[i]} for r, i in enumerate(top_indices)]


def save_heatmap(sim_matrix: np.ndarray, index: list[dict], output_path: str):
    labels = [f"{e['video_id']}_{e['idx']}" for e in index]
    plt.figure(figsize=(max(10, len(labels) // 2), max(8, len(labels) // 2)))
    sns.heatmap(sim_matrix, xticklabels=False, yticklabels=False, cmap="viridis", vmin=0, vmax=1)
    plt.title("Frame Vibe Similarity Heatmap (Cosine)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[similarity] Heatmap saved to {output_path}")


def save_topk_report(query_indices: list[int], sim_matrix: np.ndarray, index: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_idx", "query_file", "rank", "match_idx", "match_file", "score"])
        writer.writeheader()
        for q_idx in query_indices:
            results = get_top_k_similar(q_idx, sim_matrix, index, k=5)
            for r in results:
                writer.writerow({
                    "query_idx": q_idx,
                    "query_file": index[q_idx]["file_path"],
                    "rank": r["rank"],
                    "match_idx": r["idx"],
                    "match_file": r["file_path"],
                    "score": r["score"]
                })
    print(f"[similarity] Top-K report saved to {output_path}")


def run_similarity(embeddings_dir: str, outputs_dir: str, n_queries: int = 3):
    embeddings, index = load_embeddings(embeddings_dir)
    sim_matrix = compute_similarity_matrix(embeddings)

    save_heatmap(sim_matrix, index, os.path.join(outputs_dir, "similarity_heatmap.png"))

    # Pick evenly spaced query frames across the dataset
    query_indices = np.linspace(0, len(index) - 1, n_queries, dtype=int).tolist()
    save_topk_report(query_indices, sim_matrix, index, os.path.join(outputs_dir, "topk_similarity.csv"))

    print(f"[similarity] Done. Queries used: {query_indices}")
