import argparse
from pipeline.frame_extractor import run_extraction
from pipeline.embedding_engine import compute_embeddings
from pipeline.similarity_engine import run_similarity


def main(mode: str = "interval", interval_sec: float = 2.0, n_queries: int = 3):
    print("=== Step 1: Frame Extraction ===")
    metadata_csv = run_extraction(
        videos_dir="data/videos",
        frames_dir="data/frames",
        mode=mode,
        interval_sec=interval_sec
    )

    print("\n=== Step 2: Embedding Computation ===")
    compute_embeddings(
        metadata_csv=metadata_csv,
        embeddings_dir="embeddings"
    )

    print("\n=== Step 3: Similarity Analysis ===")
    run_similarity(
        embeddings_dir="embeddings",
        outputs_dir="outputs",
        n_queries=n_queries
    )

    print("\n✅ Pipeline complete. Check outputs/ for results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="interval", choices=["interval", "scene"])
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--queries", type=int, default=3)
    args = parser.parse_args()
    main(mode=args.mode, interval_sec=args.interval, n_queries=args.queries)
