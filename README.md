# GenTA GACS Prototype — Mood & Style Embedding Pipeline

A minimal end-to-end affective computing pipeline built as a competency assessment for GenTA's AI R&D Engineer role. The system ingests art/marketing videos, extracts representative frames, computes CLIP-based mood/style embeddings, and visualizes vibe similarity across videos.

## Project Structure

```
genta-gacs-prototype/
├── data/
│   ├── videos/          # place input .mp4 files here
│   ├── frames/          # auto-generated extracted frames + metadata.csv
│   └── sources.json     # video licensing info
├── embeddings/          # frame_embeddings.npy + frame_index.json
├── outputs/             # similarity_heatmap.png + topk_similarity.csv
├── pipeline/
│   ├── frame_extractor.py
│   ├── embedding_engine.py
│   ├── similarity_engine.py
│   └── tests/
│       └── test_pipeline.py
├── run_pipeline.py
├── requirements.txt
├── AI_USAGE.md
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/genta-gacs-prototype
cd genta-gacs-prototype
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch>=2.1.0 torchvision>=0.16.0 transformers>=4.38.0 opencv-python>=4.9.0 scenedetect>=0.6.3 Pillow>=10.2.0 numpy>=1.26.0 scikit-learn>=1.4.0 matplotlib>=3.8.0 seaborn>=0.13.0 tqdm>=4.66.0 pytest>=8.0.0
```

## Video Data

Download the three sample videos and place them in `data/videos/` as:
- `art_gallery.mp4` — [Sergey Semenov, Pixabay](https://pixabay.com/videos/id-43328/) — Pixabay Content License
- `artist.mp4` — [cottonbro studio, Pexels](https://www.pexels.com/video/3795833/) — Pexels License
- `painting.mp4` — [Nicole Learner, Pexels](https://www.pexels.com/video/6189526/) — Pexels License

## Run

```bash
# Default: interval-based frame extraction every 2 seconds
python run_pipeline.py

# Scene-detection mode
python run_pipeline.py --mode scene

# Custom interval and query count
python run_pipeline.py --mode interval --interval 3.0 --queries 5
```

## Test

```bash
pytest pipeline/tests/ -v
```

## Outputs

| File | Description |
|---|---|
| `data/frames/metadata.csv` | Frame index with video_id, timestamp, file path |
| `embeddings/frame_embeddings.npy` | (59, 512) CLIP embedding matrix |
| `embeddings/frame_index.json` | Index mapping embedding rows to frame metadata |
| `outputs/similarity_heatmap.png` | Pairwise cosine similarity heatmap across all frames |
| `outputs/topk_similarity.csv` | Top-5 most similar frames for 3 query frames |

## Results Summary

All three query frames (one per video) retrieved their top-5 matches exclusively from within their own video, confirming that CLIP captures distinct video-level vibe signatures. Intra-video similarity ranged from 0.88–0.97, with `painting` showing the tightest stylistic coherence and `art_gallery` showing the most visual diversity.


