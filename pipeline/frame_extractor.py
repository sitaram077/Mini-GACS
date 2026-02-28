import cv2
import os
import json
import csv
from pathlib import Path

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False


def extract_frames_interval(video_path: str, output_dir: str, video_id: str, interval_sec: float = 2.0) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps * interval_sec))
    os.makedirs(output_dir, exist_ok=True)

    metadata, frame_count, saved_count = [], 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            filename = f"{video_id}_frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            metadata.append({
                "video_id": video_id,
                "timestamp_sec": round(timestamp, 3),
                "scene_id": None,
                "file_path": filepath
            })
            saved_count += 1
        frame_count += 1

    cap.release()
    return metadata


def extract_frames_scene(video_path: str, output_dir: str, video_id: str) -> list[dict]:
    if not SCENEDETECT_AVAILABLE:
        raise ImportError("scenedetect not installed. Use interval mode.")

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for scene_id, (start, end) in enumerate(scenes):
        mid_frame = (start.get_frames() + end.get_frames()) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = mid_frame / fps
        filename = f"{video_id}_scene_{scene_id:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        metadata.append({
            "video_id": video_id,
            "timestamp_sec": round(timestamp, 3),
            "scene_id": scene_id,
            "file_path": filepath
        })

    cap.release()
    return metadata


def save_metadata(metadata: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "timestamp_sec", "scene_id", "file_path"])
        writer.writeheader()
        writer.writerows(metadata)


def run_extraction(videos_dir: str, frames_dir: str, mode: str = "interval", interval_sec: float = 2.0) -> str:
    all_metadata = []
    video_files = sorted(Path(videos_dir).glob("*.mp4"))

    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {videos_dir}")

    for video_path in video_files:
        video_id = video_path.stem
        out_dir = os.path.join(frames_dir, video_id)

        if mode == "scene" and SCENEDETECT_AVAILABLE:
            meta = extract_frames_scene(str(video_path), out_dir, video_id)
        else:
            meta = extract_frames_interval(str(video_path), out_dir, video_id, interval_sec)

        all_metadata.extend(meta)
        print(f"[extractor] {video_id}: {len(meta)} frames extracted")

    metadata_path = os.path.join(frames_dir, "metadata.csv")
    save_metadata(all_metadata, metadata_path)

    # Verify frame count matches metadata
    assert len(all_metadata) > 0, "No frames were extracted"
    assert all(os.path.exists(m["file_path"]) for m in all_metadata), "Some frame files are missing on disk"

    print(f"[extractor] Total frames: {len(all_metadata)} | Metadata saved to {metadata_path}")
    return metadata_path
