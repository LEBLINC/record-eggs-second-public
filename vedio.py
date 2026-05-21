# -*- coding: utf-8 -*-

import os
import cv2

# --- Configuration ---
INPUT_DIR = r"C:\Users\12179\Desktop\VEDIO"
VIDEO_FILES = ["test.mp4", "test0.mp4", "test1.mp4"]
OUTPUT_DIR = r"C:\Users\12179\Desktop\dataset"
TARGET_FPS = 5.0  # extract 5 frames per second

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def extract_frames_5fps(video_path: str, output_dir: str, target_fps: float = 5.0) -> int:
    """
    Extract frames at target_fps from the given video and save into output_dir.
    Returns number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    # Prepare naming
    base = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Time-based sampling: save every 1/target_fps seconds
    interval_ms = 1000.0 / target_fps
    next_save_time = 0.0
    saved = 0
    frame_index = 0

    # Try reading FPS for fallback timestamp calculation if POS_MSEC is unreliable
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 1e-6:
        src_fps = 0.0  # treat as unknown

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prefer OpenCV's position in milliseconds; fallback to frame_index / src_fps
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if (timestamp_ms is None or timestamp_ms <= 0.0) and src_fps > 0.0:
            timestamp_ms = (frame_index * 1000.0) / src_fps

        # Save if we've passed the next target time
        if timestamp_ms >= next_save_time:
            saved += 1
            out_name = f"{base}_{saved:06d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            # Save as JPEG; you can adjust quality if needed: cv2.IMWRITE_JPEG_QUALITY, 95
            ok = cv2.imwrite(out_path, frame)
            if not ok:
                print(f"[WARN] Failed to write frame to {out_path}")
            next_save_time += interval_ms

        frame_index += 1

    cap.release()
    return saved

def main():
    print("[INFO] Starting frame extraction at 5 FPS ...")
    ensure_dir(OUTPUT_DIR)

    total_saved = 0
    for name in VIDEO_FILES:
        video_path = os.path.join(INPUT_DIR, name)
        if not os.path.isfile(video_path):
            print(f"[WARN] File not found: {video_path}")
            continue

        print(f"[INFO] Processing: {video_path}")
        saved = extract_frames_5fps(video_path, OUTPUT_DIR, TARGET_FPS)
        total_saved += saved
        print(f"[INFO] Saved {saved} frames from {name}")

    print(f"[DONE] All finished. Total frames saved: {total_saved}")
    print(f"[DONE] Output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
