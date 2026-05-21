# coding:utf-8
"""
将 data/capture_six 下的各 camera_* 目录中的视频抽帧到 data/beijing_data。
默认抽帧频率：2 FPS。

用法示例：
    python tools/extract_frames_2fps.py
    # 或自定义路径/频率
    python tools/extract_frames_2fps.py --src data/capture_six --dst data/beijing_data --fps 2
"""
import argparse
import os
from pathlib import Path
import cv2


def extract_video(video_path: Path, out_dir: Path, target_fps: float) -> int:
    """
    将单个视频按 target_fps 抽帧保存到 out_dir，返回保存的帧数。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 打不开视频: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = target_fps  # 回退
    frame_interval = max(int(round(src_fps / target_fps)), 1)

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx % frame_interval == 0:
            fname = f"{video_path.stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_dir / fname), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[INFO] {video_path.name}: 抽帧间隔={frame_interval}，保存 {saved} 张 -> {out_dir}")
    return saved


def main(src_root: str, dst_root: str, target_fps: float):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_root.exists():
        raise FileNotFoundError(f"源目录不存在: {src_root}")

    total_saved = 0
    # 遍历 camera_* 子目录
    for cam_dir in sorted(src_root.glob("camera_*")):
        if not cam_dir.is_dir():
            continue
        videos = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
        if not videos:
            print(f"[INFO] {cam_dir.name}: 无视频文件，跳过")
            continue

        out_dir = dst_root / cam_dir.name
        for v in videos:
            total_saved += extract_video(v, out_dir, target_fps)

    print(f"[DONE] 抽帧完成，总计保存 {total_saved} 张图片，输出目录: {dst_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/capture_six", help="源视频目录（包含 camera_* 子目录）")
    parser.add_argument("--dst", default="data/beijing_data", help="输出图片根目录")
    parser.add_argument("--fps", type=float, default=2.0, help="目标抽帧频率 (frames per second)")
    args = parser.parse_args()

    main(args.src, args.dst, args.fps)

