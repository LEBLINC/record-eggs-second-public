import os
import math
import cv2


def extract_video(video_path, out_root, fps_target=3):
    """从单个视频按目标 FPS 抽帧并保存到子目录。"""
    os.makedirs(out_root, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 打不开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = fps / fps_target if fps_target > 0 else fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if interval <= 1 or idx % math.ceil(interval) == 0:
            out_path = os.path.join(out_dir, f"{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"[OK] {video_path} → {out_dir}, 保存 {saved} 张 (总帧 {total})")


def batch_extract(in_dir, out_dir, fps_target=3):
    """遍历目录抽取所有视频的帧。"""
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    for root, _, files in os.walk(in_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                extract_video(os.path.join(root, f), out_dir, fps_target)


if __name__ == "__main__":
    # 修改为你的输入/输出路径
    in_dir = r"D:\path\to\videos"
    out_dir = r"D:\path\to\frames"
    batch_extract(in_dir, out_dir, fps_target=3)

