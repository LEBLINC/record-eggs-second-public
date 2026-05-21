import os
import cv2
import argparse
import pathlib
import math
import shutil
import random


def list_target_videos(input_dir: str, videos: list[str]) -> list[str]:
    paths = []
    for name in videos:
        # 允许传入无扩展名的基名或完整文件名
        if not name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            candidates = [f"{name}.mp4", f"{name}.MP4", f"{name}.avi", f"{name}.AVI"]
        else:
            candidates = [name]

        found = None
        for cand in candidates:
            p = os.path.join(input_dir, cand)
            if os.path.isfile(p):
                found = p
                break
        if found is None:
            print(f"[WARN] 未找到视频: {name}（在 {input_dir}）")
            continue
        paths.append(found)
    return paths


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_frames_from_video(video_path: str, out_dir: str, target_fps: float = 3.0, jpeg_quality: int = 90) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return 0

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    if source_fps is None or source_fps <= 0 or math.isnan(source_fps):
        # 回退：按帧计数取样
        source_fps = target_fps
    step = max(1, int(round(source_fps / max(0.1, target_fps))))

    ensure_dir(out_dir)
    frame_idx = 0
    saved = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % step == 0:
            ret2, frame = cap.retrieve()
            if not ret2 or frame is None:
                frame_idx += 1
                continue
            out_name = os.path.join(out_dir, f"frame_{saved:08d}.jpg")
            cv2.imwrite(out_name, frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            saved += 1
        frame_idx += 1

        # 可选：若总帧未知，避免无限循环
        if total_frames > 0 and frame_idx >= total_frames:
            break

    cap.release()
    print(f"[INFO] {os.path.basename(video_path)} -> 抽帧 {saved} 张 @ {target_fps} fps 到 {out_dir}")
    return saved


def split_dataset(images_raw_root: str, dataset_root: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    images = []
    for dirpath, _, filenames in os.walk(images_raw_root):
        for fn in filenames:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(dirpath, fn))

    random.Random(seed).shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    for split in ['train', 'val', 'test']:
        ensure_dir(os.path.join(dataset_root, 'images', split))
        ensure_dir(os.path.join(dataset_root, 'labels', split))  # 占位，等待标注后填充

    for split, paths in splits.items():
        dst_dir = os.path.join(dataset_root, 'images', split)
        for src in paths:
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy2(src, dst)
    print(f"[INFO] 数据划分完成: train={n_train}, val={n_val}, test={n_test}")


def write_data_yaml(dataset_root: str):
    yaml_path = os.path.join(dataset_root, 'data.yaml')
    content = (
        f"path: {dataset_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: egg\n"
        f"  1: qr\n"
    )
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[INFO] 生成数据配置: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='从视频抽帧并构建YOLO数据集骨架')
    parser.add_argument('--input-dir', required=True, help='包含视频文件的目录')
    parser.add_argument('--videos', nargs='+', required=True, help='视频文件名或基名（可省略扩展名）')
    parser.add_argument('--output-dir', default=os.path.join('data', 'dataset_eggs'), help='数据集根目录')
    parser.add_argument('--fps', type=float, default=3.0, help='抽帧目标fps')
    parser.add_argument('--jpeg-quality', type=int, default=90, help='JPEG质量(1-100)')
    parser.add_argument('--split', action='store_true', help='是否立即按8/1/1切分到images/train|val|test')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    dataset_root = os.path.abspath(args.output_dir)
    images_raw_root = os.path.join(dataset_root, 'images_raw')

    ensure_dir(dataset_root)
    ensure_dir(images_raw_root)

    videos = list_target_videos(input_dir, args.videos)
    if not videos:
        print('[ERROR] 未找到任何有效视频，退出')
        return

    total_saved = 0
    for vp in videos:
        stem = pathlib.Path(vp).stem
        out_dir = os.path.join(images_raw_root, stem)
        saved = extract_frames_from_video(vp, out_dir, target_fps=args.fps, jpeg_quality=args.jpeg_quality)
        total_saved += saved

    print(f"[INFO] 抽帧完成，总计保存 {total_saved} 张")

    if args.split:
        split_dataset(images_raw_root, dataset_root, 0.8, 0.1, 0.1, seed=42)

    write_data_yaml(dataset_root)
    print('[DONE] 数据集骨架就绪。请进行标注或转换后再训练。')


if __name__ == '__main__':
    main()















