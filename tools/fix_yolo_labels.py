# coding: utf-8
"""
批量修复 YOLO 检测标签（将6列转为5列，必要时归一化）。

使用场景：
- CVAT 导出“Ultralytics YOLO Detection Track 1.0”可能包含 track_id，形成 6 列：cls track_id x y w h
- YOLO 检测训练仅接受 5 列：cls x_center y_center width height（均需归一化到 0~1）

参数：
- --root 数据集根目录，内部应包含 images/train 与 labels/train

示例：
    python tools/fix_yolo_labels.py --root "data/eggs_ultralytics_yolo_20251021 (1)"
"""
import argparse
import glob
import os
import cv2


def fix_one(lbl_path: str, img_dir: str) -> bool:
    """修复单个标签文件，返回是否发生修改"""
    name = os.path.splitext(os.path.basename(lbl_path))[0]
    img_path = os.path.join(img_dir, name + ".jpg")
    if not os.path.exists(img_path):
        print(f"[WARN] image missing for {lbl_path}")
        return False

    img = cv2.imread(img_path)
    h_img = w_img = None
    if img is not None:
        h_img, w_img = img.shape[:2]

    out_lines = []
    changed = False
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # 若为6列：cls track_id x y w h -> 去掉track_id
            if len(parts) >= 6:
                cls = parts[0]
                coords = parts[2:6]
                changed = True
            else:
                cls = parts[0]
                coords = parts[1:5]

            try:
                nums = [float(x) for x in coords]
            except Exception:
                continue

            # 若坐标大于1且有图像尺寸，视为像素坐标，做归一化
            if h_img and w_img and any(v > 1.0 for v in nums):
                xc, yc, w, h = nums
                nums = [xc / w_img, yc / h_img, w / w_img, h / h_img]
                changed = True

            out_lines.append(f"{int(float(cls))} " + " ".join(f"{v:.6f}" for v in nums))

    if not out_lines:
        print(f"[WARN] empty after fixing {lbl_path}")
        return False

    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="数据集根目录（包含 images/train 与 labels/train）")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="要处理的数据集划分，逗号分隔，例如 train,val,test（默认全部）",
    )
    args = parser.parse_args()

    root = args.root
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    total_files = 0
    total_changed = 0

    for split in splits:
        img_dir = os.path.join(root, "images", split)
        label_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(label_dir):
            continue
        files = glob.glob(os.path.join(label_dir, "*.txt"))
        if not files:
            continue
        changed_cnt = 0
        for p in files:
            if fix_one(p, img_dir):
                changed_cnt += 1
        print(f"[DONE] split={split} fixed {changed_cnt}/{len(files)} label files")
        total_files += len(files)
        total_changed += changed_cnt

    if total_files == 0:
        print("[WARN] 未发现任何可处理的标签文件（labels/<split>/*.txt）")
    else:
        print(f"[DONE] all_splits fixed {total_changed}/{total_files} label files")


if __name__ == "__main__":
    main()



