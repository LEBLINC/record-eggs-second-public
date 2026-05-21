import os
import argparse
import glob
import shutil
import random


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_class_map(s: str) -> dict[int, int]:
    """
    Parse class remap string, e.g.:
      "0:1,1:0"  -> {0: 1, 1: 0}
      "0->1;1->0" -> {0: 1, 1: 0}
    """
    s = (s or "").strip()
    if not s:
        return {}
    # normalize separators
    for sep in [";", "|", "\n"]:
        s = s.replace(sep, ",")
    s = s.replace("->", ":").replace("=", ":")
    out: dict[int, int] = {}
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if ":" not in part:
            raise ValueError(f"Invalid class-map item: {part!r}, expected like '0:1'")
        a, b = [x.strip() for x in part.split(":", 1)]
        out[int(float(a))] = int(float(b))
    return out


def remap_label_file(src_path: str, dst_path: str, class_map: dict[int, int]) -> None:
    """Copy a YOLO label file and remap the first column (class id) if present."""
    if not class_map:
        shutil.copy2(src_path, dst_path)
        return
    out_lines: list[str] = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                c = int(float(parts[0]))
            except Exception:
                # keep raw line if cannot parse
                out_lines.append(line)
                continue
            parts[0] = str(class_map.get(c, c))
            out_lines.append(" ".join(parts))
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))


def collect_label_paths(labels_root: str, split: str) -> list[str]:
    base = os.path.join(labels_root, split)
    if not os.path.isdir(base):
        return []
    return [p for p in glob.glob(os.path.join(base, '**', '*.txt'), recursive=True)]


def copy_matched_images(label_paths: list[str], images_raw_root: str, out_images_split: str, img_exts=('.jpg', '.jpeg', '.png')) -> int:
    copied = 0
    for lp in label_paths:
        stem = os.path.splitext(os.path.basename(lp))[0]
        found = None
        for ext in img_exts:
            candidates = list(glob.iglob(os.path.join(images_raw_root, '**', stem + ext), recursive=True))
            if candidates:
                found = candidates[0]
                break
        if not found:
            print(f"[WARN] 未找到图片对应: {stem}，跳过")
            continue
        shutil.copy2(found, os.path.join(out_images_split, os.path.basename(found)))
        copied += 1
    return copied


def write_data_yaml(dataset_root: str):
    yaml_path = os.path.join(dataset_root, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(
            f"path: {dataset_root}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"names:\n  0: egg\n  1: qr\n"
        )
    print(f"[INFO] 生成: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='对齐CVAT导出labels与本地抽帧images，打包为YOLO数据集')
    parser.add_argument('--labels-root', required=True, help='CVAT导出labels根，例如 eggs_ultralytics_yolo_20251021/labels')
    parser.add_argument('--images-raw', required=True, help='本地抽帧根，例如 data/dataset_eggs/images_raw')
    parser.add_argument('--out-root', required=True, help='输出YOLO数据集根，例如 data/dataset_eggs')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='处理的集合')
    parser.add_argument('--val-from-train', type=int, default=0, help='当val为空或不存在时，从train抽样N张构造最小验证集')
    parser.add_argument(
        '--class-map',
        default='',
        help='可选：重映射标签类别ID，例如 "0:1,1:0"（当CVAT类别顺序与项目训练/上线不一致时很有用）'
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.out_root)
    labels_root = os.path.abspath(args.labels_root)
    images_raw = os.path.abspath(args.images_raw)
    class_map = parse_class_map(args.class_map)

    for split in args.splits:
        out_images_split = os.path.join(dataset_root, 'images', split)
        out_labels_split = os.path.join(dataset_root, 'labels', split)
        ensure_dir(out_images_split)
        ensure_dir(out_labels_split)

        label_paths = collect_label_paths(labels_root, split)
        if not label_paths:
            print(f"[INFO] {split} 无标签，跳过")
            continue

        # 复制标签（可选重映射类别ID），保持原文件名
        for lp in label_paths:
            dst = os.path.join(out_labels_split, os.path.basename(lp))
            remap_label_file(lp, dst, class_map)

        # 复制匹配的图片
        copied = copy_matched_images(label_paths, images_raw, out_images_split)
        print(f"[INFO] {split}: 拷贝图片 {copied} 张，标签 {len(label_paths)} 个")

    # 若 val 为空且需要从 train 构造最小验证集
    val_labels = os.path.join(dataset_root, 'labels', 'val')
    val_images = os.path.join(dataset_root, 'images', 'val')
    train_labels = os.path.join(dataset_root, 'labels', 'train')
    train_images = os.path.join(dataset_root, 'images', 'train')

    ensure_dir(val_labels)
    ensure_dir(val_images)

    need_make_val = args.val_from_train > 0 and len(glob.glob(os.path.join(val_labels, '*.txt'))) == 0
    if need_make_val:
        train_label_files = glob.glob(os.path.join(train_labels, '*.txt'))
        if len(train_label_files) == 0:
            print('[WARN] train 中无标签，无法构造 val')
        else:
            k = min(args.val_from_train, len(train_label_files))
            sample = random.sample(train_label_files, k)
            made = 0
            for lp in sample:
                stem = os.path.splitext(os.path.basename(lp))[0]
                # 从 train 移动到 val（避免训练/验证集泄漏）
                dst_lp = os.path.join(val_labels, os.path.basename(lp))
                try:
                    shutil.move(lp, dst_lp)
                except Exception:
                    shutil.copy2(lp, dst_lp)
                # 在 train images 中查找同名图片
                img_found = None
                for ext in ('.jpg', '.jpeg', '.png'):
                    p = os.path.join(train_images, stem + ext)
                    if os.path.isfile(p):
                        img_found = p
                        break
                # 若 train images 未找到（极端情况），从 images_raw 搜索
                if img_found is None:
                    hits = list(glob.iglob(os.path.join(images_raw, '**', stem + '.*'), recursive=True))
                    hits = [h for h in hits if os.path.splitext(h)[1].lower() in ('.jpg', '.jpeg', '.png')]
                    if hits:
                        img_found = hits[0]
                if img_found is not None:
                    dst_img = os.path.join(val_images, os.path.basename(img_found))
                    try:
                        shutil.move(img_found, dst_img)
                    except Exception:
                        shutil.copy2(img_found, dst_img)
                    made += 1
            print(f"[INFO] 由 train 抽样构造 val: {made}/{k} 张")

    write_data_yaml(dataset_root)
    print('[DONE] 对齐完成，可直接用于训练')


if __name__ == '__main__':
    main()


