import argparse
import os
import glob


def iter_images(source: str) -> list[str]:
    if os.path.isfile(source):
        return [source]
    patterns = []
    if os.path.isdir(source):
        patterns = [
            os.path.join(source, "**", "*.jpg"),
            os.path.join(source, "**", "*.jpeg"),
            os.path.join(source, "**", "*.png"),
        ]
    else:
        # 支持通配符输入
        patterns = [source]
    out = []
    for p in patterns:
        out.extend(glob.glob(p, recursive=True))
    # 去重保持顺序
    seen = set()
    uniq = []
    for p in out:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(ap)
    return uniq


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def xyxy_to_xywhn(x1, y1, x2, y2, w_img, h_img):
    xc = (x1 + x2) / 2.0 / w_img
    yc = (y1 + y2) / 2.0 / h_img
    w = (x2 - x1) / w_img
    h = (y2 - y1) / h_img
    return xc, yc, w, h


def main():
    parser = argparse.ArgumentParser(description="用现有 YOLOv8 权重对图片生成伪标签（YOLO txt），用于加速标注/微调训练")
    parser.add_argument("--model", required=True, help="权重路径，例如 runs/detect/eggs_train2/weights/best.pt")
    parser.add_argument("--source", required=True, help="图片目录或通配符或单张图片")
    parser.add_argument("--out-labels", required=True, help="输出 labels 目录（会生成 *.txt）")
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--conf-egg", type=float, default=0.15, help="写入 egg 标签的最低置信度")
    parser.add_argument("--conf-qr", type=float, default=0.25, help="写入 qr 标签的最低置信度")
    parser.add_argument("--write-empty", action="store_true", help="无目标时也写空 txt（默认不写）")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        print("[ERROR] 未找到 ultralytics，请先安装: pip install ultralytics")
        raise

    images = iter_images(args.source)
    if not images:
        print("[ERROR] 未找到任何图片:", args.source)
        return

    ensure_dir(args.out_labels)

    model = YOLO(args.model)
    min_conf = min(float(args.conf_egg), float(args.conf_qr))

    # stream=True 可避免一次性把所有结果存内存
    results = model.predict(
        source=images,
        imgsz=args.imgsz,
        conf=min_conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
        stream=True,
    )

    total = 0
    written = 0
    for r in results:
        total += 1
        img_path = getattr(r, "path", None)
        if not img_path:
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_txt = os.path.join(args.out_labels, base + ".txt")

        h_img, w_img = None, None
        try:
            if getattr(r, "orig_img", None) is not None:
                h_img, w_img = r.orig_img.shape[:2]
            elif getattr(r, "orig_shape", None) is not None:
                h_img, w_img = r.orig_shape[:2]
        except Exception:
            pass
        if not h_img or not w_img:
            # 没拿到尺寸就跳过（避免写错归一化）
            continue

        names = getattr(r, "names", None) or getattr(model, "names", {})
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            if args.write_empty:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write("")
                written += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().astype(float)

        out_lines = []
        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
            name = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
            if name == "egg":
                if cf < float(args.conf_egg):
                    continue
            elif name == "qr":
                if cf < float(args.conf_qr):
                    continue
            else:
                # 非关心类别直接跳过
                continue

            xc, yc, w, h = xyxy_to_xywhn(float(x1), float(y1), float(x2), float(y2), w_img, h_img)
            # clip 到 0~1，避免边界抖动写出非法值
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            out_lines.append(f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not out_lines:
            if args.write_empty:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write("")
                written += 1
            continue

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        written += 1

    print(f"[DONE] 处理图片 {total} 张，写出标签 {written} 份，输出目录: {os.path.abspath(args.out_labels)}")


if __name__ == "__main__":
    main()


