import argparse
import os
import sys
import traceback


def main():
    # 确保从项目根目录可导入 `model.*`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="Smoke test: run YOLOTrack.batch_track on a few frames from a video.")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--frames", type=int, default=3, help="Number of frames to read")
    parser.add_argument("--cfg", default=os.path.join("configs", "config.yaml"), help="Config yaml path")
    parser.add_argument("--print-boxes", action="store_true", help="Print first frame tracked boxes details")
    args = parser.parse_args()

    try:
        import cv2
        import yaml

        from model.track.yoloTrack import YOLOTrack

        with open(args.cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        print("[INFO] cfg.modelPath =", cfg.get("modelPath"))
        print("[INFO] cfg.tracking_config =", cfg.get("tracking_config"))

        y = YOLOTrack(cfg)

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("[ERROR] cannot open video:", args.video)
            return 2

        frames = []
        for _ in range(int(args.frames)):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()

        print("[INFO] frames_read =", len(frames))
        if not frames:
            print("[ERROR] no frames read from video")
            return 3

        res = y.batch_track(frames)
        print("[INFO] batch_track_return_is_none =", res is None)
        if res is not None:
            print("[INFO] batch_results_len =", len(res))
            try:
                if len(res) > 0 and res[0] is not None and getattr(res[0], "boxes", None) is not None:
                    print("[INFO] first_boxes_shape =", res[0].boxes.data.shape)
                    if args.print_boxes:
                        b = res[0].boxes
                        names = getattr(res[0], "names", None) or getattr(y.model, "names", {})
                        d = b.data
                        try:
                            import torch
                            if isinstance(d, torch.Tensor):
                                d = d.cpu()
                        except Exception:
                            pass
                        print("[INFO] first_boxes_data =", d)
                        try:
                            cls = b.cls
                            conf = b.conf
                            tid = b.id
                            print("[INFO] first_boxes_cls =", cls)
                            print("[INFO] first_boxes_conf =", conf)
                            print("[INFO] first_boxes_id =", tid)
                            # 显示类别名
                            if names is not None:
                                try:
                                    import numpy as np
                                    cls_np = cls.cpu().numpy().astype(int) if hasattr(cls, "cpu") else np.array(cls).astype(int)
                                    print("[INFO] first_boxes_cls_name =", [names.get(int(c), str(int(c))) for c in cls_np])
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception:
                pass

        import ultralytics

        print("[INFO] ultralytics_version =", getattr(ultralytics, "__version__", "unknown"))
        print("[INFO] ultralytics_file =", getattr(ultralytics, "__file__", "unknown"))
        return 0
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


