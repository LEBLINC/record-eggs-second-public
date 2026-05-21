import os
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='封装Ultralytics YOLOv8训练/验证（Python API 版本）')
    parser.add_argument('--data', required=True, help='data.yaml 路径')
    parser.add_argument('--imgsz', type=int, default=896)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--model', default='yolov8s.pt', help='yolov8s.pt 或 yolov8s-p2.pt')
    parser.add_argument('--project', default='runs/detect', help='Ultralytics输出目录')
    parser.add_argument('--name', default='eggs_train', help='实验名')
    args = parser.parse_args()

    try:
        # Windows 下 conda 环境可能被「用户级 site-packages（AppData/Roaming）」污染，
        # 导致导入到与当前 torch 版本不兼容的 ultralytics；也可能出现你这种“装了但找不到”的情况（装在另一个 Python 里）。
        # 这里在导入 ultralytics 前临时移除 usersite，确保优先使用当前解释器环境内的包。
        try:
            import site  # noqa

            usersite = site.getusersitepackages()
            _removed_usersite = False
            if isinstance(usersite, str) and usersite in sys.path:
                sys.path.remove(usersite)
                _removed_usersite = True
        except Exception:
            usersite = None
            _removed_usersite = False

        # 若 ultralytics 已从 usersite 被加载过，清理已加载模块避免继续复用错误版本
        try:
            if "ultralytics" in sys.modules:
                m = sys.modules.get("ultralytics")
                m_path = getattr(m, "__file__", "") or ""
                if ("AppData\\Roaming\\Python" in m_path) or ("AppData/Roaming/Python" in m_path):
                    for k in list(sys.modules.keys()):
                        if k == "ultralytics" or k.startswith("ultralytics."):
                            del sys.modules[k]
        except Exception:
            pass

        from ultralytics import YOLO

        # 恢复 usersite（追加到末尾，避免其覆盖当前环境内包）
        try:
            if _removed_usersite and isinstance(usersite, str) and usersite and usersite not in sys.path:
                sys.path.append(usersite)
        except Exception:
            pass
    except Exception as e:
        print('[ERROR] 未找到 ultralytics 包（或被用户级 site-packages 污染导致导入失败）')
        print('[ERROR] 当前解释器:', sys.executable)
        print('[ERROR] 解决方法（推荐在 conda 环境内执行）：')
        print('  1) python -m pip install --upgrade --no-user ultralytics==8.2.18')
        print('  2) 用 python 运行训练脚本：python tools/train_yolov8.py ... （不要用 py）')
        raise

    # 训练
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        cos_lr=True,
        amp=True,
    )

    # 验证（低conf以便绘制完整PR曲线）
    best_path = os.path.join(args.project, args.name, 'weights', 'best.pt')
    eval_model = YOLO(best_path) if os.path.isfile(best_path) else model
    eval_model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=0.001,
        iou=0.6,
        project=args.project,
        name=f'{args.name}_val',
    )

    print('[DONE] 训练与验证流程结束。产物位于:', os.path.join(args.project, args.name))


if __name__ == '__main__':
    main()




