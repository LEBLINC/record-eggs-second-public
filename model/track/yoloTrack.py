# coding=utf-8
"""
多摄像头优化版YOLO跟踪模块
@project: EGGRECORDQT
@Author：lzy
@file： YOLOTrack.py
@date：2023/6/15 15:02
"""

# Windows 下 conda 环境可能会被「用户级 site-packages（AppData/Roaming）」污染，
# 导致导入到与当前 torch 版本不兼容的 ultralytics，从而出现 fuse()/bn 等异常。
# 这里在导入 ultralytics 前临时移除 usersite 路径，并在必要时清理已加载的 ultralytics 模块。
# 注意：很多用户会把依赖装到 usersite（pip 默认 user 安装），因此我们会在导入完成后把 usersite 追加回 sys.path，
# 以便其它依赖（如 pynvml/nvidia-ml-py）仍可正常导入，同时保证 ultralytics 优先使用 conda 环境内版本。
import sys

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

# 恢复 usersite（追加到末尾，避免其覆盖 conda 环境内包）
try:
    if _removed_usersite and isinstance(usersite, str) and usersite and usersite not in sys.path:
        sys.path.append(usersite)
except Exception:
    pass
from model.track.tracker_zoo import on_predict_start
from functools import partial
import torch
import cv2
import numpy as np
import threading
from model.utils.exception import exception_handler
from model.utils.preprocess import preprocess_bgr


class YOLOTrack:
    def __init__(self, cfg):
        """
        初始化YOLO跟踪模型
        Args:
            cfg: 配置字典
        """
        # 加载模型
        # 兼容 torch>=2.6（2.6+ 将 torch.load 的默认 weights_only 改为 True）导致的自定义 pt 权重加载失败问题：
        # - 该项目的 resources/best.pt 属于“包含 Ultralytics DetectionModel 的 checkpoint”
        # - torch 的安全反序列化需要 allowlist 对应类，否则会抛 Weights only load failed
        # 这里在可信本地权重场景下，使用 torch.serialization.safe_globals/add_safe_globals 做兼容，
        # 以便在不同 torch 版本下都能正常启动（尤其是 Windows 上用户容易升级 torch）。
        self.model = self._load_yolo_model(cfg['modelPath'])

        # 添加跟踪回调
        self.model.add_callback('on_predict_start',
                                partial(on_predict_start, persist=True, tracking_config=cfg['tracking_config']))

        # 从配置中获取参数
        self.imgsz = cfg.get('imgsz', 640)
        self.conf = cfg.get('conf', 0.5)
        self.iou = cfg.get('iou', 0.5)
        self.preprocess_cfg = cfg.get('preprocess', {}) if isinstance(cfg, dict) else {}

        # 检查并设置GPU设备
        self.device = self._get_device()

        # 防止预热与真实推理并发调用 model.track 导致不稳定
        self._infer_lock = threading.Lock()

        # 运动模糊增强：强制开启半精度（如果设备支持），并降低置信度门槛
        # 半精度能显著提速，减少推理延迟，从而减轻“帧与帧之间位移过大”的问题
        self.half = False  # 先默认关闭半精度
        if self.device != 'cpu':
            # 检查GPU是否支持半精度
            if torch.cuda.get_device_capability()[0] >= 7:  # compute capability >= 7.0
                # 强制尝试启用半精度，除非配置显式禁用
                force_half = True 
                self.half = cfg.get('gpu_optimization', {}).get('use_fp16', force_half)
                if self.half:
                    print("启用半精度推理")
                    try:
                        self.model.model.half()  # 转换模型到半精度
                    except Exception as e:
                        print(f"半精度转换失败: {e}，使用全精度")
                        self.half = False

        # 预热模型（默认异步，避免阻塞启动）
        self._warmup_done = False
        self._warmup_async = cfg.get('warmup_async', True)
        if self._warmup_async:
            threading.Thread(target=self._warmup, name="yolo-warmup", daemon=True).start()
        else:
            self._warmup()

    def _load_yolo_model(self, model_path: str):
        """
        加载 Ultralytics YOLO 权重，并兼容 torch>=2.6 的安全反序列化机制。

        说明：
        - torch 2.6+ 默认 torch.load(weights_only=True)，会拒绝反序列化未 allowlist 的全局类；
        - Ultralytics 的 .pt 有时包含 DetectionModel 等对象引用；
        - 通过 torch.serialization.safe_globals/add_safe_globals 允许这些类，避免启动时报错。
        """
        ser = getattr(torch, "serialization", None)
        # 旧 torch：没有 safe_globals 机制，直接加载
        if ser is None or not hasattr(ser, "safe_globals"):
            return YOLO(model_path, task='detect')

        # torch>=2.6：迭代 allowlist（直到能成功加载）
        # 说明：best.pt 可能包含完整模型对象，除了 Ultralytics DetectionModel 之外还会引用 torch.nn.*（Sequential 等）
        import re
        import importlib

        safe_set = set()
        # 先放入本项目最常见的类（可显著减少迭代次数）
        try:
            from ultralytics.nn.tasks import BaseModel, DetectionModel  # type: ignore

            safe_set.update([BaseModel, DetectionModel])
        except Exception:
            pass
        try:
            from torch.nn.modules.container import Sequential  # type: ignore

            safe_set.add(Sequential)
        except Exception:
            pass

        last_exc = None
        pattern = re.compile(r"Unsupported global: GLOBAL ([A-Za-z0-9_\\.]+)")

        for _ in range(64):
            try:
                with ser.safe_globals(list(safe_set)):
                    return YOLO(model_path, task='detect')
            except Exception as e:
                last_exc = e
                msg = str(e) or ""
                m = pattern.search(msg)
                if not m:
                    break
                dotted = m.group(1)
                # 解析并动态 import，加入 allowlist 后重试
                try:
                    mod_name, attr = dotted.rsplit(".", 1)
                    mod = importlib.import_module(mod_name)
                    obj = getattr(mod, attr)
                    safe_set.add(obj)
                    continue
                except Exception:
                    break

        # 如果 safe_globals 多次迭代仍失败，再尝试一次 add_safe_globals（部分 torch/环境可能需要全局注册）
        try:
            if hasattr(ser, "add_safe_globals") and safe_set:
                ser.add_safe_globals(list(safe_set))
                return YOLO(model_path, task='detect')
        except Exception as e:
            last_exc = last_exc or e

        # 最终兜底：抛出最后一次异常，便于用户定位缺失的 safe globals
        if last_exc is not None:
            raise last_exc
        return YOLO(model_path, task='detect')

    @exception_handler
    def _get_device(self):
        """确定最佳设备"""
        try:
            if torch.cuda.is_available():
                # 如果有CUDA设备，使用第一个设备
                device = f"cuda:0"
                # 设置CUDA工作流
                torch.backends.cudnn.benchmark = True
                # 启用TF32（RTX 4060Ti支持）
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # 打印设备信息
                device_name = torch.cuda.get_device_name(0)
                print(f"使用GPU设备: {device_name}")
                # 获取GPU内存信息
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                mem_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
                mem_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                print(f"GPU内存: 总计 {mem_total:.2f}GB, 已分配 {mem_allocated:.2f}GB, 已保留 {mem_reserved:.2f}GB")
            else:
                # 否则使用CPU
                device = "cpu"
                print("CUDA不可用，使用CPU。如果你有NVIDIA显卡，请确保已安装CUDA和GPU版本的PyTorch。")
        except Exception as e:
            print(f"获取设备信息时出错: {e}")
            device = "cpu"
            print("出现异常，将使用CPU")

        return device

    @exception_handler
    def _warmup(self):
        """预热模型，以减少首次推理时间"""
        print("预热YOLO模型...")
        try:
            # 创建一个小尺寸的随机图像
            dummy_input = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

            # 直接使用track方法进行预热，避免类型不匹配
            with self._infer_lock:
                _ = self.model.track(
                    dummy_input,
                    persist=True,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                    imgsz=320,
                    device=self.device,
                    half=self.half  # 使用配置的半精度设置
                )

            # 强制清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("YOLO模型预热完成")
            self._warmup_done = True
        except Exception as e:
            print(f"模型预热异常: {e}")
            # 如果半精度预热失败，尝试关闭半精度
            if self.half:
                print("尝试关闭半精度重新预热...")
                self.half = False
                try:
                    with self._infer_lock:
                        _ = self.model.track(
                            dummy_input,
                            persist=True,
                            conf=0.25,
                            iou=0.45,
                            verbose=False,
                            imgsz=320,
                            device=self.device,
                            half=False
                        )
                    print("全精度模式预热成功")
                except Exception as e2:
                    print(f"全精度预热也失败: {e2}")

    @exception_handler
    def track(self, frame):
        """
        使用YOLO模型进行目标跟踪
        Args:
            frame: 输入帧
        Returns:
            跟踪结果
        """
        inp = preprocess_bgr(frame, self.preprocess_cfg)
        with self._infer_lock:
            results = self.model.track(
                inp,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                show=False,
                save_txt=False,
                show_labels=False,
                verbose=False,
                exist_ok=False,
                imgsz=self.imgsz,
                vid_stride=1,
                line_width=None,
                device=self.device,
                half=self.half  # 使用配置的半精度设置
            )
        return results

    @exception_handler
    def batch_track(self, frames):
        """
        批量处理多个帧
        Args:
            frames: 包含多个帧的列表
        Returns:
            包含每个帧跟踪结果的列表
        """
        if not frames:
            return []

        # 批量推理（可选预处理，保持尺寸不变，坐标仍可用于原图）
        frames_inp = [preprocess_bgr(f, self.preprocess_cfg) for f in frames]
        with self._infer_lock:
            results = self.model.track(
                frames_inp,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                show=False,
                save_txt=False,
                show_labels=False,
                verbose=False,
                exist_ok=False,
                imgsz=self.imgsz,
                vid_stride=1,
                line_width=None,
                device=self.device,
                batch=len(frames),
                half=self.half  # 使用配置的半精度设置
            )

        return results

    def __del__(self):
        """清理资源"""
        self.model = None
        # 清除CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
