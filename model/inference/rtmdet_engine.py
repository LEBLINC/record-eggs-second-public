# coding=utf-8
"""
RTMDet 统一推理引擎
整合 RTMDet-OBB-P2（二维码旋转框检测）和 RTMDet-Ins-P2（种蛋实例分割）双分支推理
提供与 YOLOTrack 兼容的 track() 接口
@project: EGGRECORDQT
@Author：lzy
@file： rtmdet_engine.py
"""
import os
import numpy as np
from typing import List, Dict, Optional

from model.inference.rtmdet_obb_inference import RTMDetOBBInference
from model.inference.rtmdet_ins_inference import RTMDetInsInference
from model.utils.exception import exception_handler


class RTMDetInferenceEngine:
    """
    RTMDet 双分支推理引擎，替代 YOLOTrack。

    整合两个检测分支：
      - RTMDet-OBB-P2：二维码旋转框检测（含有效性分类）
      - RTMDet-Ins-P2：种蛋实例分割（含中心点提取）

    提供与 YOLOTrack 兼容的 track() 和 batch_track() 接口。

    Config dict 支持的键（两套命名都接受，优先简短版与 yaml 一致）：
      - obb_model / rtmdet_obb_model:  OBB ONNX 模型文件路径（必填）
      - ins_model / rtmdet_ins_model:  Ins ONNX 模型文件路径（必填）
      - imgsz:               推理分辨率，正方形（默认 640）
      - conf:                置信度阈值（默认 0.5）
      - iou:                 IoU 阈值（默认 0.5）
      - validity_threshold:  QR 有效性阈值（默认 0.7）
      - use_fp16:            是否使用 FP16 推理（默认 False）。
                             仅当 ONNX 模型在导出时就以 FP16 输入构图时才能启用，
                             否则 ONNX Runtime 会因输入类型不匹配而报错。
    """

    def __init__(self, cfg: dict):
        """
        初始化推理引擎，加载 OBB 和 Ins 两个模型并执行预热。

        Args:
            cfg: 配置字典，支持的键见类文档

        Raises:
            FileNotFoundError: 当 OBB 或 Ins 模型文件不存在时
            KeyError:          当必填配置键缺失时
            ImportError:       当 onnxruntime-gpu 未安装时
        """
        # 读取配置，缺失时使用默认值并打印警告
        self.imgsz              = cfg.get('imgsz', 640)
        self.conf               = cfg.get('conf', 0.5)
        self.iou                = cfg.get('iou', 0.5)
        self.validity_threshold = cfg.get('validity_threshold', 0.7)
        self.use_fp16           = cfg.get('use_fp16', False)

        # 检查必填模型路径（同时支持 'obb_model'/'ins_model' 与
        # 旧版 'rtmdet_obb_model'/'rtmdet_ins_model'，与 configs/config.yaml
        # 的 rtmdet 子节命名保持一致）
        obb_model_path = cfg.get('obb_model') or cfg.get('rtmdet_obb_model')
        ins_model_path = cfg.get('ins_model') or cfg.get('rtmdet_ins_model')

        if not obb_model_path:
            raise KeyError(
                "配置缺少必填键 'obb_model'（或旧名 'rtmdet_obb_model'），"
                "请在 rtmdet 配置节中指定 OBB 模型路径。"
            )
        if not ins_model_path:
            raise KeyError(
                "配置缺少必填键 'ins_model'（或旧名 'rtmdet_ins_model'），"
                "请在 rtmdet 配置节中指定 Ins 模型路径。"
            )

        # 提前检查文件是否存在，给出清晰的错误信息
        if not os.path.isfile(obb_model_path):
            raise FileNotFoundError(
                f"RTMDet-OBB 模型文件未找到: {obb_model_path}\n"
                "请将训练导出的 ONNX 文件放置到指定路径后重试。"
            )
        if not os.path.isfile(ins_model_path):
            raise FileNotFoundError(
                f"RTMDet-Ins 模型文件未找到: {ins_model_path}\n"
                "请将训练导出的 ONNX 文件放置到指定路径后重试。"
            )

        # 打印缺失可选配置的警告
        _optional_defaults = {
            'imgsz': 640,
            'conf': 0.5,
            'iou': 0.5,
            'validity_threshold': 0.7,
            'use_fp16': False,
        }
        for key, default in _optional_defaults.items():
            if key not in cfg:
                print(f"RTMDetInferenceEngine: 配置缺少 '{key}'，使用默认值 {default}")

        print("RTMDetInferenceEngine: 正在加载 OBB 模型...")
        self.obb_model = RTMDetOBBInference(
            model_path=obb_model_path,
            conf_threshold=self.conf,
            iou_threshold=self.iou,
            target_size=self.imgsz,
            use_fp16=self.use_fp16,
        )

        print("RTMDetInferenceEngine: 正在加载 Ins 模型...")
        self.ins_model = RTMDetInsInference(
            model_path=ins_model_path,
            conf_threshold=self.conf,
            target_size=self.imgsz,
            use_fp16=self.use_fp16,
        )

        # 执行模型预热
        self._warmup()

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------

    @exception_handler
    def _warmup(self, num_iters: int = 2):
        """
        对两个模型分别执行预热推理，减少首次推理延迟。

        Args:
            num_iters: 预热迭代次数（默认 2）
        """
        print(f"RTMDetInferenceEngine: 开始预热（{num_iters} 次）...")
        self.obb_model.warmup(num_iters=num_iters)
        self.ins_model.warmup(num_iters=num_iters)
        print("RTMDetInferenceEngine: 预热完成")

    # ------------------------------------------------------------------
    # 主推理接口
    # ------------------------------------------------------------------

    @exception_handler
    def track(self, frame: np.ndarray) -> Dict:
        """
        对单帧图像执行双分支推理，返回统一检测结果。

        与 YOLOTrack.track() 接口兼容。

        Args:
            frame: BGR 图像，shape (H, W, 3)，dtype uint8

        Returns:
            dict，包含以下键：
              - qr_detections:  List[dict]，每个元素来自 RTMDetOBBInference，包含：
                                  rotated_box, hbb, score, class_id, validity_score
              - egg_detections: List[dict]，每个元素来自 RTMDetInsInference，包含：
                                  bbox, mask, center, ellipse_params, score
        """
        if frame is None or frame.size == 0:
            print("RTMDetInferenceEngine: 收到空帧，跳过推理")
            return {'qr_detections': [], 'egg_detections': []}

        # 顺序执行两个分支（简单可靠，避免并行带来的复杂性）
        qr_detections  = self._detect_qr(frame)
        egg_detections = self._detect_egg(frame)

        return {
            'qr_detections':  qr_detections,
            'egg_detections': egg_detections,
        }

    @exception_handler
    def batch_track(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        批量处理多帧图像，镜像 YOLOTrack.batch_track() 接口。

        对每帧独立调用 track()，顺序处理。

        Args:
            frames: 包含多个 BGR 图像的列表

        Returns:
            包含每帧检测结果的列表，每个元素格式与 track() 返回值相同
        """
        if not frames:
            return []

        results = []
        for frame in frames:
            result = self.track(frame)
            # exception_handler 在异常时返回 None，此处做保护
            if result is None:
                result = {'qr_detections': [], 'egg_detections': []}
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # 单分支推理（供外部直接调用）
    # ------------------------------------------------------------------

    @exception_handler
    def detect_qr(self, frame: np.ndarray) -> List[Dict]:
        """
        仅运行 QR 码检测分支。

        Args:
            frame: BGR 图像，shape (H, W, 3)，dtype uint8

        Returns:
            QR 检测结果列表（格式同 RTMDetOBBInference.infer()）
        """
        return self._detect_qr(frame)

    @exception_handler
    def detect_egg(self, frame: np.ndarray) -> List[Dict]:
        """
        仅运行种蛋检测分支。

        Args:
            frame: BGR 图像，shape (H, W, 3)，dtype uint8

        Returns:
            种蛋检测结果列表（格式同 RTMDetInsInference.infer()）
        """
        return self._detect_egg(frame)

    # ------------------------------------------------------------------
    # 内部推理方法（不带 exception_handler，由调用方统一处理）
    # ------------------------------------------------------------------

    def _detect_qr(self, frame: np.ndarray) -> List[Dict]:
        """执行 OBB 分支推理，返回 QR 检测结果列表。"""
        detections = self.obb_model.infer(frame)
        if detections is None:
            return []
        return detections

    def _detect_egg(self, frame: np.ndarray) -> List[Dict]:
        """执行 Ins 分支推理，返回种蛋检测结果列表。"""
        detections = self.ins_model.infer(frame)
        if detections is None:
            return []
        return detections

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    def release(self):
        """
        显式释放两个模型的推理会话资源。

        在不再需要引擎时调用，以释放 GPU 显存。
        """
        if self.obb_model is not None:
            self.obb_model.session = None
            self.obb_model = None
        if self.ins_model is not None:
            self.ins_model.session = None
            self.ins_model = None
        print("RTMDetInferenceEngine: 资源已释放")

    def __del__(self):
        """析构时释放推理会话资源。"""
        try:
            if hasattr(self, 'obb_model') and self.obb_model is not None:
                self.obb_model.session = None
            if hasattr(self, 'ins_model') and self.ins_model is not None:
                self.ins_model.session = None
        except Exception:
            pass
