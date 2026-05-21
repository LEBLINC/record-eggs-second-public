# coding=utf-8
"""
RTMDet 图像预处理模块
负责 letterbox 缩放、BGR→RGB 转换、归一化和张量格式转换
@project: EGGRECORDQT
@Author：lzy
@file： preprocessor.py
"""
import cv2
import numpy as np
from typing import Tuple


class RTMDetPreprocessor:
    """
    RTMDet 模型图像预处理器

    处理流程：
      1. Letterbox 缩放（保持宽高比，灰边填充）
      2. BGR → RGB 通道转换
      3. 归一化（减均值除标准差）
      4. HWC → CHW → NCHW 张量转换（float32）

    归一化参数来自 RTMDet 官方配置：
      mean = [103.53, 116.28, 123.675]  (BGR 顺序，对应 ImageNet BGR mean)
      std  = [57.375, 57.12,  58.395]   (BGR 顺序)
    注意：归一化在 RGB 转换之后按 RGB 顺序应用，因此内部存储为 RGB 顺序。
    """

    # RTMDet 官方归一化参数（RGB 顺序：R, G, B）
    # 原始 BGR 均值 [103.53, 116.28, 123.675] 对应 RGB [123.675, 116.28, 103.53]
    MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD_RGB  = np.array([58.395,  57.12,  57.375], dtype=np.float32)

    def __init__(self, target_size: int = 640):
        """
        初始化预处理器

        Args:
            target_size: 目标推理分辨率（正方形），默认 640
        """
        self.target_size = target_size

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        对输入图像执行完整预处理流程，返回模型输入张量和逆变换参数。

        Args:
            image: 原始 BGR 图像，shape (H, W, 3)，dtype uint8

        Returns:
            blob:        float32 ndarray，shape (1, 3, target_size, target_size)
            meta:        逆变换所需的元信息字典，包含：
                           - 'orig_shape':  (orig_h, orig_w)
                           - 'pad_top':     顶部填充像素数
                           - 'pad_left':    左侧填充像素数
                           - 'scale':       缩放比例（resize 时使用的统一比例）
        """
        orig_h, orig_w = image.shape[:2]

        # 1. Letterbox 缩放
        resized, pad_top, pad_left, scale = self._letterbox(image)

        # 2. BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. 归一化
        normalized = self._normalize(rgb)

        # 4. HWC → CHW → NCHW
        blob = normalized.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
        blob = np.ascontiguousarray(blob, dtype=np.float32)

        meta = {
            'orig_shape': (orig_h, orig_w),
            'pad_top':    pad_top,
            'pad_left':   pad_left,
            'scale':      scale,
        }
        return blob, meta

    def inverse_transform_boxes(self, boxes: np.ndarray, meta: dict) -> np.ndarray:
        """
        将模型输出的检测框坐标（在 letterbox 图像空间中）映射回原始图像坐标。

        Args:
            boxes: 检测框数组，shape (N, 4+)，前四列为 [x1, y1, x2, y2]（letterbox 空间）
            meta:  preprocess() 返回的元信息字典

        Returns:
            原始图像坐标系下的检测框，shape 与输入相同
        """
        if boxes is None or len(boxes) == 0:
            return boxes

        boxes = boxes.copy().astype(np.float32)
        pad_top  = meta['pad_top']
        pad_left = meta['pad_left']
        scale    = meta['scale']
        orig_h, orig_w = meta['orig_shape']

        # 去除 padding 偏移，再除以缩放比例
        boxes[:, 0] = (boxes[:, 0] - pad_left) / scale  # x1
        boxes[:, 1] = (boxes[:, 1] - pad_top)  / scale  # y1
        boxes[:, 2] = (boxes[:, 2] - pad_left) / scale  # x2
        boxes[:, 3] = (boxes[:, 3] - pad_top)  / scale  # y2

        # 裁剪到原始图像边界
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        return boxes

    def inverse_transform_points(self, points: np.ndarray, meta: dict) -> np.ndarray:
        """
        将模型输出的点坐标（在 letterbox 图像空间中）映射回原始图像坐标。

        Args:
            points: 点坐标数组，shape (N, 2)，列为 [x, y]（letterbox 空间）
            meta:   preprocess() 返回的元信息字典

        Returns:
            原始图像坐标系下的点坐标，shape (N, 2)
        """
        if points is None or len(points) == 0:
            return points

        points = points.copy().astype(np.float32)
        pad_top  = meta['pad_top']
        pad_left = meta['pad_left']
        scale    = meta['scale']
        orig_h, orig_w = meta['orig_shape']

        points[:, 0] = (points[:, 0] - pad_left) / scale  # x
        points[:, 1] = (points[:, 1] - pad_top)  / scale  # y

        points[:, 0] = np.clip(points[:, 0], 0, orig_w)
        points[:, 1] = np.clip(points[:, 1], 0, orig_h)

        return points

    def inverse_transform_rotated_boxes(self, rotated_boxes: np.ndarray, meta: dict) -> np.ndarray:
        """
        将旋转框参数（cx, cy, w, h, angle）从 letterbox 空间映射回原始图像坐标。

        Args:
            rotated_boxes: shape (N, 5+)，前五列为 [cx, cy, w, h, angle]
            meta:          preprocess() 返回的元信息字典

        Returns:
            原始图像坐标系下的旋转框，shape 与输入相同
        """
        if rotated_boxes is None or len(rotated_boxes) == 0:
            return rotated_boxes

        rotated_boxes = rotated_boxes.copy().astype(np.float32)
        pad_top  = meta['pad_top']
        pad_left = meta['pad_left']
        scale    = meta['scale']

        # 中心点去 padding 并反缩放
        rotated_boxes[:, 0] = (rotated_boxes[:, 0] - pad_left) / scale  # cx
        rotated_boxes[:, 1] = (rotated_boxes[:, 1] - pad_top)  / scale  # cy
        # 宽高只需反缩放，不涉及 padding
        rotated_boxes[:, 2] = rotated_boxes[:, 2] / scale  # w
        rotated_boxes[:, 3] = rotated_boxes[:, 3] / scale  # h
        # angle 不变

        return rotated_boxes

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, int, int, float]:
        """
        Letterbox 缩放：保持宽高比，将图像缩放至 target_size × target_size，
        不足部分用灰色（114, 114, 114）填充。

        Args:
            image: BGR 图像，shape (H, W, 3)

        Returns:
            out:      缩放并填充后的图像，shape (target_size, target_size, 3)
            pad_top:  顶部填充像素数
            pad_left: 左侧填充像素数
            scale:    缩放比例
        """
        h, w = image.shape[:2]
        target = self.target_size

        # 计算保持宽高比的缩放比例
        scale = min(target / h, target / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 计算填充量（居中放置）
        pad_h = target - new_h
        pad_w = target - new_w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        # 用灰色填充
        out = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        return out, pad_top, pad_left, scale

    def _normalize(self, rgb: np.ndarray) -> np.ndarray:
        """
        对 RGB 图像执行归一化：(pixel - mean) / std

        Args:
            rgb: RGB 图像，shape (H, W, 3)，dtype uint8

        Returns:
            归一化后的 float32 数组，shape (H, W, 3)
        """
        img = rgb.astype(np.float32)
        img = (img - self.MEAN_RGB) / self.STD_RGB
        return img
