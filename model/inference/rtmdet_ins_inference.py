# coding=utf-8
"""
RTMDet-Ins-P2 鸡蛋实例分割推理模块
基于 ONNX Runtime 实现，支持 GPU/FP16 加速
@project: EGGRECORDQT
@Author：lzy
@file： rtmdet_ins_inference.py
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from model.inference.preprocessor import RTMDetPreprocessor


# ---------------------------------------------------------------------------
# 掩码工具函数（模块级，可独立使用）
# ---------------------------------------------------------------------------

def mask_to_center(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    通过图像矩（moments）计算二值掩码的质心坐标。

    Args:
        mask: 二值掩码，shape (H, W)，dtype uint8 或 bool

    Returns:
        (cx, cy) 质心坐标（float），若掩码为空则返回 None
    """
    mask_u8 = mask.astype(np.uint8)
    M = cv2.moments(mask_u8)
    if M['m00'] == 0:
        return None
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    return float(cx), float(cy)


def mask_to_ellipse(mask: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
    """
    对二值掩码的轮廓拟合椭圆，返回椭圆参数。

    Args:
        mask: 二值掩码，shape (H, W)，dtype uint8 或 bool

    Returns:
        (cx, cy, major_axis, minor_axis, angle) 椭圆参数，若拟合失败则返回 None
        - cx, cy:      椭圆中心坐标
        - major_axis:  长轴长度（像素）
        - minor_axis:  短轴长度（像素）
        - angle:       旋转角度（度，OpenCV 约定）
    """
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 取面积最大的轮廓
    largest = max(contours, key=cv2.contourArea)

    # cv2.fitEllipse 至少需要 5 个点
    if len(largest) < 5:
        return None

    try:
        (cx, cy), (minor_axis, major_axis), angle = cv2.fitEllipse(largest)
        # cv2.fitEllipse 返回 (width, height)，其中 width <= height
        # 确保 major_axis >= minor_axis
        if major_axis < minor_axis:
            major_axis, minor_axis = minor_axis, major_axis
        return float(cx), float(cy), float(major_axis), float(minor_axis), float(angle)
    except cv2.error:
        return None


def resize_mask_to_original(
    mask_crop: np.ndarray,
    bbox: Tuple[float, float, float, float],
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """
    将模型输出的掩码裁剪区域缩放并放置回原始图像尺寸。

    Args:
        mask_crop: 掩码裁剪区域，shape (mh, mw)，值为 0/1 或 float
        bbox:      对应的检测框 (x1, y1, x2, y2)，原始图像坐标
        orig_h:    原始图像高度
        orig_w:    原始图像宽度

    Returns:
        全尺寸二值掩码，shape (orig_h, orig_w)，dtype uint8
    """
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(orig_w, round(x2)))
    y2 = int(min(orig_h, round(y2)))

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # 缩放掩码到 bbox 尺寸
    mask_resized = cv2.resize(
        mask_crop.astype(np.float32),
        (box_w, box_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # 二值化
    mask_bin = (mask_resized > 0.5).astype(np.uint8)

    # 放置到全尺寸画布
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    full_mask[y1:y1 + box_h, x1:x1 + box_w] = mask_bin

    return full_mask


# ---------------------------------------------------------------------------
# 主推理类
# ---------------------------------------------------------------------------

class RTMDetInsInference:
    """
    RTMDet-Ins-P2 鸡蛋实例分割推理器

    类别定义（与训练 metainfo classes=('egg', 'invalidegg') 一致）：
      - class 0: egg         （正常蛋）
      - class 1: invalidegg  （破损/异常蛋）

    支持 MMDetection 导出的常见 ONNX 输出格式：
      - 格式 A（三输出）：
          dets:   (1, N, 5)  → [x1, y1, x2, y2, score]
          labels: (1, N)     → class_id
          masks:  (1, N, H, W) → 二值掩码（模型输出分辨率）
      - 格式 B（两输出）：
          dets:   (1, N, 5)  → [x1, y1, x2, y2, score]
          masks:  (1, N, H, W) → 二值掩码
      - 格式 C（单输出）：
          output: (1, N, 5+num_classes+mask_dim) → 合并格式
    """

    NUM_CLASSES = 2
    CLASS_EGG         = 0  # 正常蛋
    CLASS_INVALID_EGG = 1  # 破损/异常蛋

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        target_size: int      = 640,
        use_fp16: bool        = False,
    ):
        """
        初始化 RTMDet-Ins 推理器。

        Args:
            model_path:      ONNX 模型文件路径
            conf_threshold:  置信度过滤阈值（默认 0.5）
            target_size:     推理分辨率（正方形，默认 640）
            use_fp16:        是否使用 FP16 输入（需要 GPU 支持）
        """
        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.target_size    = target_size
        self.use_fp16       = use_fp16

        self.session     = None
        self.input_name  = None
        self.output_names: List[str] = []

        self.preprocessor = RTMDetPreprocessor(target_size=target_size)

        self._init_session()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_session(self):
        """初始化 ONNX Runtime 推理会话，优先使用 GPU（CUDAExecutionProvider）。"""
        import os
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"RTMDet-Ins 模型文件未找到: {self.model_path}\n"
                "请将训练导出的 ONNX 文件放置到指定路径后重试。"
            )

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime-gpu 未安装。请执行: pip install onnxruntime-gpu"
            )

        # 构建 Provider 列表：优先 CUDA，回退 CPU
        available_providers = ort.get_available_providers()
        providers = []

        if 'CUDAExecutionProvider' in available_providers:
            cuda_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4 GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            providers.append(('CUDAExecutionProvider', cuda_options))
            print("RTMDet-Ins: 使用 CUDAExecutionProvider")
        else:
            print("RTMDet-Ins: CUDA 不可用，回退到 CPUExecutionProvider")

        providers.append('CPUExecutionProvider')

        # 会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3  # 仅显示 ERROR

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
        )

        # 记录输入/输出名称
        self.input_name   = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 打印模型信息
        inp = self.session.get_inputs()[0]
        print(f"RTMDet-Ins 模型加载成功: {self.model_path}")
        print(f"  输入: {inp.name}, shape={inp.shape}, dtype={inp.type}")
        for out in self.session.get_outputs():
            print(f"  输出: {out.name}, shape={out.shape}, dtype={out.type}")

        active_providers = self.session.get_providers()
        print(f"  活跃 Provider: {active_providers}")

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def infer(self, image: np.ndarray) -> List[Dict]:
        """
        对单帧图像执行鸡蛋实例分割检测。

        Args:
            image: BGR 图像，shape (H, W, 3)，dtype uint8

        Returns:
            检测结果列表，每个元素为字典：
              {
                'bbox':          [x1, y1, x2, y2],                    # 原始图像坐标
                'mask':          np.ndarray (H, W) uint8,              # 原始图像尺寸二值掩码
                'center':        (cx, cy),                             # 质心坐标（原始图像）
                'ellipse_params': (cx, cy, major, minor, angle) | None, # 椭圆参数
                'score':         float,                                # 置信度分数
              }
        """
        if self.session is None:
            print("RTMDet-Ins: 推理会话未初始化，跳过推理")
            return []

        # 1. 预处理
        blob, meta = self.preprocessor.preprocess(image)

        # 2. FP16 转换（如需要）
        if self.use_fp16:
            blob = blob.astype(np.float16)

        # 3. ONNX 推理
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # 4. 后处理
        detections = self._postprocess(outputs, meta)

        return detections

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------

    def _postprocess(self, outputs: List[np.ndarray], meta: dict) -> List[Dict]:
        """
        解析 ONNX 输出，执行置信度过滤，并将掩码映射回原始图像坐标。

        支持三种输出格式（见类文档）。

        Args:
            outputs: ONNX Runtime 输出张量列表
            meta:    预处理元信息（用于坐标逆变换）

        Returns:
            过滤后的检测结果列表
        """
        orig_h, orig_w = meta['orig_shape']
        num_outputs = len(outputs)

        # ------------------------------------------------------------------
        # 解析各种输出格式
        # ------------------------------------------------------------------
        if num_outputs >= 3:
            # 格式 A：dets (1,N,5), labels (1,N), masks (1,N,H,W)
            dets   = outputs[0]  # (1, N, 5) or (N, 5)
            labels = outputs[1]  # (1, N) or (N,)
            masks  = outputs[2]  # (1, N, mH, mW) or (N, mH, mW)

            dets   = dets[0]   if dets.ndim == 3   else dets    # (N, 5)
            labels = labels[0] if labels.ndim == 2 else labels  # (N,)
            masks  = masks[0]  if masks.ndim == 4  else masks   # (N, mH, mW)

            if dets.shape[0] == 0:
                return []

            boxes_lbspace = dets[:, :4]   # (N, 4) [x1, y1, x2, y2]
            scores        = dets[:, 4]    # (N,)

        elif num_outputs == 2:
            # 格式 B：dets (1,N,5), masks (1,N,H,W)
            dets  = outputs[0]
            masks = outputs[1]

            dets  = dets[0]  if dets.ndim == 3  else dets   # (N, 5)
            masks = masks[0] if masks.ndim == 4 else masks  # (N, mH, mW)

            if dets.shape[0] == 0:
                return []

            boxes_lbspace = dets[:, :4]
            scores        = dets[:, 4]
            labels        = np.zeros(len(scores), dtype=np.int64)

        elif num_outputs == 1:
            # 格式 C：单输出合并格式 (1, N, 5+num_classes+mask_dim)
            raw = outputs[0]
            raw = raw[0] if raw.ndim == 3 else raw  # (N, C)

            if raw.shape[0] == 0:
                return []

            boxes_lbspace = raw[:, :4]
            scores        = raw[:, 4]
            labels        = np.zeros(len(scores), dtype=np.int64)
            # 格式 C 不含独立掩码输出，返回空掩码
            masks         = None

        else:
            print(f"RTMDet-Ins: 未知输出数量 {num_outputs}，跳过后处理")
            return []

        # ------------------------------------------------------------------
        # 置信度过滤
        # ------------------------------------------------------------------
        keep_mask = scores >= self.conf_threshold
        if not np.any(keep_mask):
            return []

        boxes_lbspace = boxes_lbspace[keep_mask]
        scores        = scores[keep_mask]
        labels        = labels[keep_mask] if labels is not None else None
        if masks is not None:
            masks = masks[keep_mask]

        # ------------------------------------------------------------------
        # 坐标逆变换：letterbox 空间 → 原始图像空间
        # ------------------------------------------------------------------
        boxes_orig = self.preprocessor.inverse_transform_boxes(boxes_lbspace, meta)

        # ------------------------------------------------------------------
        # 组装结果
        # ------------------------------------------------------------------
        results = []
        for i in range(len(boxes_orig)):
            x1, y1, x2, y2 = boxes_orig[i].tolist()
            score = float(scores[i])
            bbox  = [x1, y1, x2, y2]

            # 掩码处理
            if masks is not None:
                mask_crop = masks[i]  # (mH, mW)，可能是 float 或 uint8
                full_mask = resize_mask_to_original(mask_crop, bbox, orig_h, orig_w)
            else:
                # 无掩码输出时，用 bbox 区域生成矩形掩码
                full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                ix1 = int(max(0, round(x1)))
                iy1 = int(max(0, round(y1)))
                ix2 = int(min(orig_w, round(x2)))
                iy2 = int(min(orig_h, round(y2)))
                full_mask[iy1:iy2, ix1:ix2] = 1

            # 质心提取
            center = mask_to_center(full_mask)
            if center is None:
                # 掩码为空时退回到 bbox 中心
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            # 椭圆拟合
            ellipse_params = mask_to_ellipse(full_mask)

            # 类别 ID（0=egg 好蛋, 1=invalidegg 坏蛋）
            cls_id = int(labels[i]) if labels is not None else 0

            results.append({
                'bbox':          bbox,
                'mask':          full_mask,
                'center':        center,
                'ellipse_params': ellipse_params,
                'score':         score,
                'class_id':      cls_id,
                'is_invalid':    cls_id == 1,
            })

        return results

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    def warmup(self, num_iters: int = 2):
        """
        执行模型预热，减少首次推理延迟。

        Args:
            num_iters: 预热迭代次数（默认 2）
        """
        if self.session is None:
            return

        print(f"RTMDet-Ins: 开始预热（{num_iters} 次）...")
        dummy = np.random.randint(0, 255, (self.target_size, self.target_size, 3), dtype=np.uint8)
        for _ in range(num_iters):
            self.infer(dummy)
        print("RTMDet-Ins: 预热完成")

    def __del__(self):
        """释放推理会话资源。"""
        self.session = None
