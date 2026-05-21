# coding=utf-8
"""
RTMDet-OBB-P2 二维码旋转框检测推理模块
基于 ONNX Runtime 实现，支持 GPU/FP16 加速
@project: EGGRECORDQT
@Author：lzy
@file： rtmdet_obb_inference.py
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from model.inference.preprocessor import RTMDetPreprocessor
from model.utils.exception import exception_handler


# ---------------------------------------------------------------------------
# OBB → HBB 工具函数（模块级，可独立使用）
# ---------------------------------------------------------------------------

def obb_to_hbb(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Tuple[float, float, float, float]:
    """
    将旋转框参数转换为水平外接矩形（Axis-Aligned Bounding Box）。

    使用 cv2.boxPoints 计算旋转矩形的四个顶点，再取 x/y 的最小/最大值。

    Args:
        cx:        旋转框中心 x 坐标
        cy:        旋转框中心 y 坐标
        w:         旋转框宽度
        h:         旋转框高度
        angle_deg: 旋转角度（度），遵循 OpenCV 约定

    Returns:
        (x1, y1, x2, y2) 水平外接矩形，左上角和右下角坐标
    """
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
    pts = cv2.boxPoints(rect)          # shape (4, 2)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return x1, y1, x2, y2


def obb_array_to_hbb(rotated_boxes: np.ndarray) -> np.ndarray:
    """
    批量将旋转框数组转换为水平外接矩形数组。

    Args:
        rotated_boxes: shape (N, 5)，列为 [cx, cy, w, h, angle_deg]

    Returns:
        shape (N, 4)，列为 [x1, y1, x2, y2]
    """
    if rotated_boxes is None or len(rotated_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    hbbs = []
    for row in rotated_boxes:
        cx, cy, w, h, angle = row[:5]
        hbbs.append(obb_to_hbb(cx, cy, w, h, angle))
    return np.array(hbbs, dtype=np.float32)


# ---------------------------------------------------------------------------
# 旋转 NMS
# ---------------------------------------------------------------------------

def _rotated_iou_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    计算两个旋转框的近似 IoU（通过 HBB 外接矩形近似）。

    Args:
        box_a: [cx, cy, w, h, angle]
        box_b: [cx, cy, w, h, angle]

    Returns:
        IoU 值 [0, 1]
    """
    def _box_area(box):
        return float(box[2]) * float(box[3])

    # 用 HBB 近似计算 IoU
    x1a, y1a, x2a, y2a = obb_to_hbb(*box_a[:5])
    x1b, y1b, x2b, y2b = obb_to_hbb(*box_b[:5])

    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def rotated_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    对旋转框执行非极大值抑制（NMS）。

    Args:
        boxes:         shape (N, 5)，列为 [cx, cy, w, h, angle]
        scores:        shape (N,)，置信度分数
        iou_threshold: IoU 阈值，超过此值的重叠框将被抑制

    Returns:
        保留框的索引数组，shape (K,)
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # 按分数降序排列
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(idx)

        if len(order) == 1:
            break

        # 计算当前最高分框与其余框的 IoU
        rest = order[1:]
        suppress = []
        for j, other_idx in enumerate(rest):
            iou = _rotated_iou_single(boxes[idx], boxes[other_idx])
            if iou > iou_threshold:
                suppress.append(j)

        # 移除被抑制的框（注意 suppress 是 rest 中的索引）
        mask = np.ones(len(rest), dtype=bool)
        mask[suppress] = False
        order = rest[mask]

    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# 主推理类
# ---------------------------------------------------------------------------

class RTMDetOBBInference:
    """
    RTMDet-OBB-P2 二维码旋转框检测推理器

    支持三种 ONNX 输出格式：
      - 格式 9T：9 raw head tensors [cls×3, bbox×3, angle×3] → decode + rotated NMS
      - 格式 A：(1, N, 6)  → [cx, cy, w, h, angle, score]（单类或最高分类）
      - 格式 B：(1, N, 5+num_classes) → [cx, cy, w, h, angle, cls0_score, cls1_score, ...]

    类别定义：
      - class 0: valid_qr   （有效二维码）
      - class 1: invalid_qr （无效二维码）
    """

    NUM_CLASSES = 2
    CLASS_VALID_QR   = 0
    CLASS_INVALID_QR = 1

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float  = 0.5,
        target_size: int      = 640,
        use_fp16: bool        = False,
    ):
        """
        初始化 RTMDet-OBB 推理器。

        Args:
            model_path:      ONNX 模型文件路径
            conf_threshold:  置信度过滤阈值（默认 0.5）
            iou_threshold:   旋转 NMS IoU 阈值（默认 0.5）
            target_size:     推理分辨率（正方形，默认 640）
            use_fp16:        是否使用 FP16 输入（需要 GPU 支持）
        """
        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.target_size    = target_size
        self.use_fp16       = use_fp16

        self.session     = None
        self.input_name  = None
        self.output_names: List[str] = []

        self.preprocessor = RTMDetPreprocessor(target_size=target_size)

        self._init_session()

        # Hard-fail import policy — inside __init__(), NOT at module top level.
        # Rationale: model/inference/__init__.py eagerly imports RTMDetOBBInference
        # (the class symbol) for YOLO-path users. If obb_postprocess were imported
        # at module top level, a missing module would crash unrelated import paths.
        # The check activates only when RTMDetOBBInference is actually instantiated.
        try:
            from model.inference.obb_postprocess import obb_postprocess as _obb_pp
            self._obb_postprocess = _obb_pp
        except Exception as e:
            raise ImportError(
                f"Failed to import model.inference.obb_postprocess: {e}\n"
                "The deploy-side OBB postprocess module is required for "
                "RTMDet-OBB inference. Ensure model/inference/obb_postprocess.py exists."
            ) from e

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_session(self):
        """初始化 ONNX Runtime 推理会话，优先使用 GPU（CUDAExecutionProvider）。"""
        # 先检查模型文件是否存在（在导入 onnxruntime 之前，确保错误信息清晰）
        import os
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"RTMDet-OBB 模型文件未找到: {self.model_path}\n"
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
            print("RTMDet-OBB: 使用 CUDAExecutionProvider")
        else:
            print("RTMDet-OBB: CUDA 不可用，回退到 CPUExecutionProvider")

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
        print(f"RTMDet-OBB 模型加载成功: {self.model_path}")
        print(f"  输入: {inp.name}, shape={inp.shape}, dtype={inp.type}")
        for out in self.session.get_outputs():
            print(f"  输出: {out.name}, shape={out.shape}, dtype={out.type}")

        # 打印实际使用的 Provider
        active_providers = self.session.get_providers()
        print(f"  活跃 Provider: {active_providers}")

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def infer(self, image: np.ndarray) -> List[Dict]:
        """
        对单帧图像执行 QR 码旋转框检测。

        Args:
            image: BGR 图像，shape (H, W, 3)，dtype uint8

        Returns:
            检测结果列表，每个元素为字典：
              {
                'rotated_box':    [cx, cy, w, h, angle],  # 原始图像坐标
                'hbb':            [x1, y1, x2, y2],       # 水平外接矩形，原始图像坐标
                'score':          float,                   # 置信度分数
                'class_id':       int,                     # 0=valid_qr, 1=invalid_qr
                'validity_score': float,                   # 1.0 if class_id==0 else 0.0
              }
        """
        if self.session is None:
            print("RTMDet-OBB: 推理会话未初始化，跳过推理")
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
        解析 ONNX 输出，执行置信度过滤、旋转 NMS，并映射回原始图像坐标。

        支持三种输出格式：
          - 格式 9T: 9 raw head tensors (cls×3, bbox×3, angle×3) — 9-tensor decode+NMS
          - 格式 A:  (1, N, 6)  → [cx, cy, w, h, angle, score]
          - 格式 B:  (1, N, 5+num_classes) → [cx, cy, w, h, angle, cls0, cls1, ...]

        Args:
            outputs: ONNX Runtime 输出张量列表
            meta:    预处理元信息（用于坐标逆变换）

        Returns:
            过滤并 NMS 后的检测结果列表
        """
        # --- 9-tensor branch: raw RTMDet OBB head output ---
        # This check MUST come first, before single-tensor branches.
        if isinstance(outputs, (list, tuple)) and len(outputs) == 9:
            return self._postprocess_nine_tensor(outputs, meta)

        # --- Single-tensor branches (legacy mmdeploy formats) ---
        # 取第一个输出张量，去掉 batch 维度
        raw = outputs[0]  # (1, N, C) 或 (N, C)
        if raw.ndim == 3:
            raw = raw[0]  # (N, C)

        if raw.shape[0] == 0:
            return []

        num_cols = raw.shape[1]

        # 判断输出格式
        if num_cols == 6:
            # 格式 A: [cx, cy, w, h, angle, score]（单类，无类别区分）
            boxes_lbspace = raw[:, :5]   # (N, 5)
            scores        = raw[:, 5]    # (N,)
            class_ids     = np.zeros(len(scores), dtype=np.int64)  # 默认 valid_qr
        elif num_cols >= 5 + self.NUM_CLASSES:
            # 格式 B: [cx, cy, w, h, angle, cls0_score, cls1_score, ...]
            boxes_lbspace = raw[:, :5]                          # (N, 5)
            cls_scores    = raw[:, 5:5 + self.NUM_CLASSES]      # (N, num_classes)
            class_ids     = np.argmax(cls_scores, axis=1).astype(np.int64)
            scores        = cls_scores[np.arange(len(class_ids)), class_ids]
        else:
            # Unrecognized format — structured warning with shapes
            import logging
            logging.warning(
                "RTMDet-OBB: unrecognized output format. "
                f"Expected 9 tensors or single tensor with 6 or {5 + self.NUM_CLASSES}+ cols, "
                f"got {len(outputs)} tensor(s) with shapes: "
                f"{[o.shape for o in outputs]}"
            )
            return []

        # 置信度过滤
        keep_mask = scores >= self.conf_threshold
        if not np.any(keep_mask):
            return []

        boxes_lbspace = boxes_lbspace[keep_mask]
        scores        = scores[keep_mask]
        class_ids     = class_ids[keep_mask]

        # 旋转 NMS（按类别分别执行）
        final_indices = self._class_aware_rotated_nms(boxes_lbspace, scores, class_ids)

        boxes_lbspace = boxes_lbspace[final_indices]
        scores        = scores[final_indices]
        class_ids     = class_ids[final_indices]

        # 坐标逆变换：letterbox 空间 → 原始图像空间
        boxes_orig = self.preprocessor.inverse_transform_rotated_boxes(boxes_lbspace, meta)

        # 组装结果
        results = []
        for i in range(len(boxes_orig)):
            cx, cy, w, h, angle = boxes_orig[i].tolist()
            cid   = int(class_ids[i])
            score = float(scores[i])

            x1, y1, x2, y2 = obb_to_hbb(cx, cy, w, h, angle)

            # validity_score：class 0 (valid_qr) → 1.0，class 1 (invalid_qr) → 0.0
            validity_score = 1.0 if cid == self.CLASS_VALID_QR else 0.0

            results.append({
                'rotated_box':    [cx, cy, w, h, angle],
                'hbb':            [x1, y1, x2, y2],
                'score':          score,
                'class_id':       cid,
                'validity_score': validity_score,
            })

        return results

    def _class_aware_rotated_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        """
        按类别分别执行旋转 NMS，返回保留框的全局索引。

        Args:
            boxes:     (N, 5) [cx, cy, w, h, angle]
            scores:    (N,)
            class_ids: (N,)

        Returns:
            保留框的全局索引数组
        """
        all_keep = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_indices = np.where(cls_mask)[0]
            cls_boxes  = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            keep_local = rotated_nms(cls_boxes, cls_scores, self.iou_threshold)
            all_keep.extend(cls_indices[keep_local].tolist())

        return np.array(all_keep, dtype=np.int64)

    # ------------------------------------------------------------------
    # 9-tensor raw head output postprocessing
    # ------------------------------------------------------------------

    def _postprocess_nine_tensor(self, outputs: List[np.ndarray], meta: dict) -> List[Dict]:
        """
        Decode 9-tensor raw RTMDet OBB head output via the deploy postprocess module,
        apply letterbox inverse transform, and assemble detection result dicts.

        The deploy postprocess module (model.inference.obb_postprocess) handles:
          - FPN grid generation, distance2obb decode, sigmoid, score filtering
          - cv2-based rotated NMS

        This method handles:
          - Calling obb_postprocess with appropriate thresholds
          - Inverse-transforming (cx, cy, w, h) from letterbox space to original image space
          - Preserving angle (le90 radians) through the inverse transform
          - Computing HBB and validity_score to produce the standard detection dict format

        Args:
            outputs: List of 9 numpy arrays (raw RTMDet OBB head tensors)
            meta:    Preprocessor meta dict with 'pad_top', 'pad_left', 'scale', 'orig_shape'

        Returns:
            Detection results list, same format as (N,6) and (N,5+C) branches:
              [{'rotated_box': [...], 'hbb': [...], 'score': float,
                'class_id': int, 'validity_score': float}, ...]
        """
        # Decode + rotated NMS via deploy postprocess module
        dets, labels = self._obb_postprocess(
            outputs,
            img_size=self.target_size,
            score_thr=self.conf_threshold,
            nms_iou_thr=self.iou_threshold,
        )

        if len(dets) == 0:
            return []

        # dets: (N, 6) [cx, cy, w, h, angle_rad, score] in letterbox space
        # Separate boxes (first 5 cols) and scores (col 5)
        boxes_lbspace = dets[:, :5]   # (N, 5) [cx, cy, w, h, angle_rad]
        scores = dets[:, 5]           # (N,)
        class_ids = labels            # (N,) int64

        # Inverse transform: letterbox space → original image space
        # inverse_transform_rotated_boxes transforms cx, cy, w, h (cols 0-3)
        # and preserves angle (col 4) unchanged
        boxes_orig = self.preprocessor.inverse_transform_rotated_boxes(boxes_lbspace, meta)

        # Assemble results — same dict format as (N,6) and (N,5+C) branches
        results = []
        for i in range(len(boxes_orig)):
            cx, cy, w, h, angle = boxes_orig[i].tolist()
            cid = int(class_ids[i])
            score = float(scores[i])

            # HBB from rotated box (angle is in radians, obb_to_hbb expects degrees)
            angle_deg = float(np.degrees(angle))
            x1, y1, x2, y2 = obb_to_hbb(cx, cy, w, h, angle_deg)

            # validity_score: class 0 (valid_qr) → 1.0, class 1 (invalid_qr) → 0.0
            validity_score = 1.0 if cid == self.CLASS_VALID_QR else 0.0

            results.append({
                'rotated_box':    [cx, cy, w, h, angle],
                'hbb':            [x1, y1, x2, y2],
                'score':          score,
                'class_id':       cid,
                'validity_score': validity_score,
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

        print(f"RTMDet-OBB: 开始预热（{num_iters} 次）...")
        dummy = np.random.randint(0, 255, (self.target_size, self.target_size, 3), dtype=np.uint8)
        for _ in range(num_iters):
            self.infer(dummy)
        print("RTMDet-OBB: 预热完成")

    def __del__(self):
        """释放推理会话资源。"""
        self.session = None
