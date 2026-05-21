# coding=utf-8
"""
RTMDet 结果适配器
将 RTMDet 推理引擎输出转换为与 OCSORT 跟踪器和 MatchingCounting 兼容的格式
@project: EGGRECORDQT
@Author：lzy
@file： result_adapter.py
"""
import numpy as np
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 模拟 Ultralytics Tensor 接口的辅助类
# ---------------------------------------------------------------------------

class _MockTensor:
    """
    模拟 Ultralytics 张量对象，支持 .cpu().numpy() 调用链。

    unpack_results() 中使用的访问模式：
        results[0].boxes.xyxy.cpu().numpy().astype(int)
        results[0].boxes.cls.cpu().numpy().astype(int)
        results[0].boxes.id.cpu().numpy().astype(int)
        results[0].boxes.conf.cpu().numpy().astype(float)
    """

    def __init__(self, array: np.ndarray):
        self._array = array

    def cpu(self) -> '_MockTensor':
        """模拟 .cpu() 调用，返回自身（数据已在 CPU 上）。"""
        return self

    def numpy(self) -> np.ndarray:
        """返回底层 numpy 数组。"""
        return self._array


class _MockBoxes:
    """
    模拟 Ultralytics Boxes 对象，提供 xyxy、cls、id、conf 属性。
    """

    def __init__(
        self,
        xyxy: np.ndarray,
        cls: np.ndarray,
        ids: np.ndarray,
        conf: np.ndarray,
    ):
        """
        Args:
            xyxy: shape (N, 4)，[x1, y1, x2, y2]，float32
            cls:  shape (N,)，类别 ID，int64
            ids:  shape (N,)，跟踪 ID，int64
            conf: shape (N,)，置信度，float32
        """
        self.xyxy = _MockTensor(xyxy)
        self.cls  = _MockTensor(cls)
        self.id   = _MockTensor(ids)
        self.conf = _MockTensor(conf)


class _MockResult:
    """
    模拟 Ultralytics Results 对象，提供 names 和 boxes 属性。

    unpack_results() 访问模式：
        results[0].names   → dict {class_id: class_name}
        results[0].boxes   → _MockBoxes 对象
    """

    # 类别名称映射，与原始 YOLO 模型保持一致
    # 0 = egg（鸡蛋），1 = qr（二维码）
    NAMES = {0: 'egg', 1: 'qr'}

    def __init__(self, boxes: _MockBoxes):
        self.names = self.NAMES
        self.boxes = boxes


# ---------------------------------------------------------------------------
# ResultAdapter 主类
# ---------------------------------------------------------------------------

class ResultAdapter:
    """
    RTMDet 结果适配器。

    将 RTMDetInferenceEngine.track() 返回的字典格式转换为：
      1. OCSORT 跟踪器所需的 [x1, y1, x2, y2, score, class] ndarray 格式
      2. MatchingCounting.match() 所需的 legacy results 格式（兼容 unpack_results()）

    RTMDet 引擎输出格式（detections dict）：
      - qr_detections:  List[dict]，每个元素包含：
                          hbb=[x1,y1,x2,y2], score, class_id (0=valid_qr, 1=invalid_qr)
      - egg_detections: List[dict]，每个元素包含：
                          bbox=[x1,y1,x2,y2], score

    Legacy 类别映射（与原始 YOLO 模型一致）：
      - 0 = egg
      - 1 = qr
    """

    # Legacy 类别 ID（与 unpack_results() 中 names 字典对应）
    CLASS_EGG = 0
    CLASS_QR  = 1

    @staticmethod
    def to_tracker_format(detections: dict) -> np.ndarray:
        """
        将 RTMDet 检测结果转换为 OCSORT 跟踪器所需的格式。

        OCSORT 输入格式：每行 [x1, y1, x2, y2, score, class_id]

        Args:
            detections: RTMDetInferenceEngine.track() 返回的字典，包含：
                          - qr_detections:  List[dict]，含 hbb 和 score
                          - egg_detections: List[dict]，含 bbox 和 score

        Returns:
            shape (N, 6) 的 float32 ndarray，列为 [x1, y1, x2, y2, score, class_id]
            若无检测结果则返回 shape (0, 6) 的空数组
        """
        rows = []

        # 处理 QR 码检测（legacy class_id = 1）
        for det in detections.get('qr_detections', []):
            x1, y1, x2, y2 = det['hbb']
            score = float(det['score'])
            rows.append([float(x1), float(y1), float(x2), float(y2), score, float(ResultAdapter.CLASS_QR)])

        # 处理鸡蛋检测（legacy class_id = 0）
        for det in detections.get('egg_detections', []):
            x1, y1, x2, y2 = det['bbox']
            score = float(det['score'])
            rows.append([float(x1), float(y1), float(x2), float(y2), score, float(ResultAdapter.CLASS_EGG)])

        if not rows:
            return np.empty((0, 6), dtype=np.float32)

        return np.array(rows, dtype=np.float32)

    @staticmethod
    def to_legacy_results(
        detections: dict,
        track_ids: Optional[np.ndarray] = None,
    ) -> List['_MockResult']:
        """
        将 RTMDet 检测结果和跟踪 ID 转换为与 unpack_results() 兼容的 legacy 格式。

        unpack_results() 期望的访问模式：
            results[0].names                          → {0: 'egg', 1: 'qr'}
            results[0].boxes.xyxy.cpu().numpy()       → shape (N, 4)
            results[0].boxes.cls.cpu().numpy()        → shape (N,)
            results[0].boxes.id.cpu().numpy()         → shape (N,)
            results[0].boxes.conf.cpu().numpy()       → shape (N,)

        Args:
            detections: RTMDetInferenceEngine.track() 返回的字典
            track_ids:  OCSORT 分配的跟踪 ID 数组，shape (N,)，与 to_tracker_format()
                        输出的行顺序对应（先 QR 后 egg）。
                        若为 None，则自动分配从 1 开始的连续 ID。

        Returns:
            长度为 1 的列表 [_MockResult]，与 Ultralytics results 列表格式一致，
            可直接传入 MatchingCounting.match(results, frame)。
        """
        qr_dets  = detections.get('qr_detections', [])
        egg_dets = detections.get('egg_detections', [])

        n_qr  = len(qr_dets)
        n_egg = len(egg_dets)
        n_total = n_qr + n_egg

        # 构建各属性数组（顺序：先 QR，后 egg，与 to_tracker_format 一致）
        xyxy_list  = []
        cls_list   = []
        conf_list  = []

        for det in qr_dets:
            x1, y1, x2, y2 = det['hbb']
            xyxy_list.append([float(x1), float(y1), float(x2), float(y2)])
            cls_list.append(ResultAdapter.CLASS_QR)
            conf_list.append(float(det['score']))

        for det in egg_dets:
            x1, y1, x2, y2 = det['bbox']
            xyxy_list.append([float(x1), float(y1), float(x2), float(y2)])
            cls_list.append(ResultAdapter.CLASS_EGG)
            conf_list.append(float(det['score']))

        if n_total == 0:
            # 无检测结果：返回空的 mock result
            xyxy_arr = np.empty((0, 4), dtype=np.float32)
            cls_arr  = np.empty((0,),   dtype=np.int64)
            ids_arr  = np.empty((0,),   dtype=np.int64)
            conf_arr = np.empty((0,),   dtype=np.float32)
        else:
            xyxy_arr = np.array(xyxy_list, dtype=np.float32)
            cls_arr  = np.array(cls_list,  dtype=np.int64)
            conf_arr = np.array(conf_list, dtype=np.float32)

            # 处理跟踪 ID
            if track_ids is not None:
                ids_arr = np.array(track_ids, dtype=np.int64).flatten()
                if len(ids_arr) != n_total:
                    # 长度不匹配时打印警告并回退到自动分配
                    print(
                        f"ResultAdapter: track_ids 长度 {len(ids_arr)} 与检测数量 {n_total} 不匹配，"
                        "回退到自动分配 ID"
                    )
                    ids_arr = np.arange(1, n_total + 1, dtype=np.int64)
            else:
                # 未提供跟踪 ID 时，自动分配从 1 开始的连续 ID
                ids_arr = np.arange(1, n_total + 1, dtype=np.int64)

        boxes  = _MockBoxes(xyxy=xyxy_arr, cls=cls_arr, ids=ids_arr, conf=conf_arr)
        result = _MockResult(boxes=boxes)

        return [result]

    @staticmethod
    def get_qr_detections(detections: dict) -> List[Dict]:
        """
        从检测结果中提取 QR 码检测列表。

        Args:
            detections: RTMDetInferenceEngine.track() 返回的字典

        Returns:
            QR 码检测列表，每个元素包含 hbb、score、class_id、validity_score
        """
        return detections.get('qr_detections', [])

    @staticmethod
    def get_egg_detections(detections: dict) -> List[Dict]:
        """
        从检测结果中提取鸡蛋检测列表。

        Args:
            detections: RTMDetInferenceEngine.track() 返回的字典

        Returns:
            鸡蛋检测列表，每个元素包含 bbox、mask、center、ellipse_params、score
        """
        return detections.get('egg_detections', [])

    @staticmethod
    def assign_track_ids(
        tracker_output: np.ndarray,
    ) -> np.ndarray:
        """
        从 OCSORT 跟踪器输出中提取跟踪 ID。

        OCSORT 输出格式：每行 [x1, y1, x2, y2, track_id, ...]
        （跟踪器在 to_tracker_format() 输出基础上追加 track_id 列）

        Args:
            tracker_output: OCSORT 输出数组，shape (N, 5+)，第 5 列为 track_id

        Returns:
            跟踪 ID 数组，shape (N,)，dtype int64
            若输入为空则返回空数组
        """
        if tracker_output is None or len(tracker_output) == 0:
            return np.empty((0,), dtype=np.int64)

        if tracker_output.ndim == 1:
            tracker_output = tracker_output.reshape(1, -1)

        if tracker_output.shape[1] < 5:
            print(f"ResultAdapter: 跟踪器输出列数 {tracker_output.shape[1]} < 5，无法提取 track_id")
            return np.empty((0,), dtype=np.int64)

        return tracker_output[:, 4].astype(np.int64)
