# coding=utf-8
"""
RTMDet 管道核心逻辑模块（无 GUI/HTTP 依赖）
提取自 QTInterface.py，便于独立测试。

包含：
  - _NumpyWrapper:          为 OCSort 直接调用提供 .data 属性包装
  - decode_qr_for_detections(): 对 QR 检测裁剪区域执行 zbar 解码，注入 decode_id
  - rtmdet_track_frame():   RTMDet 跟踪分支（单帧）
  - rtmdet_match_frame():   RTMDet 匹配分支（单帧）
  - draw_rtmdet_detections(): 在帧上绘制 RTMDet 检测结果
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

# 尝试导入 pyzbar；若环境缺失则降级为不解码（仍能跑通流程）
try:
    from pyzbar.pyzbar import decode as _zbar_decode
    _ZBAR_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    _zbar_decode = None
    _ZBAR_AVAILABLE = False
    print(f"pipeline_logic: pyzbar 不可用（{_e}），QR 解码将被跳过。"
          "可执行 `pip install pyzbar` 启用解码。")


class _NumpyWrapper:
    """
    将 numpy 数组包装为具有 .data 属性的对象，
    以满足 OCSort.update(dets, img) 中 dets = dets.data 的调用约定。
    """
    def __init__(self, array: np.ndarray):
        self.data = array


# ---------------------------------------------------------------------------
# QR 解码：把 hbb 裁剪区域送给 pyzbar，将 decode_id 写回检测字典
# ---------------------------------------------------------------------------

def decode_qr_for_detections(
    frame: np.ndarray,
    qr_dets: List[Dict],
    pad: int = 6,
    only_valid: bool = True,
) -> List[Dict]:
    """
    对每个 QR 检测的水平外接矩形区域执行 pyzbar 解码，
    解码成功时把字符串写入 det['decode_id']（也写入 'cage_id'，保证下游兼容）。

    Args:
        frame:       BGR 图像
        qr_dets:     QR 检测列表，每个元素来自 RTMDetOBBInference
        pad:         裁剪区域外扩像素数（适度外扩有助于 zbar 检测边角）
        only_valid:  True 时仅对 class_id == 0 (valid_qr) 的检测尝试解码

    Returns:
        原列表（同时原地修改字典），便于链式调用
    """
    if not _ZBAR_AVAILABLE or frame is None or not qr_dets:
        return qr_dets

    h, w = frame.shape[:2]
    for det in qr_dets:
        # 已解码过则跳过
        if det.get('decode_id') is not None:
            continue
        if only_valid and det.get('class_id', 0) != 0:
            continue

        hbb = det.get('hbb')
        if hbb is None or len(hbb) < 4:
            continue

        x1 = max(0, int(round(hbb[0])) - pad)
        y1 = max(0, int(round(hbb[1])) - pad)
        x2 = min(w, int(round(hbb[2])) + pad)
        y2 = min(h, int(round(hbb[3])) + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue

        crop = frame[y1:y2, x1:x2]
        try:
            results = _zbar_decode(crop)
        except Exception:  # noqa: BLE001
            continue
        if not results:
            continue

        # 取第一个能解码的结果
        for r in results:
            try:
                payload = r.data.decode('utf-8', errors='ignore').strip()
            except Exception:  # noqa: BLE001
                payload = ''
            if payload:
                det['decode_id'] = payload
                if det.get('cage_id') is None:
                    det['cage_id'] = payload
                break

    return qr_dets


# ---------------------------------------------------------------------------
# RTMDet → tracker_ids 反向匹配：保证 track_ids 与原 detections 顺序一一对应
# ---------------------------------------------------------------------------

def _hbb_iou(box_a, box_b) -> float:
    """计算两个 HBB 的 IoU。box: [x1, y1, x2, y2]"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_track_ids_to_detections(
    detections: Dict,
    tracker_output: np.ndarray,
    iou_thresh: float = 0.3,
) -> np.ndarray:
    """
    OCSort 输出的行数和顺序与输入的 detections 不一定一致，
    本函数通过 HBB IoU 反向匹配，给原始 detections（先 QR 后 egg 顺序）
    分配跟踪 ID；匹配不上的位置返回 0。

    Args:
        detections:      RTMDetInferenceEngine.track() 字典
        tracker_output:  OCSort 输出 (M, 5+)，前 4 列为 [x1,y1,x2,y2]，第 5 列 track_id
        iou_thresh:      最低 IoU 阈值

    Returns:
        长度等于 detections 中 QR + egg 总数的 int64 数组
    """
    qr_dets = detections.get('qr_detections', [])
    egg_dets = detections.get('egg_detections', [])
    n_total = len(qr_dets) + len(egg_dets)
    track_ids = np.zeros((n_total,), dtype=np.int64)

    if tracker_output is None or len(tracker_output) == 0 or n_total == 0:
        return track_ids

    if tracker_output.ndim == 1:
        tracker_output = tracker_output.reshape(1, -1)

    # 收集所有 detection 的 HBB，顺序：先 QR 后 egg
    det_boxes: List[List[float]] = []
    for d in qr_dets:
        det_boxes.append(list(d.get('hbb', [0, 0, 0, 0])[:4]))
    for d in egg_dets:
        det_boxes.append(list(d.get('bbox', [0, 0, 0, 0])[:4]))

    # 与 tracker 输出做贪心 IoU 匹配
    used_tracker = set()
    for det_idx, det_box in enumerate(det_boxes):
        best_iou = iou_thresh
        best_t = -1
        for t in range(tracker_output.shape[0]):
            if t in used_tracker:
                continue
            trk_box = tracker_output[t, :4]
            iou = _hbb_iou(det_box, trk_box)
            if iou > best_iou:
                best_iou = iou
                best_t = t
        if best_t >= 0:
            track_ids[det_idx] = int(tracker_output[best_t, 4])
            used_tracker.add(best_t)

    return track_ids


# ---------------------------------------------------------------------------
# 跟踪分支
# ---------------------------------------------------------------------------

def rtmdet_track_frame(
    engine,
    frame: np.ndarray,
    ocsort_tracker=None,
    decode_qr: bool = True,
) -> Tuple[Dict, np.ndarray]:
    """
    RTMDet 跟踪分支：对单帧执行检测 + 可选 OCSORT 跟踪 + 二维码解码。

    Args:
        engine:          RTMDetInferenceEngine 实例（或兼容对象）
        frame:           BGR 图像，shape (H, W, 3)
        ocsort_tracker:  OCSort 实例，或 None。RTMDet 路径默认不依赖 OCSORT
                         （cage_id 来自二维码解码），传入也安全
        decode_qr:       是否对 QR hbb 区域执行 pyzbar 解码（默认 True）

    Returns:
        (detections_dict, track_ids)
          - detections_dict: engine.track() 返回的字典（QR 字典中已注入 decode_id/cage_id）
          - track_ids:       与 detections（先 QR 后 egg）一一对应的跟踪 ID，
                             未启用 OCSORT 或匹配失败的位置为 0
    """
    from model.inference.result_adapter import ResultAdapter

    detections = engine.track(frame)
    if detections is None:
        detections = {'qr_detections': [], 'egg_detections': []}

    # QR 解码：把 cage_id 写入字典，供 TopologyMatcher 使用
    if decode_qr:
        decode_qr_for_detections(frame, detections.get('qr_detections', []))

    # 转换为 OCSORT 输入格式 [x1,y1,x2,y2,score,class]
    tracker_input = ResultAdapter.to_tracker_format(detections)
    n_total = (len(detections.get('qr_detections', []))
               + len(detections.get('egg_detections', [])))
    track_ids = np.zeros((n_total,), dtype=np.int64)

    if ocsort_tracker is not None and len(tracker_input) > 0:
        try:
            wrapped = _NumpyWrapper(tracker_input)
            tracker_output = ocsort_tracker.update(wrapped, frame)
            if tracker_output is not None and len(tracker_output) > 0:
                # 注意：OCSort 内部会做边缘过滤、二阶段关联等，输出行数和
                # 顺序与 tracker_input 不一致；用 IoU 反向匹配保证对齐。
                track_ids = _match_track_ids_to_detections(detections, tracker_output)
        except Exception as e:
            print(f"rtmdet_track_frame: OCSORT 跟踪异常: {e}")

    return detections, track_ids


# ---------------------------------------------------------------------------
# 匹配分支
# ---------------------------------------------------------------------------

def rtmdet_match_frame(
    frame: np.ndarray,
    track_results: Tuple[Dict, np.ndarray],
    topology_matcher=None,
) -> List[Dict]:
    """
    RTMDet 匹配分支：从跟踪结果中提取蛋中心点和 QR 检测，
    调用 TopologyMatcher 进行蛋-笼匹配，并在帧上绘制检测框。

    Args:
        frame:             BGR 图像（会被原地修改以绘制检测框）
        track_results:     (detections_dict, track_ids) 元组
        topology_matcher:  TopologyMatcher 实例，或 None

    Returns:
        TopologyMatcher.match() 返回的匹配结果列表，
        若 topology_matcher 为 None 则返回空列表。
    """
    detections, track_ids = track_results
    qr_dets = detections.get('qr_detections', [])
    egg_dets = detections.get('egg_detections', [])

    # 可视化：绘制检测框
    draw_rtmdet_detections(frame, qr_dets, egg_dets)

    # 提取 egg 中心点（保留 class_id，供后续质量统计使用）
    egg_centers: List[Tuple[float, float]] = []
    egg_meta: List[Dict] = []  # 与 egg_centers 一一对应
    for det in egg_dets:
        center = det.get('center')
        if center is not None:
            egg_centers.append((float(center[0]), float(center[1])))
        else:
            bbox = det.get('bbox', [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            egg_centers.append((cx, cy))
        egg_meta.append({
            'class_id':   det.get('class_id', 0),
            'is_invalid': bool(det.get('is_invalid', False)),
            'score':      det.get('score', 0.0),
        })

    # 构建 TopologyMatcher 所需的 qr_detections 格式
    tm_qr_dets: List[Dict] = []
    for det in qr_dets:
        hbb = det.get('hbb', [0, 0, 0, 0])
        cx = (hbb[0] + hbb[2]) / 2.0
        cy = (hbb[1] + hbb[3]) / 2.0
        tm_qr_dets.append({
            'center': (cx, cy),
            'hbb': hbb,
            'validity_score': det.get('validity_score', det.get('score', 0.0)),
            'decode_id': det.get('decode_id', None),
            'cage_id': det.get('cage_id', det.get('decode_id', None)),
        })

    # 调用 TopologyMatcher
    if topology_matcher is None:
        return []

    try:
        match_results = topology_matcher.match(
            egg_centers, tm_qr_dets, frame, egg_meta=egg_meta,
        )
        return match_results if match_results is not None else []
    except TypeError:
        # 兼容旧版 match() 不带 egg_meta 参数
        try:
            match_results = topology_matcher.match(egg_centers, tm_qr_dets, frame)
            return match_results if match_results is not None else []
        except Exception as e:
            print(f"rtmdet_match_frame: TopologyMatcher 异常: {e}")
            return []
    except Exception as e:
        print(f"rtmdet_match_frame: TopologyMatcher 异常: {e}")
        return []


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _to_int_points(pts: np.ndarray) -> np.ndarray:
    """
    把 cv2.boxPoints 返回的浮点顶点转为 int 数组。
    兼容 NumPy 1.x（np.int0）与 NumPy 2.x（np.int0 已移除）。
    """
    return pts.astype(np.int32)


def draw_rtmdet_detections(
    frame: np.ndarray,
    qr_dets: List[Dict],
    egg_dets: List[Dict],
) -> None:
    """
    在帧上绘制 RTMDet 检测结果（QR 框和蛋中心点）。

    Args:
        frame:    BGR 图像（原地修改）
        qr_dets:  QR 码检测列表，每个元素包含 hbb、rotated_box（可选）、score、validity_score
        egg_dets: 蛋检测列表，每个元素包含 bbox、center（可选）、score
    """
    try:
        # 绘制 QR 码水平外接矩形
        for det in qr_dets:
            hbb = det.get('hbb')
            if hbb is not None and len(hbb) >= 4:
                x1, y1, x2, y2 = int(hbb[0]), int(hbb[1]), int(hbb[2]), int(hbb[3])
                # valid_qr=绿色，invalid_qr=红色
                color = (0, 255, 0) if det.get('class_id', 0) == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                validity = det.get('validity_score', det.get('score', 0.0))
                decode_id = det.get('decode_id') or det.get('cage_id')
                if decode_id:
                    label = f"QR {decode_id} ({validity:.2f})"
                else:
                    label = f"QR {validity:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # 不再单独绘制旋转框轮廓，避免双框视觉干扰

        # 绘制蛋边界框和中心点
        for det in egg_dets:
            bbox = det.get('bbox')
            if bbox is not None and len(bbox) >= 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

            center = det.get('center')
            if center is not None:
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                score = det.get('score', 0.0)
                cv2.putText(frame, f"egg {score:.2f}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    except Exception as e:
        print(f"draw_rtmdet_detections: 绘制检测框异常: {e}")
