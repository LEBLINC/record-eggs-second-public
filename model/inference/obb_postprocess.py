# coding=utf-8
"""
obb_postprocess.py — Deploy-side OBB postprocessing for RTMDet raw ONNX outputs.

Self-contained decode + rotated NMS for the 9-tensor RTMDet OBB head output.
Imports ONLY from: Python standard library, numpy, cv2.
Does NOT import: mmcv, mmdet, mmrotate, mmdeploy, torch, or anything from training/.

Angle conventions:
  - Internal computation and public return value: le90 radians, range [-π/2, π/2)
  - cv2.rotatedRectangleIntersection calls: OpenCV degrees, converted via numpy.degrees()

Nine-tensor output schema (3 FPN levels × {cls, bbox, angle}):
    Index 0: cls_s8    (1, num_classes, 80, 80)
    Index 1: cls_s16   (1, num_classes, 40, 40)
    Index 2: cls_s32   (1, num_classes, 20, 20)
    Index 3: bbox_s8   (1, 4, 80, 80)
    Index 4: bbox_s16  (1, 4, 40, 40)
    Index 5: bbox_s32  (1, 4, 20, 20)
    Index 6: angle_s8  (1, 1, 80, 80)
    Index 7: angle_s16 (1, 1, 40, 40)
    Index 8: angle_s32 (1, 1, 20, 20)

@project: EGGRECORDQT
"""

import numpy as np
import cv2
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Anchor-free grid generation
# ---------------------------------------------------------------------------

def _make_grid(feat_h: int, feat_w: int, stride: int) -> np.ndarray:
    """
    Build (H*W, 2) grid of anchor-free center points in input-image coordinates.
    Points are at (stride/2 + col*stride, stride/2 + row*stride).

    Args:
        feat_h: Feature map height
        feat_w: Feature map width
        stride: FPN stride for this level

    Returns:
        points: (H*W, 2) array of (x, y) center coordinates
    """
    ys = np.arange(feat_h, dtype=np.float32) * stride + stride / 2.0
    xs = np.arange(feat_w, dtype=np.float32) * stride + stride / 2.0
    grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W) each
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (H*W, 2)
    return points


# ---------------------------------------------------------------------------
# Distance-to-OBB decode (le90 convention)
# ---------------------------------------------------------------------------

def _distance2obb(points: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Decode (l, t, r, b, angle) predictions relative to anchor points.

    Implements the RTMDet distance-to-bbox transform:
        cx = px + (r - l) / 2
        cy = py + (b - t) / 2
        w  = l + r
        h  = t + b

    Args:
        points: (N, 2) — (px, py) anchor-free grid points
        pred:   (N, 5) — (l, t, r, b, angle_raw) where distances are already
                scaled by stride

    Returns:
        boxes: (N, 5) — (cx, cy, w, h, angle) in le90 convention
    """
    l, t, r, b = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    angle = pred[:, 4]  # already in radians (le90)

    x1 = points[:, 0] - l
    y1 = points[:, 1] - t
    x2 = points[:, 0] + r
    y2 = points[:, 1] + b

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    # Clip angle to valid le90 range [-π/2, π/2)
    angle = np.clip(angle, -np.pi / 2.0, np.pi / 2.0)

    return np.stack([cx, cy, w, h, angle], axis=1)  # (N, 5)


# ---------------------------------------------------------------------------
# Angle convention conversion
# ---------------------------------------------------------------------------

def _le90_to_cv2_degrees(angles_rad: np.ndarray) -> np.ndarray:
    """
    Convert le90 radians to the degree convention expected by
    cv2.rotatedRectangleIntersection.

    Formula: degrees = numpy.degrees(radians)

    The le90 convention stores angles in radians within [-π/2, π/2).
    OpenCV's RotatedRect uses degrees. The mapping is a direct unit conversion.

    Args:
        angles_rad: NumPy array of angles in le90 radians

    Returns:
        angles_deg: NumPy array of angles in degrees
    """
    return np.degrees(angles_rad)


# ---------------------------------------------------------------------------
# cv2-based rotated IoU
# ---------------------------------------------------------------------------

def _rotated_iou_cv2(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute rotated IoU between two OBB boxes using cv2.rotatedRectangleIntersection.

    Each box is (cx, cy, w, h, angle_rad) in le90 radians.
    Internally converts to OpenCV degree format for the cv2 call.

    IoU = intersection_area / (area_a + area_b - intersection_area),
    clamped to [0.0, 1.0].

    Handling of cv2.rotatedRectangleIntersection return values:
      - retval == 0 (INTERSECT_NONE): intersection area = 0
      - retval == 1 (INTERSECT_PARTIAL): area = cv2.contourArea(polygon)
      - retval == 2 (INTERSECT_FULL): area = min(area_a, area_b)

    Args:
        box_a: (5,) array [cx, cy, w, h, angle_rad]
        box_b: (5,) array [cx, cy, w, h, angle_rad]

    Returns:
        IoU value in [0.0, 1.0]
    """
    # Convert to OpenCV RotatedRect format: ((cx, cy), (w, h), angle_deg)
    angle_a_deg = float(np.degrees(box_a[4]))
    angle_b_deg = float(np.degrees(box_b[4]))

    rect_a = ((float(box_a[0]), float(box_a[1])),
              (float(box_a[2]), float(box_a[3])),
              angle_a_deg)
    rect_b = ((float(box_b[0]), float(box_b[1])),
              (float(box_b[2]), float(box_b[3])),
              angle_b_deg)

    area_a = float(box_a[2]) * float(box_a[3])
    area_b = float(box_b[2]) * float(box_b[3])

    # Handle degenerate boxes
    if area_a <= 0 or area_b <= 0:
        return 0.0

    retval, intersection_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)

    # cv2.rotatedRectangleIntersection retval meanings:
    #   0 = INTERSECT_NONE (no overlap)
    #   1 = INTERSECT_PARTIAL (polygon intersection)
    #   2 = INTERSECT_FULL (one rectangle fully contains the other)
    if retval == 0 or intersection_pts is None:
        # No intersection
        intersection_area = 0.0
    elif retval == 1:
        # Partial intersection — compute polygon area
        # intersection_pts shape: (K, 1, 2)
        ordered = cv2.convexHull(intersection_pts)
        intersection_area = float(cv2.contourArea(ordered))
    elif retval == 2:
        # Full containment — intersection is area of the smaller rectangle
        intersection_area = min(area_a, area_b)
    else:
        # Unexpected retval — treat as no intersection (defensive)
        intersection_area = 0.0

    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0

    iou = intersection_area / union_area
    # Clamp to [0.0, 1.0]
    return float(np.clip(iou, 0.0, 1.0))


# ---------------------------------------------------------------------------
# cv2-based greedy rotated NMS
# ---------------------------------------------------------------------------

def _rotated_nms_cv2(boxes: np.ndarray, scores: np.ndarray,
                     iou_thr: float = 0.1) -> np.ndarray:
    """
    Greedy rotated NMS using cv2-based rotated IoU.

    Sorts candidates by score descending, iteratively keeps the top-scoring
    remaining box, and suppresses any remaining box whose rotated IoU with
    the kept box is >= iou_thr.

    No AABB fallback. This is the single canonical rotated NMS implementation
    for the deploy environment.

    Args:
        boxes:   (N, 5) array [cx, cy, w, h, angle_rad]
        scores:  (N,) array of confidence scores
        iou_thr: IoU threshold for suppression (default 0.1)

    Returns:
        keep: (K,) int64 array of kept indices into the original arrays
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(idx)

        if len(order) == 1:
            break

        # Compute IoU of top box against all remaining
        rest = order[1:]
        suppress_mask = np.zeros(len(rest), dtype=bool)

        for j, other_idx in enumerate(rest):
            iou = _rotated_iou_cv2(boxes[idx], boxes[other_idx])
            if iou >= iou_thr:
                suppress_mask[j] = True

        # Keep only non-suppressed boxes
        order = rest[~suppress_mask]

    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main post-processing entry point
# ---------------------------------------------------------------------------

def obb_postprocess(
    raw_outputs: List[np.ndarray],
    img_size: int = 640,
    score_thr: float = 0.05,
    nms_pre: int = 2000,
    nms_iou_thr: float = 0.1,
    max_per_img: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode 9-tensor RTMDet OBB head output and apply rotated NMS.

    Args:
        raw_outputs: List/tuple of 9 numpy arrays ordered as:
                     [cls_s8, cls_s16, cls_s32,
                      bbox_s8, bbox_s16, bbox_s32,
                      angle_s8, angle_s16, angle_s32]
        img_size:    Input image size (square, default 640)
        score_thr:   Pre-NMS score threshold (default 0.05)
        nms_pre:     Max candidates before NMS per level (default 2000)
        nms_iou_thr: Rotated NMS IoU threshold (default 0.1)
        max_per_img: Max detections after NMS (default 2000)

    Returns:
        dets:   np.ndarray (N, 6) float32 — [cx, cy, w, h, angle_rad, score]
                Angles are in le90 radians [-π/2, π/2).
        labels: np.ndarray (N,) int64 — class indices.
                For 4-class model (third round): 0=valid_qr, 1=invalid_qr,
                2=tag_qr, 3=obsolete_qr.
                For legacy 2-class model: 0=valid_qr, 1=invalid_qr.
    """
    strides = [8, 16, 32]
    num_levels = 3

    # Unpack outputs: [cls×3, bbox×3, angle×3]
    cls_scores = raw_outputs[0:3]    # each (1, num_classes, H, W)
    bbox_preds = raw_outputs[3:6]    # each (1, 4, H, W)
    angle_preds = raw_outputs[6:9]   # each (1, 1, H, W)

    all_boxes = []
    all_scores = []
    all_labels = []

    # Infer num_classes from the cls head channel dimension. Old training runs
    # had 2 classes (valid_qr / invalid_qr); the third-round model expanded to
    # 4 classes (+ tag_qr, obsolete_qr). Reading it from the tensor shape keeps
    # this postprocess valid for both checkpoints.
    num_classes = int(cls_scores[0].shape[1])

    for lvl in range(num_levels):
        stride = strides[lvl]

        cls = cls_scores[lvl][0]     # (num_classes, H, W)
        bbox = bbox_preds[lvl][0]    # (4, H, W)
        ang = angle_preds[lvl][0]    # (1, H, W)

        _, feat_h, feat_w = cls.shape

        # (num_classes, H, W) → (H*W, num_classes), apply sigmoid
        cls_flat = cls.reshape(num_classes, -1).T          # (H*W, num_classes)
        cls_flat = 1.0 / (1.0 + np.exp(-cls_flat))        # sigmoid

        # (4, H, W) → (H*W, 4)
        # NOTE: The ONNX bbox output is already in pixel units (stride already baked in
        # by the export). Do NOT multiply by stride again.
        bbox_flat = bbox.reshape(4, -1).T                  # (H*W, 4)

        # (1, H, W) → (H*W, 1)
        ang_flat = ang.reshape(1, -1).T                    # (H*W, 1)

        # Grid points
        points = _make_grid(feat_h, feat_w, stride)        # (H*W, 2)

        # Class score (max across classes) and label (argmax)
        scores = cls_flat.max(axis=1)                      # (H*W,)
        labels = cls_flat.argmax(axis=1)                   # (H*W,)

        # Pre-NMS score filter: retain only candidates >= score_thr
        mask = scores >= score_thr
        if mask.sum() == 0:
            continue

        scores = scores[mask]
        labels = labels[mask]
        bbox_flat = bbox_flat[mask]
        ang_flat = ang_flat[mask]
        points = points[mask]

        # Pre-NMS topk per level
        if len(scores) > nms_pre:
            topk_idx = np.argpartition(scores, -nms_pre)[-nms_pre:]
            scores = scores[topk_idx]
            labels = labels[topk_idx]
            bbox_flat = bbox_flat[topk_idx]
            ang_flat = ang_flat[topk_idx]
            points = points[topk_idx]

        # Decode: (l, t, r, b, angle) → (cx, cy, w, h, angle)
        pred5 = np.concatenate([bbox_flat, ang_flat], axis=1)  # (N, 5)
        boxes = _distance2obb(points, pred5)                    # (N, 5)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes = np.concatenate(all_boxes, axis=0)     # (N_total, 5)
    scores = np.concatenate(all_scores, axis=0)   # (N_total,)
    labels = np.concatenate(all_labels, axis=0)   # (N_total,)

    # Global topk before NMS
    if len(scores) > nms_pre:
        topk_idx = np.argpartition(scores, -nms_pre)[-nms_pre:]
        boxes = boxes[topk_idx]
        scores = scores[topk_idx]
        labels = labels[topk_idx]

    # Rotated NMS (cv2-based, no AABB fallback)
    keep = _rotated_nms_cv2(boxes, scores, iou_thr=nms_iou_thr)

    if len(keep) == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Max per image
    if len(scores) > max_per_img:
        topk_idx = np.argpartition(scores, -max_per_img)[-max_per_img:]
        boxes = boxes[topk_idx]
        scores = scores[topk_idx]
        labels = labels[topk_idx]

    # Sort by score descending
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]

    # Final output: (N, 6) = [cx, cy, w, h, angle_rad, score]
    dets = np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)
    return dets, labels.astype(np.int64)
