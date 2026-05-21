# coding=utf-8
"""
RTMDet-OBB 推理模块单元测试
测试 OBB→HBB 转换、旋转 NMS、后处理逻辑
@project: EGGRECORDQT
@file： test_rtmdet_obb_inference.py
"""
import math
import numpy as np
import pytest

from model.inference.rtmdet_obb_inference import (
    obb_to_hbb,
    obb_array_to_hbb,
    rotated_nms,
    RTMDetOBBInference,
)


# ---------------------------------------------------------------------------
# obb_to_hbb 测试
# ---------------------------------------------------------------------------

class TestObbToHbb:
    """测试 OBB → HBB 转换工具函数"""

    def test_axis_aligned_box_no_rotation(self):
        """无旋转时，HBB 应等于原始矩形"""
        x1, y1, x2, y2 = obb_to_hbb(cx=50, cy=50, w=40, h=20, angle_deg=0)
        assert abs(x1 - 30) < 1e-3, f"x1 expected ~30, got {x1}"
        assert abs(y1 - 40) < 1e-3, f"y1 expected ~40, got {y1}"
        assert abs(x2 - 70) < 1e-3, f"x2 expected ~70, got {x2}"
        assert abs(y2 - 60) < 1e-3, f"y2 expected ~60, got {y2}"

    def test_90_degree_rotation_swaps_wh(self):
        """旋转 90° 后，宽高互换，外接矩形尺寸应对应交换"""
        x1, y1, x2, y2 = obb_to_hbb(cx=50, cy=50, w=40, h=20, angle_deg=90)
        hbb_w = x2 - x1
        hbb_h = y2 - y1
        # 旋转 90° 后，原宽 40 变为高方向，原高 20 变为宽方向
        assert abs(hbb_w - 20) < 1e-3, f"HBB width expected ~20, got {hbb_w}"
        assert abs(hbb_h - 40) < 1e-3, f"HBB height expected ~40, got {hbb_h}"

    def test_45_degree_rotation_enlarges_hbb(self):
        """旋转 45° 后，外接矩形应比原始矩形更大"""
        w, h = 40.0, 20.0
        x1, y1, x2, y2 = obb_to_hbb(cx=100, cy=100, w=w, h=h, angle_deg=45)
        hbb_w = x2 - x1
        hbb_h = y2 - y1
        # 旋转 45° 后外接矩形边长 = (w+h)/sqrt(2) * sqrt(2) = w+h... 实际为 (w*cos45+h*sin45)*2
        expected_side = (w + h) * math.cos(math.radians(45))
        assert hbb_w > w, "旋转后 HBB 宽度应大于原始宽度"
        assert hbb_h > h, "旋转后 HBB 高度应大于原始高度"
        assert abs(hbb_w - expected_side) < 1.0, f"HBB width expected ~{expected_side:.1f}, got {hbb_w:.1f}"

    def test_center_preserved(self):
        """HBB 中心应与旋转框中心一致"""
        cx, cy = 123.0, 456.0
        x1, y1, x2, y2 = obb_to_hbb(cx=cx, cy=cy, w=60, h=30, angle_deg=37)
        hbb_cx = (x1 + x2) / 2
        hbb_cy = (y1 + y2) / 2
        assert abs(hbb_cx - cx) < 1e-3, f"HBB center x expected {cx}, got {hbb_cx}"
        assert abs(hbb_cy - cy) < 1e-3, f"HBB center y expected {cy}, got {hbb_cy}"

    def test_hbb_always_valid_rectangle(self):
        """HBB 的 x2 > x1 且 y2 > y1"""
        for angle in [0, 15, 30, 45, 60, 75, 90]:
            x1, y1, x2, y2 = obb_to_hbb(cx=200, cy=200, w=80, h=40, angle_deg=angle)
            assert x2 > x1, f"angle={angle}: x2 ({x2}) should be > x1 ({x1})"
            assert y2 > y1, f"angle={angle}: y2 ({y2}) should be > y1 ({y1})"

    def test_square_box_rotation_invariant_size(self):
        """正方形旋转后，外接矩形尺寸不变"""
        side = 50.0
        x1_0, y1_0, x2_0, y2_0 = obb_to_hbb(cx=100, cy=100, w=side, h=side, angle_deg=0)
        x1_45, y1_45, x2_45, y2_45 = obb_to_hbb(cx=100, cy=100, w=side, h=side, angle_deg=45)
        # 正方形旋转后外接矩形边长 = side * sqrt(2)
        assert abs((x2_0 - x1_0) - side) < 1e-3
        expected_rotated = side * math.sqrt(2)
        assert abs((x2_45 - x1_45) - expected_rotated) < 1.0


class TestObbArrayToHbb:
    """测试批量 OBB → HBB 转换"""

    def test_empty_input(self):
        result = obb_array_to_hbb(np.empty((0, 5), dtype=np.float32))
        assert result.shape == (0, 4)

    def test_single_box(self):
        boxes = np.array([[50, 50, 40, 20, 0]], dtype=np.float32)
        result = obb_array_to_hbb(boxes)
        assert result.shape == (1, 4)
        assert abs(result[0, 0] - 30) < 1e-3  # x1
        assert abs(result[0, 2] - 70) < 1e-3  # x2

    def test_multiple_boxes(self):
        boxes = np.array([
            [50, 50, 40, 20, 0],
            [100, 100, 60, 30, 45],
            [200, 200, 80, 40, 90],
        ], dtype=np.float32)
        result = obb_array_to_hbb(boxes)
        assert result.shape == (3, 4)
        # 每行 x2 > x1, y2 > y1
        for i in range(3):
            assert result[i, 2] > result[i, 0], f"row {i}: x2 should > x1"
            assert result[i, 3] > result[i, 1], f"row {i}: y2 should > y1"

    def test_none_input(self):
        result = obb_array_to_hbb(None)
        assert result.shape == (0, 4)


# ---------------------------------------------------------------------------
# rotated_nms 测试
# ---------------------------------------------------------------------------

class TestRotatedNMS:
    """测试旋转 NMS 函数"""

    def test_empty_input(self):
        keep = rotated_nms(np.empty((0, 5)), np.array([]), iou_threshold=0.5)
        assert len(keep) == 0

    def test_single_box_always_kept(self):
        boxes  = np.array([[50, 50, 40, 20, 0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = rotated_nms(boxes, scores, iou_threshold=0.5)
        assert list(keep) == [0]

    def test_non_overlapping_boxes_all_kept(self):
        """完全不重叠的框应全部保留"""
        boxes = np.array([
            [50,  50,  40, 20, 0],
            [500, 500, 40, 20, 0],
            [250, 250, 40, 20, 0],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = rotated_nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 3

    def test_identical_boxes_only_highest_score_kept(self):
        """完全重叠的框只保留分数最高的"""
        boxes = np.array([
            [50, 50, 40, 20, 0],
            [50, 50, 40, 20, 0],
            [50, 50, 40, 20, 0],
        ], dtype=np.float32)
        scores = np.array([0.7, 0.9, 0.8], dtype=np.float32)
        keep = rotated_nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        # 保留的应是分数最高的（索引 1）
        assert 1 in keep

    def test_high_iou_threshold_keeps_more(self):
        """IoU 阈值越高，保留的框越多"""
        boxes = np.array([
            [50, 50, 40, 20, 0],
            [55, 50, 40, 20, 0],  # 轻微偏移，有重叠
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        keep_strict = rotated_nms(boxes, scores, iou_threshold=0.1)
        keep_loose  = rotated_nms(boxes, scores, iou_threshold=0.9)
        assert len(keep_strict) <= len(keep_loose)

    def test_highest_score_box_always_in_result(self):
        """分数最高的框必须出现在结果中"""
        boxes = np.array([
            [50, 50, 40, 20, 0],
            [52, 50, 40, 20, 0],
            [48, 50, 40, 20, 0],
        ], dtype=np.float32)
        scores = np.array([0.6, 0.95, 0.7], dtype=np.float32)
        keep = rotated_nms(boxes, scores, iou_threshold=0.5)
        assert 1 in keep  # 索引 1 分数最高


# ---------------------------------------------------------------------------
# RTMDetOBBInference 后处理逻辑测试（不需要真实模型）
# ---------------------------------------------------------------------------

class TestRTMDetOBBPostprocess:
    """测试 RTMDetOBBInference 的后处理逻辑（绕过 ONNX 会话）"""

    def _make_inference_obj(self):
        """创建一个跳过模型加载的推理对象（用于测试后处理）"""
        obj = RTMDetOBBInference.__new__(RTMDetOBBInference)
        obj.conf_threshold = 0.5
        obj.iou_threshold  = 0.5
        obj.target_size    = 640
        obj.use_fp16       = False
        obj.session        = None
        obj.input_name     = None
        obj.output_names   = []
        from model.inference.preprocessor import RTMDetPreprocessor
        obj.preprocessor   = RTMDetPreprocessor(target_size=640)
        return obj

    def _make_meta(self):
        """构造一个恒等变换的 meta（无缩放无 padding）"""
        return {
            'orig_shape': (640, 640),
            'pad_top':    0,
            'pad_left':   0,
            'scale':      1.0,
        }

    def test_format_a_single_detection(self):
        """格式 A (N, 6)：单个检测框，分数超过阈值"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        # [cx, cy, w, h, angle, score]
        raw = np.array([[[100, 200, 50, 30, 15, 0.9]]], dtype=np.float32)  # (1, 1, 6)
        results = obj._postprocess([raw], meta)

        assert len(results) == 1
        r = results[0]
        assert 'rotated_box' in r
        assert 'hbb' in r
        assert 'score' in r
        assert 'class_id' in r
        assert 'validity_score' in r
        assert abs(r['score'] - 0.9) < 1e-5
        assert r['class_id'] == 0  # 格式 A 默认 valid_qr
        assert r['validity_score'] == 1.0

    def test_format_a_below_threshold_filtered(self):
        """格式 A：分数低于阈值的框应被过滤"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        raw = np.array([[[100, 200, 50, 30, 0, 0.3]]], dtype=np.float32)
        results = obj._postprocess([raw], meta)
        assert len(results) == 0

    def test_format_b_two_classes(self):
        """格式 B (N, 7)：两类别，选择分数最高的类"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        # [cx, cy, w, h, angle, cls0_score, cls1_score]
        # cls1 分数更高 → class_id=1 (invalid_qr)
        raw = np.array([[[100, 200, 50, 30, 0, 0.3, 0.8]]], dtype=np.float32)
        results = obj._postprocess([raw], meta)

        assert len(results) == 1
        r = results[0]
        assert r['class_id'] == 1
        assert abs(r['score'] - 0.8) < 1e-5
        assert r['validity_score'] == 0.0  # invalid_qr

    def test_format_b_valid_qr_class(self):
        """格式 B：cls0 分数更高 → valid_qr，validity_score=1.0"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        raw = np.array([[[100, 200, 50, 30, 0, 0.9, 0.2]]], dtype=np.float32)
        results = obj._postprocess([raw], meta)

        assert len(results) == 1
        assert results[0]['class_id'] == 0
        assert results[0]['validity_score'] == 1.0

    def test_empty_output(self):
        """空输出应返回空列表"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        raw = np.zeros((1, 0, 6), dtype=np.float32)
        results = obj._postprocess([raw], meta)
        assert results == []

    def test_hbb_coordinates_valid(self):
        """HBB 坐标应满足 x2>x1, y2>y1"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        raw = np.array([[[300, 300, 80, 40, 30, 0.85]]], dtype=np.float32)
        results = obj._postprocess([raw], meta)

        assert len(results) == 1
        x1, y1, x2, y2 = results[0]['hbb']
        assert x2 > x1
        assert y2 > y1

    def test_rotated_box_has_five_elements(self):
        """rotated_box 应包含 5 个元素 [cx, cy, w, h, angle]"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        raw = np.array([[[150, 250, 60, 30, 45, 0.75]]], dtype=np.float32)
        results = obj._postprocess([raw], meta)

        assert len(results) == 1
        assert len(results[0]['rotated_box']) == 5

    def test_multiple_detections_nms_applied(self):
        """多个高度重叠的框经 NMS 后应只保留一个"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        # 三个几乎完全重叠的框
        raw = np.array([[
            [100, 100, 50, 30, 0, 0.9],
            [101, 100, 50, 30, 0, 0.8],
            [100, 101, 50, 30, 0, 0.7],
        ]], dtype=np.float32)
        results = obj._postprocess([raw], meta)
        assert len(results) == 1
        assert abs(results[0]['score'] - 0.9) < 1e-5

    def test_coordinate_inverse_transform_applied(self):
        """坐标逆变换：有 padding 和缩放时，坐标应正确映射回原始空间"""
        obj = self._make_inference_obj()
        # 模拟：原始图像 320x320，缩放到 640x640（scale=2.0），无 padding
        meta = {
            'orig_shape': (320, 320),
            'pad_top':    0,
            'pad_left':   0,
            'scale':      2.0,
        }
        # letterbox 空间中心 (200, 200)，原始空间应为 (100, 100)
        raw = np.array([[[200, 200, 100, 60, 0, 0.9]]], dtype=np.float32)
        results = obj._postprocess([raw], meta)

        assert len(results) == 1
        cx, cy, w, h, angle = results[0]['rotated_box']
        assert abs(cx - 100) < 1e-3, f"cx expected 100, got {cx}"
        assert abs(cy - 100) < 1e-3, f"cy expected 100, got {cy}"
        assert abs(w - 50) < 1e-3,   f"w expected 50, got {w}"
        assert abs(h - 30) < 1e-3,   f"h expected 30, got {h}"


# ---------------------------------------------------------------------------
# RTMDetOBBInference 初始化测试（FileNotFoundError）
# ---------------------------------------------------------------------------

class TestRTMDetOBBInferenceInit:
    """测试推理器初始化行为"""

    def test_missing_model_raises_file_not_found(self):
        """模型文件不存在时应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            RTMDetOBBInference(
                model_path='nonexistent_model.onnx',
                conf_threshold=0.5,
                iou_threshold=0.5,
            )
        assert 'nonexistent_model.onnx' in str(exc_info.value)

    def test_missing_model_error_message_contains_path(self):
        """FileNotFoundError 消息应包含模型路径"""
        path = 'path/to/missing/model.onnx'
        with pytest.raises(FileNotFoundError) as exc_info:
            RTMDetOBBInference(model_path=path)
        assert path in str(exc_info.value)
