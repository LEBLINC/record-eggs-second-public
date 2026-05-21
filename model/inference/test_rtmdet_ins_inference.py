# coding=utf-8
"""
RTMDet-Ins 推理模块单元测试
测试掩码质心提取、椭圆拟合、掩码缩放、后处理逻辑
@project: EGGRECORDQT
@file： test_rtmdet_ins_inference.py
"""
import math
import numpy as np
import pytest

from model.inference.rtmdet_ins_inference import (
    mask_to_center,
    mask_to_ellipse,
    resize_mask_to_original,
    RTMDetInsInference,
)


# ---------------------------------------------------------------------------
# mask_to_center 测试
# ---------------------------------------------------------------------------

class TestMaskToCenter:
    """测试掩码质心提取函数"""

    def test_empty_mask_returns_none(self):
        """全零掩码应返回 None"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = mask_to_center(mask)
        assert result is None

    def test_full_mask_center_is_image_center(self):
        """全填充掩码的质心应为图像中心"""
        mask = np.ones((100, 100), dtype=np.uint8)
        cx, cy = mask_to_center(mask)
        assert abs(cx - 49.5) < 0.5
        assert abs(cy - 49.5) < 0.5

    def test_single_pixel_center(self):
        """单像素掩码的质心应为该像素坐标"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30, 40] = 1  # row=30 → y=30, col=40 → x=40
        cx, cy = mask_to_center(mask)
        assert abs(cx - 40.0) < 1e-5
        assert abs(cy - 30.0) < 1e-5

    def test_rectangular_region_center(self):
        """矩形区域的质心应为矩形中心"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # 填充 [y=50:150, x=60:140]，中心应为 (99.5, 99.5)
        mask[50:150, 60:140] = 1
        cx, cy = mask_to_center(mask)
        assert abs(cx - 99.5) < 0.5
        assert abs(cy - 99.5) < 0.5

    def test_bool_mask_accepted(self):
        """bool 类型掩码应被正确处理"""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        result = mask_to_center(mask)
        assert result is not None
        cx, cy = result
        assert abs(cx - 14.5) < 0.5
        assert abs(cy - 14.5) < 0.5

    def test_returns_float_tuple(self):
        """返回值应为 float 元组"""
        mask = np.ones((50, 50), dtype=np.uint8)
        result = mask_to_center(mask)
        assert result is not None
        cx, cy = result
        assert isinstance(cx, float)
        assert isinstance(cy, float)


# ---------------------------------------------------------------------------
# mask_to_ellipse 测试
# ---------------------------------------------------------------------------

class TestMaskToEllipse:
    """测试掩码椭圆拟合函数"""

    def test_empty_mask_returns_none(self):
        """全零掩码应返回 None"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = mask_to_ellipse(mask)
        assert result is None

    def test_too_few_points_returns_none(self):
        """轮廓点数不足 5 时应返回 None"""
        mask = np.zeros((50, 50), dtype=np.uint8)
        # 只有 2 个像素，轮廓点不足 5
        mask[10, 10] = 1
        mask[10, 11] = 1
        result = mask_to_ellipse(mask)
        assert result is None

    def test_circle_ellipse_params(self):
        """圆形掩码的椭圆拟合：长轴≈短轴，中心接近圆心"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        import cv2 as _cv2
        _cv2.circle(mask, (100, 100), 40, 1, -1)

        result = mask_to_ellipse(mask)
        assert result is not None
        cx, cy, major, minor, angle = result
        assert abs(cx - 100) < 3.0
        assert abs(cy - 100) < 3.0
        # 圆形：长轴和短轴应接近（允许 10% 误差）
        assert abs(major - minor) / max(major, minor) < 0.15

    def test_ellipse_returns_five_elements(self):
        """返回值应为 5 元素元组"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        import cv2 as _cv2
        _cv2.ellipse(mask, (100, 100), (50, 25), 0, 0, 360, 1, -1)

        result = mask_to_ellipse(mask)
        assert result is not None
        assert len(result) == 5

    def test_major_axis_ge_minor_axis(self):
        """长轴应大于等于短轴"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        import cv2 as _cv2
        _cv2.ellipse(mask, (100, 100), (60, 20), 30, 0, 360, 1, -1)

        result = mask_to_ellipse(mask)
        assert result is not None
        cx, cy, major, minor, angle = result
        assert major >= minor

    def test_all_return_values_are_float(self):
        """所有返回值应为 float 类型"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        import cv2 as _cv2
        _cv2.circle(mask, (100, 100), 30, 1, -1)

        result = mask_to_ellipse(mask)
        assert result is not None
        for val in result:
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# resize_mask_to_original 测试
# ---------------------------------------------------------------------------

class TestResizeMaskToOriginal:
    """测试掩码缩放与放置函数"""

    def test_output_shape_matches_original(self):
        """输出掩码尺寸应与原始图像一致"""
        mask_crop = np.ones((28, 28), dtype=np.float32)
        bbox = (10.0, 20.0, 110.0, 120.0)
        result = resize_mask_to_original(mask_crop, bbox, orig_h=480, orig_w=640)
        assert result.shape == (480, 640)

    def test_output_dtype_is_uint8(self):
        """输出掩码应为 uint8 类型"""
        mask_crop = np.ones((28, 28), dtype=np.float32)
        bbox = (0.0, 0.0, 100.0, 100.0)
        result = resize_mask_to_original(mask_crop, bbox, orig_h=200, orig_w=200)
        assert result.dtype == np.uint8

    def test_full_mask_crop_fills_bbox_region(self):
        """全填充掩码裁剪区域应在 bbox 内全为 1"""
        mask_crop = np.ones((28, 28), dtype=np.float32)
        x1, y1, x2, y2 = 10, 20, 110, 120
        result = resize_mask_to_original(mask_crop, (x1, y1, x2, y2), orig_h=200, orig_w=200)
        # bbox 内区域应全为 1
        assert np.all(result[y1:y2, x1:x2] == 1)
        # bbox 外区域应全为 0
        assert result[0, 0] == 0
        assert result[199, 199] == 0

    def test_empty_mask_crop_produces_zero_mask(self):
        """全零掩码裁剪区域应产生全零输出"""
        mask_crop = np.zeros((28, 28), dtype=np.float32)
        result = resize_mask_to_original(mask_crop, (10.0, 10.0, 50.0, 50.0), orig_h=100, orig_w=100)
        assert np.all(result == 0)

    def test_bbox_clipped_to_image_bounds(self):
        """超出图像边界的 bbox 应被裁剪"""
        mask_crop = np.ones((28, 28), dtype=np.float32)
        # bbox 超出图像右下角
        result = resize_mask_to_original(mask_crop, (580.0, 460.0, 700.0, 540.0), orig_h=480, orig_w=640)
        assert result.shape == (480, 640)
        # 不应抛出异常，且输出有效

    def test_binary_values_only(self):
        """输出掩码只应包含 0 和 1"""
        mask_crop = np.random.rand(28, 28).astype(np.float32)
        result = resize_mask_to_original(mask_crop, (10.0, 10.0, 100.0, 100.0), orig_h=200, orig_w=200)
        unique_vals = np.unique(result)
        assert set(unique_vals).issubset({0, 1})


# ---------------------------------------------------------------------------
# RTMDetInsInference 后处理逻辑测试（不需要真实模型）
# ---------------------------------------------------------------------------

class TestRTMDetInsPostprocess:
    """测试 RTMDetInsInference 的后处理逻辑（绕过 ONNX 会话）"""

    def _make_inference_obj(self):
        """创建一个跳过模型加载的推理对象（用于测试后处理）"""
        obj = RTMDetInsInference.__new__(RTMDetInsInference)
        obj.conf_threshold = 0.5
        obj.target_size    = 640
        obj.use_fp16       = False
        obj.session        = None
        obj.input_name     = None
        obj.output_names   = []
        from model.inference.preprocessor import RTMDetPreprocessor
        obj.preprocessor   = RTMDetPreprocessor(target_size=640)
        return obj

    def _make_meta(self, orig_h=640, orig_w=640):
        """构造一个恒等变换的 meta（无缩放无 padding）"""
        return {
            'orig_shape': (orig_h, orig_w),
            'pad_top':    0,
            'pad_left':   0,
            'scale':      1.0,
        }

    def test_format_a_single_detection_keys(self):
        """格式 A（三输出）：单个检测框，结果字典应包含所有必需键"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets   = np.array([[[100, 100, 200, 200, 0.9]]], dtype=np.float32)  # (1,1,5)
        labels = np.array([[0]], dtype=np.int64)                             # (1,1)
        masks  = np.ones((1, 1, 28, 28), dtype=np.float32)                  # (1,1,28,28)

        results = obj._postprocess([dets, labels, masks], meta)

        assert len(results) == 1
        r = results[0]
        assert 'bbox' in r
        assert 'mask' in r
        assert 'center' in r
        assert 'ellipse_params' in r
        assert 'score' in r

    def test_format_a_score_value(self):
        """格式 A：score 值应正确传递"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets   = np.array([[[100, 100, 200, 200, 0.85]]], dtype=np.float32)
        labels = np.array([[0]], dtype=np.int64)
        masks  = np.ones((1, 1, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        assert len(results) == 1
        assert abs(results[0]['score'] - 0.85) < 1e-5

    def test_format_a_below_threshold_filtered(self):
        """格式 A：分数低于阈值的框应被过滤"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets   = np.array([[[100, 100, 200, 200, 0.3]]], dtype=np.float32)
        labels = np.array([[0]], dtype=np.int64)
        masks  = np.ones((1, 1, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        assert len(results) == 0

    def test_format_a_bbox_has_four_elements(self):
        """格式 A：bbox 应包含 4 个元素 [x1, y1, x2, y2]"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets   = np.array([[[50, 60, 150, 160, 0.9]]], dtype=np.float32)
        labels = np.array([[0]], dtype=np.int64)
        masks  = np.ones((1, 1, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        assert len(results[0]['bbox']) == 4

    def test_format_a_mask_shape_matches_original(self):
        """格式 A：mask 尺寸应与原始图像一致"""
        obj = self._make_inference_obj()
        meta = self._make_meta(orig_h=480, orig_w=640)

        dets   = np.array([[[50, 60, 150, 160, 0.9]]], dtype=np.float32)
        labels = np.array([[0]], dtype=np.int64)
        masks  = np.ones((1, 1, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        assert results[0]['mask'].shape == (480, 640)

    def test_format_a_center_is_tuple_of_floats(self):
        """格式 A：center 应为 float 元组"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets   = np.array([[[100, 100, 300, 300, 0.9]]], dtype=np.float32)
        labels = np.array([[0]], dtype=np.int64)
        masks  = np.ones((1, 1, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        cx, cy = results[0]['center']
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    def test_format_a_empty_output(self):
        """格式 A：空检测输出应返回空列表"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets   = np.zeros((1, 0, 5), dtype=np.float32)
        labels = np.zeros((1, 0), dtype=np.int64)
        masks  = np.zeros((1, 0, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        assert results == []

    def test_format_b_two_outputs(self):
        """格式 B（两输出）：dets + masks，应正确解析"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets  = np.array([[[100, 100, 200, 200, 0.9]]], dtype=np.float32)
        masks = np.ones((1, 1, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, masks], meta)
        assert len(results) == 1
        assert abs(results[0]['score'] - 0.9) < 1e-5

    def test_format_c_single_output(self):
        """格式 C（单输出）：合并格式，应正确解析"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        # (1, N, 5) 单输出
        raw = np.array([[[100, 100, 200, 200, 0.9]]], dtype=np.float32)

        results = obj._postprocess([raw], meta)
        assert len(results) == 1
        assert abs(results[0]['score'] - 0.9) < 1e-5

    def test_multiple_detections_all_above_threshold(self):
        """多个检测框均超过阈值时，应全部返回"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets = np.array([[
            [50,  50,  150, 150, 0.9],
            [200, 200, 350, 350, 0.8],
            [400, 400, 550, 550, 0.7],
        ]], dtype=np.float32)
        labels = np.zeros((1, 3), dtype=np.int64)
        masks  = np.ones((1, 3, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        assert len(results) == 3

    def test_coordinate_inverse_transform_applied(self):
        """坐标逆变换：有 padding 和缩放时，bbox 应正确映射回原始空间"""
        obj = self._make_inference_obj()
        # 原始图像 320x320，缩放到 640x640（scale=2.0），无 padding
        meta = {
            'orig_shape': (320, 320),
            'pad_top':    0,
            'pad_left':   0,
            'scale':      2.0,
        }
        # letterbox 空间 bbox [200, 200, 400, 400]，原始空间应为 [100, 100, 200, 200]
        dets   = np.array([[[200, 200, 400, 400, 0.9]]], dtype=np.float32)
        labels = np.array([[0]], dtype=np.int64)
        masks  = np.ones((1, 1, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        assert len(results) == 1
        x1, y1, x2, y2 = results[0]['bbox']
        assert abs(x1 - 100) < 1e-3
        assert abs(y1 - 100) < 1e-3
        assert abs(x2 - 200) < 1e-3
        assert abs(y2 - 200) < 1e-3

    def test_center_within_bbox(self):
        """质心坐标应在 bbox 范围内（对于全填充掩码）"""
        obj = self._make_inference_obj()
        meta = self._make_meta()

        dets   = np.array([[[100, 100, 300, 300, 0.9]]], dtype=np.float32)
        labels = np.array([[0]], dtype=np.int64)
        masks  = np.ones((1, 1, 28, 28), dtype=np.float32)

        results = obj._postprocess([dets, labels, masks], meta)
        cx, cy = results[0]['center']
        x1, y1, x2, y2 = results[0]['bbox']
        assert x1 <= cx <= x2
        assert y1 <= cy <= y2


# ---------------------------------------------------------------------------
# RTMDetInsInference 初始化测试（FileNotFoundError）
# ---------------------------------------------------------------------------

class TestRTMDetInsInferenceInit:
    """测试推理器初始化行为"""

    def test_missing_model_raises_file_not_found(self):
        """模型文件不存在时应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            RTMDetInsInference(
                model_path='nonexistent_ins_model.onnx',
                conf_threshold=0.5,
            )
        assert 'nonexistent_ins_model.onnx' in str(exc_info.value)

    def test_missing_model_error_message_contains_path(self):
        """FileNotFoundError 消息应包含模型路径"""
        path = 'path/to/missing/ins_model.onnx'
        with pytest.raises(FileNotFoundError) as exc_info:
            RTMDetInsInference(model_path=path)
        assert path in str(exc_info.value)
