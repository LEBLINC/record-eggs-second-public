# coding=utf-8
"""
RTMDetInferenceEngine 单元测试
测试引擎初始化、track() 接口、batch_track() 接口及错误处理
@project: EGGRECORDQT
@file： test_rtmdet_engine.py
"""
import os
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 辅助工具：创建最小合法 ONNX 模型文件（占位符）
# ---------------------------------------------------------------------------

def _make_dummy_onnx(path: str):
    """在指定路径创建一个空文件，模拟 ONNX 文件存在。"""
    with open(path, 'wb') as f:
        f.write(b'\x00' * 16)  # 最小占位内容


# ---------------------------------------------------------------------------
# 测试：FileNotFoundError 当模型文件不存在时
# ---------------------------------------------------------------------------

class TestRTMDetEngineFileNotFound:
    """验证当模型文件缺失时引擎抛出 FileNotFoundError。"""

    def test_obb_model_not_found_raises(self, tmp_path):
        """OBB 模型文件不存在时应抛出 FileNotFoundError。"""
        from model.inference.rtmdet_engine import RTMDetInferenceEngine

        cfg = {
            'rtmdet_obb_model': str(tmp_path / 'nonexistent_obb.onnx'),
            'rtmdet_ins_model': str(tmp_path / 'nonexistent_ins.onnx'),
        }
        with pytest.raises(FileNotFoundError) as exc_info:
            RTMDetInferenceEngine(cfg)

        assert 'nonexistent_obb.onnx' in str(exc_info.value)

    def test_ins_model_not_found_raises(self, tmp_path):
        """Ins 模型文件不存在时（OBB 存在）应抛出 FileNotFoundError。"""
        from model.inference.rtmdet_engine import RTMDetInferenceEngine

        obb_path = tmp_path / 'obb.onnx'
        _make_dummy_onnx(str(obb_path))

        cfg = {
            'rtmdet_obb_model': str(obb_path),
            'rtmdet_ins_model': str(tmp_path / 'nonexistent_ins.onnx'),
        }
        with pytest.raises(FileNotFoundError) as exc_info:
            RTMDetInferenceEngine(cfg)

        assert 'nonexistent_ins.onnx' in str(exc_info.value)

    def test_missing_obb_key_raises(self, tmp_path):
        """配置缺少 rtmdet_obb_model 键时应抛出 KeyError。"""
        from model.inference.rtmdet_engine import RTMDetInferenceEngine

        cfg = {
            'rtmdet_ins_model': str(tmp_path / 'ins.onnx'),
        }
        with pytest.raises(KeyError):
            RTMDetInferenceEngine(cfg)

    def test_missing_ins_key_raises(self, tmp_path):
        """配置缺少 rtmdet_ins_model 键时应抛出 KeyError。"""
        from model.inference.rtmdet_engine import RTMDetInferenceEngine

        cfg = {
            'rtmdet_obb_model': str(tmp_path / 'obb.onnx'),
        }
        with pytest.raises(KeyError):
            RTMDetInferenceEngine(cfg)


# ---------------------------------------------------------------------------
# 测试：track() 返回值结构
# ---------------------------------------------------------------------------

class TestRTMDetEngineTrackInterface:
    """验证 track() 方法返回正确的字典结构。"""

    @pytest.fixture
    def mock_engine(self, tmp_path):
        """
        创建一个带有 mock 子模型的 RTMDetInferenceEngine。
        通过 patch 绕过真实 ONNX 加载。
        """
        from model.inference.rtmdet_engine import RTMDetInferenceEngine

        obb_path = tmp_path / 'obb.onnx'
        ins_path = tmp_path / 'ins.onnx'
        _make_dummy_onnx(str(obb_path))
        _make_dummy_onnx(str(ins_path))

        cfg = {
            'rtmdet_obb_model': str(obb_path),
            'rtmdet_ins_model': str(ins_path),
            'imgsz': 640,
            'conf': 0.5,
            'iou': 0.5,
            'validity_threshold': 0.7,
            'use_fp16': False,
        }

        # Patch 两个子模型的初始化，避免真实 ONNX 加载
        with patch('model.inference.rtmdet_engine.RTMDetOBBInference') as MockOBB, \
             patch('model.inference.rtmdet_engine.RTMDetInsInference') as MockIns:

            mock_obb = MagicMock()
            mock_ins = MagicMock()
            MockOBB.return_value = mock_obb
            MockIns.return_value = mock_ins

            engine = RTMDetInferenceEngine(cfg)
            engine.obb_model = mock_obb
            engine.ins_model = mock_ins

        return engine

    def test_track_returns_dict(self, mock_engine):
        """track() 应返回 dict 类型。"""
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mock_engine.track(frame)

        assert isinstance(result, dict)

    def test_track_has_required_keys(self, mock_engine):
        """track() 返回的 dict 必须包含 qr_detections 和 egg_detections 键。"""
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mock_engine.track(frame)

        assert 'qr_detections' in result
        assert 'egg_detections' in result

    def test_track_qr_detections_is_list(self, mock_engine):
        """qr_detections 应为列表类型。"""
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mock_engine.track(frame)

        assert isinstance(result['qr_detections'], list)

    def test_track_egg_detections_is_list(self, mock_engine):
        """egg_detections 应为列表类型。"""
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mock_engine.track(frame)

        assert isinstance(result['egg_detections'], list)

    def test_track_passes_qr_detections_from_obb_model(self, mock_engine):
        """track() 应将 OBB 模型的输出放入 qr_detections。"""
        expected_qr = [
            {'rotated_box': [100, 100, 50, 50, 0.0], 'hbb': [75, 75, 125, 125],
             'score': 0.9, 'class_id': 0, 'validity_score': 1.0}
        ]
        mock_engine.obb_model.infer.return_value = expected_qr
        mock_engine.ins_model.infer.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mock_engine.track(frame)

        assert result['qr_detections'] == expected_qr

    def test_track_passes_egg_detections_from_ins_model(self, mock_engine):
        """track() 应将 Ins 模型的输出放入 egg_detections。"""
        expected_egg = [
            {'bbox': [10, 10, 60, 60], 'mask': np.zeros((480, 640), dtype=np.uint8),
             'center': (35.0, 35.0), 'ellipse_params': None, 'score': 0.85}
        ]
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = expected_egg

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mock_engine.track(frame)

        assert result['egg_detections'] == expected_egg

    def test_track_empty_frame_returns_empty_lists(self, mock_engine):
        """传入空帧（None）时应返回空列表，不抛出异常。"""
        result = mock_engine.track(None)

        assert result is not None
        assert result.get('qr_detections', []) == []
        assert result.get('egg_detections', []) == []

    def test_track_calls_both_models(self, mock_engine):
        """track() 应同时调用 OBB 和 Ins 两个模型。"""
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_engine.track(frame)

        mock_engine.obb_model.infer.assert_called_once()
        mock_engine.ins_model.infer.assert_called_once()


# ---------------------------------------------------------------------------
# 测试：batch_track() 接口
# ---------------------------------------------------------------------------

class TestRTMDetEngineBatchTrack:
    """验证 batch_track() 方法行为。"""

    @pytest.fixture
    def mock_engine(self, tmp_path):
        from model.inference.rtmdet_engine import RTMDetInferenceEngine

        obb_path = tmp_path / 'obb.onnx'
        ins_path = tmp_path / 'ins.onnx'
        _make_dummy_onnx(str(obb_path))
        _make_dummy_onnx(str(ins_path))

        cfg = {
            'rtmdet_obb_model': str(obb_path),
            'rtmdet_ins_model': str(ins_path),
        }

        with patch('model.inference.rtmdet_engine.RTMDetOBBInference') as MockOBB, \
             patch('model.inference.rtmdet_engine.RTMDetInsInference') as MockIns:

            mock_obb = MagicMock()
            mock_ins = MagicMock()
            MockOBB.return_value = mock_obb
            MockIns.return_value = mock_ins

            engine = RTMDetInferenceEngine(cfg)
            engine.obb_model = mock_obb
            engine.ins_model = mock_ins

        return engine

    def test_batch_track_empty_list_returns_empty(self, mock_engine):
        """传入空列表时应返回空列表。"""
        result = mock_engine.batch_track([])
        assert result == []

    def test_batch_track_returns_list(self, mock_engine):
        """batch_track() 应返回列表类型。"""
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = []

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        result = mock_engine.batch_track(frames)

        assert isinstance(result, list)

    def test_batch_track_length_matches_input(self, mock_engine):
        """batch_track() 返回列表长度应与输入帧数相同。"""
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = []

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        result = mock_engine.batch_track(frames)

        assert len(result) == 4

    def test_batch_track_each_result_has_required_keys(self, mock_engine):
        """batch_track() 每个结果都应包含 qr_detections 和 egg_detections 键。"""
        mock_engine.obb_model.infer.return_value = []
        mock_engine.ins_model.infer.return_value = []

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        results = mock_engine.batch_track(frames)

        for result in results:
            assert 'qr_detections' in result
            assert 'egg_detections' in result


# ---------------------------------------------------------------------------
# 测试：配置默认值
# ---------------------------------------------------------------------------

class TestRTMDetEngineConfigDefaults:
    """验证配置缺失时使用正确的默认值。"""

    @pytest.fixture
    def engine_with_minimal_cfg(self, tmp_path):
        from model.inference.rtmdet_engine import RTMDetInferenceEngine

        obb_path = tmp_path / 'obb.onnx'
        ins_path = tmp_path / 'ins.onnx'
        _make_dummy_onnx(str(obb_path))
        _make_dummy_onnx(str(ins_path))

        cfg = {
            'rtmdet_obb_model': str(obb_path),
            'rtmdet_ins_model': str(ins_path),
            # 不提供可选参数，测试默认值
        }

        with patch('model.inference.rtmdet_engine.RTMDetOBBInference') as MockOBB, \
             patch('model.inference.rtmdet_engine.RTMDetInsInference') as MockIns:

            MockOBB.return_value = MagicMock()
            MockIns.return_value = MagicMock()

            engine = RTMDetInferenceEngine(cfg)

        return engine

    def test_default_imgsz(self, engine_with_minimal_cfg):
        assert engine_with_minimal_cfg.imgsz == 640

    def test_default_conf(self, engine_with_minimal_cfg):
        assert engine_with_minimal_cfg.conf == 0.5

    def test_default_iou(self, engine_with_minimal_cfg):
        assert engine_with_minimal_cfg.iou == 0.5

    def test_default_validity_threshold(self, engine_with_minimal_cfg):
        assert engine_with_minimal_cfg.validity_threshold == 0.7

    def test_default_use_fp16(self, engine_with_minimal_cfg):
        assert engine_with_minimal_cfg.use_fp16 is False
