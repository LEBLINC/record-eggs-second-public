# coding=utf-8
"""
ResultAdapter 单元测试
验证 RTMDet 结果适配器与 unpack_results() 的兼容性
@project: EGGRECORDQT
@file： test_result_adapter.py
"""
import numpy as np
import pytest

from model.inference.result_adapter import ResultAdapter
from model.track.matchUtils import unpack_results


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

def _make_detections(n_qr: int = 2, n_egg: int = 3) -> dict:
    """构造模拟的 RTMDet 引擎输出字典。"""
    qr_dets = []
    for i in range(n_qr):
        qr_dets.append({
            'rotated_box': [100.0 + i * 50, 100.0, 40.0, 40.0, 0.0],
            'hbb':         [80.0 + i * 50, 80.0, 120.0 + i * 50, 120.0],
            'score':       0.9 - i * 0.05,
            'class_id':    0,          # valid_qr
            'validity_score': 1.0,
        })

    egg_dets = []
    for i in range(n_egg):
        egg_dets.append({
            'bbox':          [200.0 + i * 60, 200.0, 250.0 + i * 60, 250.0],
            'mask':          np.zeros((480, 640), dtype=np.uint8),
            'center':        (225.0 + i * 60, 225.0),
            'ellipse_params': None,
            'score':         0.85 - i * 0.05,
        })

    return {'qr_detections': qr_dets, 'egg_detections': egg_dets}


# ---------------------------------------------------------------------------
# to_tracker_format() 测试
# ---------------------------------------------------------------------------

class TestToTrackerFormat:

    def test_output_shape(self):
        """输出形状应为 (N, 6)，N = QR 数 + egg 数。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        result = ResultAdapter.to_tracker_format(dets)
        assert result.shape == (5, 6), f"期望 (5, 6)，实际 {result.shape}"

    def test_output_dtype(self):
        """输出应为 float32。"""
        dets = _make_detections(n_qr=1, n_egg=1)
        result = ResultAdapter.to_tracker_format(dets)
        assert result.dtype == np.float32

    def test_empty_detections(self):
        """无检测结果时应返回 shape (0, 6) 的空数组。"""
        dets = {'qr_detections': [], 'egg_detections': []}
        result = ResultAdapter.to_tracker_format(dets)
        assert result.shape == (0, 6)

    def test_class_ids_correct(self):
        """QR 码 class_id 应为 1，鸡蛋 class_id 应为 0。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        result = ResultAdapter.to_tracker_format(dets)
        # 前 2 行是 QR（class=1），后 3 行是 egg（class=0）
        assert np.all(result[:2, 5] == 1.0), "QR class_id 应为 1"
        assert np.all(result[2:, 5] == 0.0), "egg class_id 应为 0"

    def test_coordinates_preserved(self):
        """坐标值应与输入一致。"""
        dets = _make_detections(n_qr=1, n_egg=0)
        result = ResultAdapter.to_tracker_format(dets)
        expected_hbb = dets['qr_detections'][0]['hbb']
        np.testing.assert_allclose(result[0, :4], expected_hbb, rtol=1e-5)

    def test_only_qr(self):
        """仅有 QR 检测时输出形状正确。"""
        dets = _make_detections(n_qr=3, n_egg=0)
        result = ResultAdapter.to_tracker_format(dets)
        assert result.shape == (3, 6)
        assert np.all(result[:, 5] == 1.0)

    def test_only_egg(self):
        """仅有 egg 检测时输出形状正确。"""
        dets = _make_detections(n_qr=0, n_egg=4)
        result = ResultAdapter.to_tracker_format(dets)
        assert result.shape == (4, 6)
        assert np.all(result[:, 5] == 0.0)


# ---------------------------------------------------------------------------
# to_legacy_results() 测试
# ---------------------------------------------------------------------------

class TestToLegacyResults:

    def test_returns_list_of_length_one(self):
        """应返回长度为 1 的列表，模拟 Ultralytics results 格式。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        legacy = ResultAdapter.to_legacy_results(dets)
        assert isinstance(legacy, list)
        assert len(legacy) == 1

    def test_names_dict(self):
        """results[0].names 应包含 egg 和 qr 映射。"""
        dets = _make_detections(n_qr=1, n_egg=1)
        legacy = ResultAdapter.to_legacy_results(dets)
        names = legacy[0].names
        assert names[0] == 'egg'
        assert names[1] == 'qr'

    def test_boxes_xyxy_shape(self):
        """boxes.xyxy.cpu().numpy() 应为 shape (N, 4)。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        legacy = ResultAdapter.to_legacy_results(dets)
        xyxy = legacy[0].boxes.xyxy.cpu().numpy()
        assert xyxy.shape == (5, 4), f"期望 (5, 4)，实际 {xyxy.shape}"

    def test_boxes_cls_shape(self):
        """boxes.cls.cpu().numpy() 应为 shape (N,)。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        legacy = ResultAdapter.to_legacy_results(dets)
        cls = legacy[0].boxes.cls.cpu().numpy()
        assert cls.shape == (5,)

    def test_boxes_id_shape(self):
        """boxes.id.cpu().numpy() 应为 shape (N,)。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        legacy = ResultAdapter.to_legacy_results(dets)
        ids = legacy[0].boxes.id.cpu().numpy()
        assert ids.shape == (5,)

    def test_boxes_conf_shape(self):
        """boxes.conf.cpu().numpy() 应为 shape (N,)。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        legacy = ResultAdapter.to_legacy_results(dets)
        conf = legacy[0].boxes.conf.cpu().numpy()
        assert conf.shape == (5,)

    def test_with_track_ids(self):
        """提供 track_ids 时应正确赋值。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        track_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        legacy = ResultAdapter.to_legacy_results(dets, track_ids=track_ids)
        ids = legacy[0].boxes.id.cpu().numpy().astype(int)
        np.testing.assert_array_equal(ids, track_ids)

    def test_auto_track_ids(self):
        """未提供 track_ids 时应自动分配从 1 开始的连续 ID。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        legacy = ResultAdapter.to_legacy_results(dets)
        ids = legacy[0].boxes.id.cpu().numpy().astype(int)
        np.testing.assert_array_equal(ids, [1, 2, 3, 4, 5])

    def test_empty_detections(self):
        """无检测结果时各数组应为空。"""
        dets = {'qr_detections': [], 'egg_detections': []}
        legacy = ResultAdapter.to_legacy_results(dets)
        assert legacy[0].boxes.xyxy.cpu().numpy().shape == (0, 4)
        assert legacy[0].boxes.cls.cpu().numpy().shape  == (0,)
        assert legacy[0].boxes.id.cpu().numpy().shape   == (0,)
        assert legacy[0].boxes.conf.cpu().numpy().shape == (0,)


# ---------------------------------------------------------------------------
# unpack_results() 兼容性测试（端到端）
# ---------------------------------------------------------------------------

class TestUnpackResultsCompatibility:

    def test_unpack_results_runs_without_error(self):
        """to_legacy_results() 输出应能被 unpack_results() 正常解析。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        track_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        legacy = ResultAdapter.to_legacy_results(dets, track_ids=track_ids)

        names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs = \
            unpack_results(legacy)

        assert names == {0: 'egg', 1: 'qr'}

    def test_qr_egg_separation(self):
        """unpack_results() 应正确分离 QR 和 egg 检测。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        track_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        legacy = ResultAdapter.to_legacy_results(dets, track_ids=track_ids)

        names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs = \
            unpack_results(legacy)

        assert len(qr_boxes) == 2,  f"期望 2 个 QR 框，实际 {len(qr_boxes)}"
        assert len(egg_boxes) == 3, f"期望 3 个 egg 框，实际 {len(egg_boxes)}"
        assert len(qr_track_ids) == 2
        assert len(egg_track_ids) == 3
        assert len(qr_confs) == 2
        assert len(egg_confs) == 3

    def test_only_qr_detections(self):
        """仅有 QR 检测时 egg 列表应为空。"""
        dets = _make_detections(n_qr=2, n_egg=0)
        legacy = ResultAdapter.to_legacy_results(dets)

        names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs = \
            unpack_results(legacy)

        assert len(qr_boxes) == 2
        assert len(egg_boxes) == 0

    def test_only_egg_detections(self):
        """仅有 egg 检测时 QR 列表应为空。"""
        dets = _make_detections(n_qr=0, n_egg=3)
        legacy = ResultAdapter.to_legacy_results(dets)

        names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs = \
            unpack_results(legacy)

        assert len(qr_boxes) == 0
        assert len(egg_boxes) == 3

    def test_track_ids_preserved(self):
        """track_ids 应在 unpack_results() 后正确保留（顺序可能因排序而变化）。"""
        dets = _make_detections(n_qr=1, n_egg=1)
        # QR track_id=10, egg track_id=20（顺序：先 QR 后 egg）
        track_ids = np.array([10, 20], dtype=np.int64)
        legacy = ResultAdapter.to_legacy_results(dets, track_ids=track_ids)

        names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs = \
            unpack_results(legacy)

        assert 10 in qr_track_ids,  "QR track_id=10 应在 qr_track_ids 中"
        assert 20 in egg_track_ids, "egg track_id=20 应在 egg_track_ids 中"

    def test_confidence_values_preserved(self):
        """置信度值应在 unpack_results() 后正确保留。"""
        dets = _make_detections(n_qr=1, n_egg=1)
        legacy = ResultAdapter.to_legacy_results(dets)

        names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs = \
            unpack_results(legacy)

        # QR 置信度应约为 0.9（第一个 QR 检测）
        assert len(qr_confs) == 1
        assert abs(float(qr_confs[0]) - 0.9) < 0.01, f"QR 置信度期望约 0.9，实际 {qr_confs[0]}"

        # egg 置信度应约为 0.85（第一个 egg 检测）
        assert len(egg_confs) == 1
        assert abs(float(egg_confs[0]) - 0.85) < 0.01, f"egg 置信度期望约 0.85，实际 {egg_confs[0]}"


# ---------------------------------------------------------------------------
# 辅助方法测试
# ---------------------------------------------------------------------------

class TestHelperMethods:

    def test_get_qr_detections(self):
        """get_qr_detections() 应返回 QR 检测列表。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        qr = ResultAdapter.get_qr_detections(dets)
        assert len(qr) == 2
        assert all('hbb' in d for d in qr)

    def test_get_egg_detections(self):
        """get_egg_detections() 应返回 egg 检测列表。"""
        dets = _make_detections(n_qr=2, n_egg=3)
        eggs = ResultAdapter.get_egg_detections(dets)
        assert len(eggs) == 3
        assert all('bbox' in d for d in eggs)

    def test_assign_track_ids_from_tracker_output(self):
        """assign_track_ids() 应从跟踪器输出中提取第 5 列作为 track_id。"""
        # 模拟 OCSORT 输出：[x1, y1, x2, y2, track_id]
        tracker_output = np.array([
            [10.0, 20.0, 50.0, 60.0, 101.0],
            [70.0, 80.0, 110.0, 120.0, 202.0],
        ], dtype=np.float32)

        ids = ResultAdapter.assign_track_ids(tracker_output)
        np.testing.assert_array_equal(ids, [101, 202])

    def test_assign_track_ids_empty(self):
        """空输入时应返回空数组。"""
        ids = ResultAdapter.assign_track_ids(np.empty((0, 5), dtype=np.float32))
        assert len(ids) == 0

    def test_assign_track_ids_none(self):
        """None 输入时应返回空数组。"""
        ids = ResultAdapter.assign_track_ids(None)
        assert len(ids) == 0
