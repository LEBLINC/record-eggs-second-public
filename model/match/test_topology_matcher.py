# coding=utf-8
"""
Regression tests for the upgraded TopologyMatcher.

Run with:
    pytest model/match/test_topology_matcher.py -v

No real image data required – all inputs are synthetic.
"""
import math
import sys
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``model.*`` resolves.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.match.topology_matcher import TopologyMatcher   # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_matcher(**extra_cfg) -> TopologyMatcher:
    """Create a TopologyMatcher with sensible test defaults."""
    cfg = {
        'max_match_distance': 300,
        'validity_threshold': 0.5,
        'min_appear_frames': 1,       # confirm on first frame for simplicity
        'picture_recognition_path': os.path.join(
            os.path.dirname(__file__), '_test_frames'
        ),
        'cost_weights': {
            'position': 0.5,
            'validity': 0.25,
            'topology': 0.1,
            'pose':     0.15,
        },
    }
    cfg.update(extra_cfg)
    return TopologyMatcher(cfg)


def _qr(cage_id: str, cx: float, cy: float, validity: float = 0.9) -> dict:
    """Construct a minimal QR detection dict."""
    return {
        'cage_id':       cage_id,
        'center':        [cx, cy],
        'validity_score': validity,
    }


ROBOT_POSE_FORWARD = {
    'x':   0.0,
    'y':   0.0,
    'yaw': 0.0,   # facing +x direction
}


# ---------------------------------------------------------------------------
# Test 1 – no robot_pose → old behaviour unchanged
# ---------------------------------------------------------------------------

class TestMatchNoPose:
    def test_match_no_pose(self):
        """Calling match() without robot_pose must return (results, match_info)."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0), (200.0, 100.0)]
        qr_dets = [
            _qr('A01', 110.0, 95.0),
            _qr('A02', 205.0, 95.0),
        ]

        # Frame 1: cages transition EMPTY → OCCUPIED
        ret = matcher.match(egg_centers, qr_dets)
        assert isinstance(ret, tuple) and len(ret) == 2, (
            "match() must return (results, match_info)"
        )
        results1, match_info1 = ret
        assert isinstance(results1, list)
        assert isinstance(match_info1, dict)
        assert match_info1['pose_available'] is False
        assert match_info1['n_assignments'] == 2

        # Frame 2: appear_num >= min_appear_frames=1 → OCCUPIED → CONFIRMED
        results2, match_info2 = matcher.match(egg_centers, qr_dets)
        assert len(results2) == 2, (
            "Both cages must be confirmed after 2 frames with min_appear_frames=1"
        )

    def test_backward_compat(self):
        """Callers that unpack only the first return value must not raise."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0)]
        qr_dets     = [_qr('B01', 105.0, 98.0)]

        # Old-style call: caller only captures results
        ret = matcher.match(egg_centers, qr_dets)
        results = ret[0]   # or: results, _ = ret

        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test 2 – with robot_pose → confidence in [0,1] and match_info has fields
# ---------------------------------------------------------------------------

class TestMatchWithPose:
    def test_match_with_pose(self):
        """With a valid robot_pose, confidence values must be in [0, 1]."""
        matcher = _make_matcher()
        egg_centers = [(50.0, 50.0), (150.0, 50.0)]
        qr_dets = [
            _qr('C01', 60.0, 45.0),
            _qr('C02', 160.0, 45.0),
        ]

        ret = matcher.match(egg_centers, qr_dets, robot_pose=ROBOT_POSE_FORWARD)
        results, match_info = ret

        assert match_info['pose_available'] is True
        assert 'n_eggs' in match_info
        assert 'n_valid_qrs' in match_info
        assert 'n_assignments' in match_info
        assert 'assignment_confidences' in match_info
        assert 'mean_confidence' in match_info
        assert 'pose_available' in match_info
        assert 'frame_id' in match_info

        for conf in match_info['assignment_confidences']:
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0,1]"

        assert 0.0 <= match_info['mean_confidence'] <= 1.0

    def test_confidence_range(self):
        """All match_confidence values in returned results must be in [0,1]."""
        matcher = _make_matcher()
        egg_centers = [(10.0, 10.0), (50.0, 10.0), (90.0, 10.0)]
        qr_dets = [
            _qr('D01', 12.0, 8.0),
            _qr('D02', 52.0, 8.0),
            _qr('D03', 92.0, 8.0),
        ]
        pose = {'x': 0.0, 'y': 0.0, 'yaw': math.pi / 4}  # 45° heading
        results, _ = matcher.match(egg_centers, qr_dets, robot_pose=pose)

        for r in results:
            conf = r.get('match_confidence', -1.0)
            assert 0.0 <= conf <= 1.0, (
                f"match_confidence {conf} out of [0,1] in result {r}"
            )


# ---------------------------------------------------------------------------
# Test 3 – empty pose dict must not raise
# ---------------------------------------------------------------------------

class TestPoseNoneGraceful:
    def test_pose_empty_dict(self):
        """An empty pose dict must be handled gracefully (no exception)."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0)]
        qr_dets = [_qr('E01', 102.0, 98.0)]

        # Should not raise
        ret = matcher.match(egg_centers, qr_dets, robot_pose={})
        results, match_info = ret

        assert match_info['pose_available'] is False

    def test_pose_none(self):
        """robot_pose=None must work identically to omitting the argument."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0)]
        qr_dets = [_qr('F01', 102.0, 98.0)]

        ret = matcher.match(egg_centers, qr_dets, robot_pose=None)
        results, match_info = ret
        assert match_info['pose_available'] is False

    def test_pose_missing_yaw(self):
        """Pose dict without 'yaw' key must degrade gracefully."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0)]
        qr_dets = [_qr('G01', 102.0, 98.0)]

        ret = matcher.match(
            egg_centers, qr_dets,
            robot_pose={'x': 1.0, 'y': 2.0}   # no 'yaw'
        )
        results, match_info = ret
        assert match_info['pose_available'] is False


# ---------------------------------------------------------------------------
# Test 4 – single egg + single QR must always produce exactly one assignment
# ---------------------------------------------------------------------------

class TestSingleEggSingleQR:
    def test_single_egg_single_qr(self):
        """1 egg + 1 QR within range must produce 1 confirmed assignment."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0)]
        qr_dets = [_qr('H01', 110.0, 105.0)]

        # Frame 1: EMPTY → OCCUPIED
        results1, match_info = matcher.match(egg_centers, qr_dets)
        assert match_info['n_assignments'] == 1

        # Frame 2: OCCUPIED → CONFIRMED (min_appear_frames=1 means appear_num
        # reaches threshold on the 2nd frame: appear_num starts at 0, +1=1 in
        # frame 1 → check >=1 → CONFIRMED on frame 2 entry when already OCCUPIED)
        results2, _ = matcher.match(egg_centers, qr_dets)
        assert len(results2) == 1
        assert results2[0]['cage_id'] == 'H01'

    def test_single_egg_out_of_range(self):
        """1 egg far from the only QR must produce 0 assignments."""
        matcher = _make_matcher()
        egg_centers = [(0.0, 0.0)]
        qr_dets = [_qr('I01', 999.0, 999.0)]   # > max_match_distance

        results, match_info = matcher.match(egg_centers, qr_dets)

        assert match_info['n_assignments'] == 0
        assert results == []


# ---------------------------------------------------------------------------
# Test 5 – cost matrix shape
# ---------------------------------------------------------------------------

class TestCostMatrixShape:
    def test_cost_matrix_shape(self):
        """Internal cost matrix must have shape (n_eggs, n_valid_qrs)."""
        matcher = _make_matcher()
        egg_centers = [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0)]
        valid_qrs = [
            {'cage_id': 'J01', 'center': [12.0, 8.0], 'validity_score': 0.9},
            {'cage_id': 'J02', 'center': [22.0, 8.0], 'validity_score': 0.9},
        ]

        cost_matrix, _ = matcher._build_cost_matrix(
            egg_centers, valid_qrs, robot_pose=None
        )
        assert cost_matrix.shape == (3, 2), (
            f"Expected (3, 2), got {cost_matrix.shape}"
        )

    def test_cost_matrix_shape_with_pose(self):
        """Cost matrix shape must be correct even when pose is provided."""
        matcher = _make_matcher()
        egg_centers = [(10.0, 10.0), (30.0, 10.0)]
        valid_qrs = [
            {'cage_id': 'K01', 'center': [12.0, 8.0], 'validity_score': 0.9},
            {'cage_id': 'K02', 'center': [32.0, 8.0], 'validity_score': 0.9},
            {'cage_id': 'K03', 'center': [52.0, 8.0], 'validity_score': 0.9},
        ]
        pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        cost_matrix, _ = matcher._build_cost_matrix(
            egg_centers, valid_qrs, robot_pose=pose
        )
        assert cost_matrix.shape == (2, 3), (
            f"Expected (2, 3), got {cost_matrix.shape}"
        )

    def test_pose_cost_matrix_shape(self):
        """_compute_pose_prior_cost must return shape (n_eggs, n_qrs)."""
        matcher = _make_matcher()
        egg_centers = [(0.0, 0.0), (10.0, 0.0)]
        qr_centers  = [(5.0, 5.0), (15.0, 5.0), (25.0, 5.0)]
        pose = {'yaw': 0.0}

        pc = matcher._compute_pose_prior_cost(egg_centers, qr_centers, pose)
        assert pc.shape == (2, 3), f"Expected (2, 3), got {pc.shape}"

    def test_pose_cost_values_range(self):
        """All values in pose prior cost matrix must be in [0, 1]."""
        matcher = _make_matcher()
        rng = np.random.default_rng(42)
        egg_centers = [(float(x), float(y))
                       for x, y in rng.uniform(0, 100, (5, 2))]
        qr_centers  = [(float(x), float(y))
                       for x, y in rng.uniform(0, 100, (4, 2))]
        for yaw in [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3]:
            pc = matcher._compute_pose_prior_cost(
                egg_centers, qr_centers, {'yaw': yaw}
            )
            assert pc.min() >= -1e-6, "Pose cost below 0"
            assert pc.max() <= 1.0 + 1e-6, "Pose cost above 1"


# ---------------------------------------------------------------------------
# Test 6 – backward compatibility: old single-value capture does not raise
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_unpack_first_value_only(self):
        """Old callers that do ``results = matcher.match(...)`` must not error."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0)]
        qr_dets = [_qr('L01', 105.0, 98.0)]

        # Simulate old calling convention: only use first element
        ret = matcher.match(egg_centers, qr_dets)
        results = ret[0]   # old style: first element
        assert isinstance(results, list)

    def test_no_robot_pose_arg_needed(self):
        """Calling match() with only required args must not raise."""
        matcher = _make_matcher()
        egg_centers = [(50.0, 50.0)]
        qr_dets = [_qr('M01', 55.0, 48.0)]

        ret = matcher.match(egg_centers, qr_dets)
        assert ret is not None

    def test_pose_used_field_present(self):
        """result dicts must always contain 'match_confidence' and 'pose_used'."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0)]
        qr_dets = [_qr('N01', 105.0, 98.0)]

        # Frame 1: EMPTY → OCCUPIED
        matcher.match(egg_centers, qr_dets, robot_pose=ROBOT_POSE_FORWARD)
        # Frame 2: OCCUPIED → CONFIRMED
        results, _ = matcher.match(
            egg_centers, qr_dets, robot_pose=ROBOT_POSE_FORWARD
        )
        assert len(results) == 1
        assert 'match_confidence' in results[0]
        assert 'pose_used' in results[0]
        assert results[0]['pose_used'] is True


# ---------------------------------------------------------------------------
# Test 7 – get_match_statistics structure
# ---------------------------------------------------------------------------

class TestMatchStatistics:
    def test_statistics_keys(self):
        """get_match_statistics must return all required keys."""
        matcher = _make_matcher()
        stats = matcher.get_match_statistics()

        assert 'total_frames' in stats
        assert 'total_confirmed' in stats
        assert 'avg_confidence' in stats
        assert 'pose_utilization_rate' in stats

    def test_statistics_pose_utilization(self):
        """pose_utilization_rate must equal frames-with-pose / total-frames."""
        matcher = _make_matcher()
        egg_centers = [(100.0, 100.0)]
        qr_dets = [_qr('O01', 105.0, 98.0)]

        matcher.match(egg_centers, qr_dets)                              # frame 1 – no pose
        matcher.match(egg_centers, qr_dets, robot_pose=ROBOT_POSE_FORWARD)  # frame 2 – pose
        matcher.match(egg_centers, qr_dets, robot_pose=ROBOT_POSE_FORWARD)  # frame 3 – pose

        stats = matcher.get_match_statistics()
        assert stats['total_frames'] == 3
        assert abs(stats['pose_utilization_rate'] - 2 / 3) < 1e-6

    def test_statistics_reset(self):
        """reset() must clear statistics counters."""
        matcher = _make_matcher()
        matcher.match([(100.0, 100.0)], [_qr('P01', 105.0, 98.0)])
        matcher.reset()

        stats = matcher.get_match_statistics()
        assert stats['total_frames'] == 0
        assert stats['total_confirmed'] == 0
        assert stats['avg_confidence'] == 0.0
        assert stats['pose_utilization_rate'] == 0.0


# ---------------------------------------------------------------------------
# Test 8 – empty inputs must not raise
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_no_eggs(self):
        matcher = _make_matcher()
        ret = matcher.match([], [_qr('Q01', 50.0, 50.0)])
        results, match_info = ret
        assert results == []
        assert match_info['n_eggs'] == 0

    def test_no_qrs(self):
        matcher = _make_matcher()
        ret = matcher.match([(50.0, 50.0)], [])
        results, match_info = ret
        assert results == []
        assert match_info['n_valid_qrs'] == 0

    def test_both_empty(self):
        matcher = _make_matcher()
        ret = matcher.match([], [])
        results, match_info = ret
        assert results == []
