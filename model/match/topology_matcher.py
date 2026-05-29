# coding=utf-8
"""
Navigation-aware topology-constrained egg-to-cage matching module.

Implements the first innovation of:
  "A navigation-aware visual matching system for cage-level egg inspection
   in stacked poultry houses"

Extends the Hungarian-algorithm-based egg–QR matching with:
  - Pose prior cost (C_pose): robot heading vs. egg→QR direction alignment
  - Per-assignment confidence scores stored in cage states and results
  - match_info return value for paper experiment evaluation (E3)
  - Fully backward-compatible interface

@project: EGGRECORDQT
@Author：lzy
@file： topology_matcher.py
"""
import os
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from model.utils.exception import exception_handler


# Cage temporal-state enumeration
STATE_EMPTY     = 'empty'       # no egg detected
STATE_OCCUPIED  = 'occupied'    # egg detected but not yet confirmed
STATE_UNCERTAIN = 'uncertain'   # unstable detection
STATE_CONFIRMED = 'confirmed'   # confirmed (appear_num >= min_appear_frames)


class TopologyMatcher:
    """
    Navigation-aware topology-constrained egg-to-cage matcher.

    Uses the Hungarian algorithm to optimally assign egg centres to valid
    QR-code cage positions.  The cost matrix combines four terms:

        C = w_pos * C_geo + w_val * C_validity
          + w_topo * C_topology + w_pose * C_pose

    C_pose (new) penalises assignments whose egg→QR direction vector is
    inconsistent with the robot's current heading, implementing the
    "navigation-aware" component described in the paper.

    Maintains a per-cage temporal state machine (empty / occupied /
    uncertain / confirmed) and emits confirmed results together with
    per-assignment confidence scores for paper experiment E3.

    Supported config keys
    ----------------------
    max_match_distance  : float  – max allowable matching distance (default 200)
    validity_threshold  : float  – min QR validity score (default 0.7)
    min_appear_frames   : int    – frames needed to confirm a cage (default 5)
    cost_weights        : dict   – sub-keys: position, validity, topology, pose
    picture_recognition_path : str – directory for saving recognition frames
    """

    def __init__(self, cfg: dict):
        """
        Initialise the topology matcher.

        Args:
            cfg: Configuration dictionary; see class docstring for keys.
        """
        self.max_match_distance = cfg.get('max_match_distance', 200)
        self.validity_threshold = cfg.get('validity_threshold', 0.7)
        self.min_appear_frames  = cfg.get('min_appear_frames', 5)

        # Cost weights – pose weight added in this version
        _weights = cfg.get('cost_weights', {})
        self.w_position = _weights.get('position', 0.5)   # reduced to make room for pose
        self.w_validity = _weights.get('validity', 0.25)
        self.w_topology = _weights.get('topology', 0.1)
        self.w_pose     = _weights.get('pose',     0.15)  # NEW: navigation-aware weight

        # Frame save path
        self.picture_recognition_path = cfg.get(
            'picture_recognition_path', 'recognition_output'
        )
        if not os.path.exists(self.picture_recognition_path):
            os.makedirs(self.picture_recognition_path)

        # Cage state dictionary: cage_id -> state dict
        self._cage_states: Dict[str, dict] = {}

        # Frame counter (used for timeout cleanup)
        self._frame_count: int = 0

        # Statistics for paper experiment evaluation
        self._total_frames: int = 0
        self._total_confirmed: int = 0
        self._confidence_history: List[float] = []
        self._pose_used_frames: int = 0     # frames where robot_pose was provided

        # Warn about missing optional config keys
        _optional_defaults = {
            'max_match_distance': 200,
            'validity_threshold': 0.7,
            'min_appear_frames': 5,
        }
        for key, default in _optional_defaults.items():
            if key not in cfg:
                print(
                    f"TopologyMatcher: config key '{key}' not found, "
                    f"using default {default}"
                )

    # ------------------------------------------------------------------
    # Primary matching interface
    # ------------------------------------------------------------------

    @exception_handler
    def match(
        self,
        egg_centers: List[Tuple[float, float]],
        qr_detections: List[Dict],
        frame: Optional[np.ndarray] = None,
        egg_meta: Optional[List[Dict]] = None,
        robot_pose: Optional[Dict] = None,  # NEW: navigation-aware pose prior
    ) -> Tuple[List[Dict], Dict]:
        """
        Execute egg-to-cage matching with optional navigation-aware pose prior.

        Backward compatible: callers that only capture the first return value
        (``results, _ = match(...)`` or ``results = match(...)``) continue to
        work without modification.

        Args:
            egg_centers:   List of egg centre coordinates [(x, y), ...].
            qr_detections: List of QR detection dicts.
            frame:         Current BGR frame image (optional).
            egg_meta:      Per-egg metadata list aligned with egg_centers.
                           Each dict may contain:
                             - class_id   (0=egg, 1=invalidegg)
                             - is_invalid (bool)
                             - score      (float confidence)
                           Defaults to all-valid if omitted.
            robot_pose:    Robot world-frame pose dict (optional).
                           Expected keys:
                             - x          (float, metres)
                             - y          (float, metres)
                             - yaw        (float, radians)
                             - timestamp  (float, Unix epoch, optional)
                             - linear_vel (float, m/s, optional)
                           When None or missing required keys, C_pose = 0
                           and w_pose is effectively zero.

        Returns:
            Tuple ``(results, match_info)`` where:

            results   – list of confirmed cage match dicts (same as before,
                        now augmented with 'match_confidence' and 'pose_used').

            match_info – dict for paper experiment evaluation::

                {
                    'n_eggs':               int,
                    'n_valid_qrs':          int,
                    'n_assignments':        int,
                    'assignment_confidences': List[float],
                    'mean_confidence':      float,
                    'pose_available':       bool,
                    'frame_id':             int,
                }
        """
        self._frame_count += 1
        self._total_frames += 1

        # Determine whether pose information is usable
        pose_available = self._is_pose_valid(robot_pose)
        if pose_available:
            self._pose_used_frames += 1

        # Cache egg metadata
        if egg_meta is not None and len(egg_meta) == len(egg_centers):
            self._current_egg_meta = list(egg_meta)
        else:
            self._current_egg_meta = [
                {'class_id': 0, 'is_invalid': False, 'score': 1.0}
                for _ in egg_centers
            ]

        # Step 1: filter valid QRs (validity_score >= validity_threshold)
        valid_qrs = self._filter_valid_qrs(qr_detections)

        # Build default match_info for early-exit paths
        match_info: Dict = {
            'n_eggs':                 len(egg_centers),
            'n_valid_qrs':            len(valid_qrs),
            'n_assignments':          0,
            'assignment_confidences': [],
            'mean_confidence':        0.0,
            'pose_available':         pose_available,
            'frame_id':               self._frame_count,
        }

        if len(egg_centers) == 0 or len(valid_qrs) == 0:
            self._update_states_no_eggs(valid_qrs)
            results = self._collect_confirmed_results()
            return results, match_info

        # Step 2: build cost matrix (rows=eggs, cols=valid QRs)
        cost_matrix, raw_costs = self._build_cost_matrix(
            egg_centers, valid_qrs, robot_pose
        )

        if np.all(np.isinf(cost_matrix)):
            self._update_states_no_eggs(valid_qrs)
            results = self._collect_confirmed_results()
            return results, match_info

        # Step 3: Hungarian assignment
        assignments = self._solve_assignment(cost_matrix)

        # Step 4: compute per-assignment confidence scores
        total_weight = self.w_position + self.w_validity + self.w_topology
        if pose_available:
            total_weight += self.w_pose

        assignment_confidences: List[float] = []
        for egg_idx, qr_idx in assignments:
            raw_cost = cost_matrix[egg_idx, qr_idx]
            if np.isfinite(raw_cost) and total_weight > 0.0:
                conf = 1.0 - raw_cost / total_weight
                conf = float(np.clip(conf, 0.0, 1.0))
            else:
                conf = 0.0
            assignment_confidences.append(conf)

        mean_conf = float(np.mean(assignment_confidences)) if assignment_confidences else 0.0
        self._confidence_history.extend(assignment_confidences)

        # Update match_info
        match_info['n_assignments']          = len(assignments)
        match_info['assignment_confidences'] = assignment_confidences
        match_info['mean_confidence']        = mean_conf

        # Step 5: update cage temporal states (pass confidences for storage)
        self._update_cage_states(
            assignments, valid_qrs, egg_centers, frame,
            confidences=assignment_confidences,
            pose_used=pose_available,
        )

        # Step 6: collect confirmed results
        results = self._collect_confirmed_results()

        return results, match_info

    # ------------------------------------------------------------------
    # Step 1: filter valid QRs
    # ------------------------------------------------------------------

    def _filter_valid_qrs(self, qr_detections: List[Dict]) -> List[Dict]:
        """
        Keep only QR detections with validity_score >= validity_threshold.

        Args:
            qr_detections: Raw QR detection list.

        Returns:
            Filtered list (original dict references preserved).
        """
        valid = []
        for qr in qr_detections:
            score = qr.get('validity_score', 0.0)
            if score >= self.validity_threshold:
                valid.append(qr)
        return valid

    # ------------------------------------------------------------------
    # Step 2: build cost matrix
    # ------------------------------------------------------------------

    def _build_cost_matrix(
        self,
        egg_centers: List[Tuple[float, float]],
        valid_qrs: List[Dict],
        robot_pose: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the egg-to-QR assignment cost matrix.

        Cost formula::

            C(egg_j, qr_k) = w_pos  * C_geo(j,k)
                           + w_val  * C_validity(k)
                           + w_topo * C_topology(j,k)
                           + w_pose * C_pose(j,k)   # 0 when pose unavailable

        All sub-costs are in [0, 1].  Pairs beyond max_match_distance
        receive cost = inf so the Hungarian solver never selects them.

        Args:
            egg_centers: Egg centre coordinates.
            valid_qrs:   Valid QR detections.
            robot_pose:  Robot pose dict (may be None).

        Returns:
            Tuple ``(cost_matrix, raw_cost_matrix)`` both shape
            ``(n_eggs, n_qrs)``.  raw_cost_matrix contains finite values
            before the inf mask is applied (useful for debugging).
        """
        n_eggs = len(egg_centers)
        n_qrs  = len(valid_qrs)

        qr_centers    = [self._get_qr_center(qr) for qr in valid_qrs]
        qr_validities = [qr.get('validity_score', 1.0) for qr in valid_qrs]

        egg_arr = np.array(egg_centers, dtype=np.float64)  # (n_eggs, 2)
        qr_arr  = np.array(qr_centers,  dtype=np.float64)  # (n_qrs,  2)

        # Euclidean distance matrix – fully vectorised
        diff        = egg_arr[:, np.newaxis, :] - qr_arr[np.newaxis, :, :]  # (n_eggs, n_qrs, 2)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))                       # (n_eggs, n_qrs)

        out_of_range = dist_matrix > self.max_match_distance
        norm_dist    = np.clip(dist_matrix / self.max_match_distance, 0.0, 1.0)

        # Validity penalty (broadcast over rows)
        validity_arr     = np.array(qr_validities, dtype=np.float64)  # (n_qrs,)
        validity_penalty = 1.0 - validity_arr                          # (n_qrs,)

        # Topology penalty
        topology_penalty = self._compute_topology_penalty(egg_centers, qr_centers)

        # Pose prior cost (zero matrix when pose is unavailable)
        pose_available = self._is_pose_valid(robot_pose)
        if pose_available:
            pose_cost   = self._compute_pose_prior_cost(egg_centers, qr_centers, robot_pose)
            w_pose_eff  = self.w_pose
        else:
            pose_cost   = np.zeros((n_eggs, n_qrs), dtype=np.float64)
            w_pose_eff  = 0.0

        raw_cost = (
            self.w_position * norm_dist
            + self.w_validity * validity_penalty[np.newaxis, :]
            + self.w_topology * topology_penalty
            + w_pose_eff     * pose_cost
        )

        cost_matrix = raw_cost.copy()
        cost_matrix[out_of_range] = np.inf

        return cost_matrix, raw_cost

    def _get_qr_center(self, qr: Dict) -> Tuple[float, float]:
        """
        Extract centre coordinates from a QR detection dict.

        Supports multiple key conventions: 'center', 'hbb'/'box',
        'rotated_box'.

        Args:
            qr: QR detection dict.

        Returns:
            ``(cx, cy)`` centre coordinates.
        """
        if 'center' in qr:
            return tuple(qr['center'][:2])

        box = qr.get('hbb') or qr.get('box')
        if box is not None and len(box) >= 4:
            return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

        rotated = qr.get('rotated_box')
        if rotated is not None and len(rotated) >= 2:
            return (float(rotated[0]), float(rotated[1]))

        print("TopologyMatcher: cannot extract QR centre, falling back to (0, 0)")
        return (0.0, 0.0)

    def _compute_topology_penalty(
        self,
        egg_centers: List[Tuple[float, float]],
        qr_centers: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Compute topology penalty matrix enforcing horizontal ordering consistency.

        Intuition: if egg_i is to the left of egg_j, its matched QR should
        also be to the left of the QR matched to egg_j.  The penalty is
        the absolute normalised x-rank difference.

        Args:
            egg_centers: Egg centre coordinate list.
            qr_centers:  QR centre coordinate list.

        Returns:
            Penalty matrix of shape ``(n_eggs, n_qrs)`` with values in [0, 1].
        """
        n_eggs = len(egg_centers)
        n_qrs  = len(qr_centers)

        if n_eggs <= 1 or n_qrs <= 1:
            return np.zeros((n_eggs, n_qrs), dtype=np.float64)

        egg_x = np.array([c[0] for c in egg_centers], dtype=np.float64)
        qr_x  = np.array([c[0] for c in qr_centers],  dtype=np.float64)

        egg_x_norm = (egg_x - egg_x.min()) / (egg_x.max() - egg_x.min() + 1e-8)
        qr_x_norm  = (qr_x  - qr_x.min())  / (qr_x.max()  - qr_x.min()  + 1e-8)

        penalty = np.abs(
            egg_x_norm[:, np.newaxis] - qr_x_norm[np.newaxis, :]
        )  # (n_eggs, n_qrs)

        return penalty

    def _compute_pose_prior_cost(
        self,
        egg_centers: List[Tuple[float, float]],
        qr_centers: List[Tuple[float, float]],
        robot_pose: Dict,
    ) -> np.ndarray:
        """
        Compute pose prior cost matrix based on heading-direction alignment.

        For each (egg_j, qr_k) pair, the cost measures the angular deviation
        between:
          - the robot's current heading vector  (cos(yaw), sin(yaw))
          - the unit direction vector from egg_j to qr_k

        Formula::

            theta     = arccos( clip( dot(d_ej_qrk, robot_facing), -1, 1 ) )
            C_pose    = theta / pi          ∈ [0, 1]

        Intuition: QR codes that the robot is directly facing should be
        preferred over QR codes in the robot's peripheral view.

        The computation is fully vectorised using NumPy broadcasting.

        Args:
            egg_centers: Egg centre coordinate list [(x, y), ...].
            qr_centers:  QR centre coordinate list  [(x, y), ...].
            robot_pose:  Robot pose dict with at least 'yaw' (radians).

        Returns:
            Pose prior cost matrix of shape ``(n_eggs, n_qrs)`` in [0, 1].
            Returns zeros if yaw is unavailable (graceful degradation).
        """
        n_eggs = len(egg_centers)
        n_qrs  = len(qr_centers)

        yaw = robot_pose.get('yaw')
        if yaw is None:
            return np.zeros((n_eggs, n_qrs), dtype=np.float64)

        yaw = float(yaw)

        # Robot facing unit vector
        robot_facing = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)  # (2,)

        egg_arr = np.array(egg_centers, dtype=np.float64)  # (n_eggs, 2)
        qr_arr  = np.array(qr_centers,  dtype=np.float64)  # (n_qrs,  2)

        # Direction vectors: egg → QR for every pair
        # Shape: (n_eggs, n_qrs, 2)
        direction = qr_arr[np.newaxis, :, :] - egg_arr[:, np.newaxis, :]

        # Euclidean norms for normalisation, shape (n_eggs, n_qrs)
        norms = np.sqrt((direction ** 2).sum(axis=2))

        # Avoid division by zero for coincident points
        safe_norms = np.where(norms < 1e-8, 1.0, norms)

        # Normalised direction vectors, shape (n_eggs, n_qrs, 2)
        unit_dir = direction / safe_norms[:, :, np.newaxis]

        # Dot product with robot facing vector, shape (n_eggs, n_qrs)
        dot_product = (unit_dir * robot_facing[np.newaxis, np.newaxis, :]).sum(axis=2)

        # Clip to valid arccos domain and compute angle
        theta = np.arccos(np.clip(dot_product, -1.0, 1.0))  # (n_eggs, n_qrs)

        # Normalise to [0, 1]
        cost = theta / np.pi

        # For coincident egg–QR pairs set cost to 0 (no penalty)
        cost = np.where(norms < 1e-8, 0.0, cost)

        return cost

    # ------------------------------------------------------------------
    # Step 3+4: Hungarian solve + filter over-distance assignments
    # ------------------------------------------------------------------

    def _solve_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Run the Hungarian algorithm and filter out over-distance (inf) pairs.

        Improvement over naïve linear_sum_assignment: rows and columns that
        are entirely inf are first removed so that the large-number proxy
        cannot pollute the optimal solution.

        Args:
            cost_matrix: Shape ``(n_eggs, n_qrs)``, may contain inf entries.

        Returns:
            Valid assignment list ``[(egg_idx, qr_idx), ...]``.
        """
        finite_mask = np.isfinite(cost_matrix)
        valid_rows  = np.where(finite_mask.any(axis=1))[0]
        valid_cols  = np.where(finite_mask.any(axis=0))[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return []

        sub_cost   = cost_matrix[np.ix_(valid_rows, valid_cols)]
        sub_finite = np.where(np.isinf(sub_cost), 1e9, sub_cost)

        row_ind, col_ind = linear_sum_assignment(sub_finite)

        assignments = []
        for r, c in zip(row_ind, col_ind):
            orig_r = valid_rows[r]
            orig_c = valid_cols[c]
            if not np.isinf(cost_matrix[orig_r, orig_c]):
                assignments.append((int(orig_r), int(orig_c)))

        return assignments

    def _filter_assignments(
        self,
        row_ind: np.ndarray,
        col_ind: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Filter out inf-cost assignments (kept for backward compatibility).

        Args:
            row_ind:     Hungarian row indices (egg indices).
            col_ind:     Hungarian column indices (QR indices).
            cost_matrix: Cost matrix.

        Returns:
            Valid assignment list ``[(egg_idx, qr_idx), ...]``.
        """
        return [
            (int(r), int(c)) for r, c in zip(row_ind, col_ind)
            if not np.isinf(cost_matrix[r, c])
        ]

    # ------------------------------------------------------------------
    # Pose validation helper
    # ------------------------------------------------------------------

    def _is_pose_valid(self, robot_pose: Optional[Dict]) -> bool:
        """
        Return True iff robot_pose is a non-None dict containing 'yaw'.

        Gracefully handles None, empty dict, or missing 'yaw' key.

        Args:
            robot_pose: Robot pose dict or None.

        Returns:
            bool indicating whether pose information is usable.
        """
        if robot_pose is None:
            return False
        if not isinstance(robot_pose, dict):
            return False
        return 'yaw' in robot_pose and robot_pose['yaw'] is not None

    # ------------------------------------------------------------------
    # Step 5: update cage temporal states
    # ------------------------------------------------------------------

    def _update_cage_states(
        self,
        assignments: List[Tuple[int, int]],
        valid_qrs: List[Dict],
        egg_centers: List[Tuple[float, float]],
        frame: Optional[np.ndarray],
        confidences: Optional[List[float]] = None,
        pose_used: bool = False,
    ) -> None:
        """
        Update per-cage temporal state machine based on current frame results.

        State transitions::

            empty     → occupied:  egg detected this frame
            occupied  → confirmed: appear_num >= min_appear_frames
            occupied  → uncertain: 2 consecutive missed frames
            uncertain → empty:     3 consecutive missed frames
            confirmed → occupied:  egg disappeared (appear_num reset)

        Args:
            assignments: Valid assignment list ``[(egg_idx, qr_idx), ...]``.
            valid_qrs:   Valid QR detection list.
            egg_centers: Egg centre coordinate list.
            frame:       Current BGR frame (may be None).
            confidences: Per-assignment confidence scores aligned with
                         ``assignments``.  None defaults to all 1.0.
            pose_used:   Whether robot_pose contributed to this frame's cost.
        """
        if confidences is None:
            confidences = [1.0] * len(assignments)

        matched_cage_ids = set()

        for idx, (egg_idx, qr_idx) in enumerate(assignments):
            qr      = valid_qrs[qr_idx]
            cage_id = self._get_cage_id(qr)
            if cage_id is None:
                continue

            matched_cage_ids.add(cage_id)
            egg_center  = egg_centers[egg_idx]
            conf        = confidences[idx] if idx < len(confidences) else 1.0

            # Egg metadata
            egg_meta_i = (
                self._current_egg_meta[egg_idx]
                if (hasattr(self, '_current_egg_meta')
                    and egg_idx < len(self._current_egg_meta))
                else {'class_id': 0, 'is_invalid': False, 'score': 1.0}
            )

            if cage_id not in self._cage_states:
                self._cage_states[cage_id] = self._init_cage_state(cage_id)

            state = self._cage_states[cage_id]
            state['appear_num']       += 1
            state['miss_frames']       = 0
            state['last_egg_center']   = egg_center
            state['last_frame_count']  = self._frame_count
            state['match_confidence']  = conf     # NEW
            state['pose_used']         = pose_used  # NEW

            # Store matched QR centre for visualisation
            try:
                qr_cx = (qr['hbb'][0] + qr['hbb'][2]) / 2.0
                qr_cy = (qr['hbb'][1] + qr['hbb'][3]) / 2.0
                state['last_qr_center'] = (qr_cx, qr_cy)
            except (KeyError, TypeError, IndexError):
                center_raw = qr.get('center')
                if center_raw is not None:
                    state['last_qr_center'] = (
                        float(center_raw[0]), float(center_raw[1])
                    )

            state['last_egg_class_id'] = int(egg_meta_i.get('class_id', 0))
            state['last_is_invalid']   = bool(egg_meta_i.get('is_invalid', False))
            if state['last_is_invalid']:
                state['ever_invalid'] = True

            if frame is not None and state['frame'] is None:
                state['frame'] = frame.copy()

            # State machine transitions
            if state['status'] == STATE_EMPTY:
                state['status'] = STATE_OCCUPIED
            elif state['status'] == STATE_OCCUPIED:
                if state['appear_num'] >= self.min_appear_frames:
                    state['status'] = STATE_CONFIRMED
                    self._total_confirmed += 1
                    if frame is not None:
                        state['frame'] = frame.copy()
                        state['record_time'] = time.strftime(
                            '%Y-%m-%d %H:%M:%S', time.localtime()
                        )
            elif state['status'] == STATE_UNCERTAIN:
                state['status'] = STATE_OCCUPIED
            elif state['status'] == STATE_CONFIRMED:
                pass  # keep confirmed

        # Update states for cages not matched this frame
        for cage_id, state in self._cage_states.items():
            if cage_id not in matched_cage_ids:
                state['miss_frames'] += 1
                if state['status'] == STATE_OCCUPIED:
                    if state['miss_frames'] >= 2:
                        state['status'] = STATE_UNCERTAIN
                elif state['status'] == STATE_UNCERTAIN:
                    if state['miss_frames'] >= 3:
                        state['status'] = STATE_EMPTY
                        state['appear_num'] = 0
                elif state['status'] == STATE_CONFIRMED:
                    if state['miss_frames'] >= 5:
                        state['status']        = STATE_OCCUPIED
                        state['appear_num']    = 0
                        state['reported']      = False
                        state['frame']         = None
                        state['ever_invalid']  = False
                        state['last_is_invalid'] = False
                        state['last_egg_class_id'] = 0
                        state['match_confidence']  = 0.0
                        state['pose_used']         = False

    def _update_states_no_eggs(self, valid_qrs: List[Dict]) -> None:
        """
        Increment miss_frames for all known cages when no eggs are detected.

        Args:
            valid_qrs: Valid QR list (used to initialise newly seen cages).
        """
        for cage_id, state in self._cage_states.items():
            state['miss_frames'] += 1
            if state['status'] in (STATE_OCCUPIED, STATE_CONFIRMED):
                if state['miss_frames'] >= 2:
                    state['status'] = STATE_UNCERTAIN
            elif state['status'] == STATE_UNCERTAIN:
                if state['miss_frames'] >= 3:
                    state['status']    = STATE_EMPTY
                    state['appear_num'] = 0

    def _init_cage_state(self, cage_id: str) -> dict:
        """
        Initialise a new cage state dict with default values.

        Args:
            cage_id: Cage identifier string.

        Returns:
            Initial state dict.
        """
        return {
            'cage_id':           cage_id,
            'status':            STATE_EMPTY,
            'appear_num':        0,
            'miss_frames':       0,
            'last_egg_center':   None,
            'last_frame_count':  self._frame_count,
            'frame':             None,
            'record_time':       None,
            'reported':          False,
            'last_egg_class_id': 0,
            'last_is_invalid':   False,
            'ever_invalid':      False,
            # New fields for paper experiment E3
            'match_confidence':  0.0,   # most recent per-assignment confidence
            'pose_used':         False,  # whether pose was used when matched
        }

    def _get_cage_id(self, qr: Dict) -> Optional[str]:
        """
        Extract cage ID from a QR detection dict.

        Supports key aliases: 'cage_id', 'decode_id', 'qr_id'.

        Args:
            qr: QR detection dict.

        Returns:
            Cage ID string, or None if not found.
        """
        for key in ('cage_id', 'decode_id', 'qr_id'):
            val = qr.get(key)
            if val is not None:
                return str(val)
        return None

    # ------------------------------------------------------------------
    # Step 6: collect confirmed results
    # ------------------------------------------------------------------

    def _collect_confirmed_results(self) -> List[Dict]:
        """
        Collect all confirmed-but-not-yet-reported cage match results.

        Returns:
            List of result dicts compatible with the upload pipeline::

                {
                    'cage_id':          str,
                    'egg_num':          int,
                    'record_time':      str,
                    'frame_path':       str,
                    'appear_num':       int,
                    'egg_class_id':     int,
                    'is_invalid':       bool,
                    'egg_class':        str,
                    'match_confidence': float,  # NEW
                    'pose_used':        bool,   # NEW
                }
        """
        results = []

        for cage_id, state in self._cage_states.items():
            if state['status'] == STATE_CONFIRMED and not state['reported']:
                record_time = state.get('record_time') or time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime()
                )

                frame_path = ''
                if state['frame'] is not None:
                    try:
                        import cv2
                        fname = f"{cage_id}_{int(time.time() * 1000)}.jpg"
                        frame_path = os.path.join(
                            self.picture_recognition_path, fname
                        )
                        cv2.imwrite(frame_path, state['frame'])
                    except Exception as e:
                        print(f"TopologyMatcher: failed to save frame: {e}")

                result = {
                    'cage_id':          cage_id,
                    'egg_num':          1,
                    'record_time':      record_time,
                    'frame_path':       frame_path,
                    'appear_num':       state['appear_num'],
                    'egg_class_id':     int(state.get('last_egg_class_id', 0)),
                    'is_invalid':       bool(state.get('ever_invalid', False)),
                    'egg_class':        (
                        'invalidegg' if state.get('ever_invalid', False) else 'egg'
                    ),
                    # NEW fields for paper experiment E3
                    'match_confidence': float(state.get('match_confidence', 0.0)),
                    'pose_used':        bool(state.get('pose_used', False)),
                }
                results.append(result)
                state['reported'] = True

        return results

    # ------------------------------------------------------------------
    # New statistics method for paper experiment evaluation
    # ------------------------------------------------------------------

    def get_match_statistics(self) -> Dict:
        """
        Return cumulative matching statistics for paper experiment evaluation.

        Supports experiment E3 (egg-to-cage matching accuracy) metrics::

            {
                'total_frames':          int   – frames processed,
                'total_confirmed':       int   – cages confirmed,
                'avg_confidence':        float – mean assignment confidence,
                'pose_utilization_rate': float – fraction of frames with pose,
            }

        Returns:
            Statistics dict.
        """
        avg_conf = (
            float(np.mean(self._confidence_history))
            if self._confidence_history else 0.0
        )
        pose_util = (
            self._pose_used_frames / self._total_frames
            if self._total_frames > 0 else 0.0
        )
        return {
            'total_frames':          self._total_frames,
            'total_confirmed':       self._total_confirmed,
            'avg_confidence':        avg_conf,
            'pose_utilization_rate': pose_util,
        }

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------

    def get_cage_state(self, cage_id: str) -> Optional[dict]:
        """
        Return the current state dict for a given cage.

        Args:
            cage_id: Cage identifier.

        Returns:
            State dict, or None if the cage is unknown.
        """
        return self._cage_states.get(cage_id)

    def get_all_cage_states(self) -> Dict[str, dict]:
        """
        Return a shallow copy of all cage state dicts.

        Returns:
            ``{cage_id: state_dict}`` dictionary.
        """
        return dict(self._cage_states)

    def reset(self) -> None:
        """
        Reset all cage states, frame counter, and statistics.
        """
        self._cage_states.clear()
        self._frame_count       = 0
        self._total_frames      = 0
        self._total_confirmed   = 0
        self._confidence_history.clear()
        self._pose_used_frames  = 0
        print("TopologyMatcher: state reset")
