# coding=utf-8
"""
拓扑约束蛋-笼匹配模块
使用匈牙利算法结合拓扑约束实现种蛋到笼位的最优匹配
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


# 笼位时序状态枚举
STATE_EMPTY     = 'empty'       # 未检测到蛋
STATE_OCCUPIED  = 'occupied'    # 检测到蛋但尚未确认
STATE_UNCERTAIN = 'uncertain'   # 检测不稳定
STATE_CONFIRMED = 'confirmed'   # 已确认（出现帧数 >= min_appear_frames）


class TopologyMatcher:
    """
    拓扑约束蛋-笼匹配器。

    使用匈牙利算法对种蛋中心点与有效二维码笼位进行最优匹配，
    并维护每个笼位的时序占用状态。

    Config dict 支持的键：
      - max_match_distance:  最大允许匹配距离（默认 200）
      - cost_weights:        代价权重字典，包含 position（默认 0.6）、
                             validity（默认 0.3）、topology（默认 0.1）
      - validity_threshold:  最低 QR 有效性分数（默认 0.7）
      - min_appear_frames:   确认匹配所需最少帧数（默认 5）
      - picture_recognition_path: 识别图片保存路径（可选）
    """

    def __init__(self, cfg: dict):
        """
        初始化拓扑匹配器。

        Args:
            cfg: 配置字典，支持的键见类文档
        """
        self.max_match_distance = cfg.get('max_match_distance', 200)
        self.validity_threshold = cfg.get('validity_threshold', 0.7)
        self.min_appear_frames  = cfg.get('min_appear_frames', 5)

        # 代价权重
        _weights = cfg.get('cost_weights', {})
        self.w_position = _weights.get('position', 0.6)
        self.w_validity = _weights.get('validity', 0.3)
        self.w_topology = _weights.get('topology', 0.1)

        # 图片保存路径
        self.picture_recognition_path = cfg.get('picture_recognition_path', 'recognition_output')
        if not os.path.exists(self.picture_recognition_path):
            os.makedirs(self.picture_recognition_path)

        # 笼位状态字典：cage_id -> 状态信息
        # 键为 cage_id（str），值为状态 dict
        self._cage_states: Dict[str, dict] = {}

        # 帧计数器（用于超时清理）
        self._frame_count: int = 0

        # 打印缺失可选配置的警告
        _optional_defaults = {
            'max_match_distance': 200,
            'validity_threshold': 0.7,
            'min_appear_frames': 5,
        }
        for key, default in _optional_defaults.items():
            if key not in cfg:
                print(f"TopologyMatcher: 配置缺少 '{key}'，使用默认值 {default}")

    # ------------------------------------------------------------------
    # 主匹配接口
    # ------------------------------------------------------------------

    @exception_handler
    def match(
        self,
        egg_centers: List[Tuple[float, float]],
        qr_detections: List[Dict],
        frame: Optional[np.ndarray] = None,
        egg_meta: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        执行蛋-笼匹配，返回与上传管道兼容的匹配结果列表。

        Args:
            egg_centers:    种蛋中心点列表，每个元素为 (x, y)
            qr_detections:  QR 检测结果列表
            frame:          当前帧图像（BGR），可选
            egg_meta:       与 egg_centers 一一对应的元信息列表，每个元素含：
                              - class_id:   0=egg, 1=invalidegg
                              - is_invalid: bool
                              - score:      置信度
                            若未提供，默认所有蛋视为正常（class_id=0）。

        Returns:
            匹配结果列表，每个元素新增 `is_invalid` / `egg_class` 字段供质量统计使用。
        """
        self._frame_count += 1

        # 把 egg_meta 缓存到实例上，供 _update_cage_states 读取
        if egg_meta is not None and len(egg_meta) == len(egg_centers):
            self._current_egg_meta = list(egg_meta)
        else:
            self._current_egg_meta = [
                {'class_id': 0, 'is_invalid': False, 'score': 1.0}
                for _ in egg_centers
            ]

        # 步骤 1：过滤有效 QR（validity_score >= validity_threshold）
        valid_qrs = self._filter_valid_qrs(qr_detections)

        # 步骤 2：构建代价矩阵（行=蛋，列=有效 QR）
        if len(egg_centers) == 0 or len(valid_qrs) == 0:
            # 无蛋或无有效 QR，更新所有已知笼位为 empty
            self._update_states_no_eggs(valid_qrs)
            return []

        cost_matrix = self._build_cost_matrix(egg_centers, valid_qrs)

        # 步骤 3：匈牙利算法求解最优分配
        # 若代价矩阵全为 inf（所有蛋都超出匹配距离），直接返回空结果
        if np.all(np.isinf(cost_matrix)):
            self._update_states_no_eggs(valid_qrs)
            return []

        # 步骤 3a：剔除"全 inf 行/列"以避免次优解。
        # 直接用大数替换 inf 后做 linear_sum_assignment 可能让本应保留的
        # 近距离分配被远距离 inf 抢占（远距离虽然代价大但仍可达）。
        assignments = self._solve_assignment(cost_matrix)

        # 步骤 5：更新笼位时序状态
        self._update_cage_states(assignments, valid_qrs, egg_centers, frame)

        # 步骤 6：收集已确认的匹配结果
        results = self._collect_confirmed_results()

        return results

    # ------------------------------------------------------------------
    # 步骤 1：过滤有效 QR
    # ------------------------------------------------------------------

    def _filter_valid_qrs(self, qr_detections: List[Dict]) -> List[Dict]:
        """
        过滤 validity_score >= validity_threshold 的 QR 检测结果。

        Args:
            qr_detections: 原始 QR 检测列表

        Returns:
            有效 QR 列表（保留原始 dict 引用）
        """
        valid = []
        for qr in qr_detections:
            score = qr.get('validity_score', 0.0)
            if score >= self.validity_threshold:
                valid.append(qr)
        return valid

    # ------------------------------------------------------------------
    # 步骤 2：构建代价矩阵
    # ------------------------------------------------------------------

    def _build_cost_matrix(
        self,
        egg_centers: List[Tuple[float, float]],
        valid_qrs: List[Dict],
    ) -> np.ndarray:
        """
        构建蛋-QR 分配代价矩阵。

        代价 = w_position * 归一化距离 + w_validity * (1 - validity_score)
               + w_topology * 拓扑惩罚

        Args:
            egg_centers: 种蛋中心点列表 [(x, y), ...]
            valid_qrs:   有效 QR 检测列表

        Returns:
            代价矩阵，shape (n_eggs, n_qrs)，dtype float64
        """
        n_eggs = len(egg_centers)
        n_qrs  = len(valid_qrs)

        cost_matrix = np.zeros((n_eggs, n_qrs), dtype=np.float64)

        # 提取 QR 中心点和有效性分数
        qr_centers   = [self._get_qr_center(qr) for qr in valid_qrs]
        qr_validities = [qr.get('validity_score', 1.0) for qr in valid_qrs]

        # 计算所有蛋-QR 欧氏距离
        egg_arr = np.array(egg_centers, dtype=np.float64)   # (n_eggs, 2)
        qr_arr  = np.array(qr_centers,  dtype=np.float64)   # (n_qrs,  2)

        # 广播计算距离矩阵
        diff = egg_arr[:, np.newaxis, :] - qr_arr[np.newaxis, :, :]  # (n_eggs, n_qrs, 2)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))                # (n_eggs, n_qrs)

        # 标记超出最大匹配距离的格子（后续设为 INF）
        out_of_range = dist_matrix > self.max_match_distance

        # 归一化距离到 [0, 1]（以 max_match_distance 为上限）
        norm_dist = np.clip(dist_matrix / self.max_match_distance, 0.0, 1.0)

        # 有效性惩罚：validity 越低，惩罚越高
        validity_arr     = np.array(qr_validities, dtype=np.float64)  # (n_qrs,)
        validity_penalty = 1.0 - validity_arr                          # (n_qrs,)

        # 拓扑惩罚（基于 QR 位置排列的相对顺序一致性）
        topology_penalty = self._compute_topology_penalty(egg_centers, qr_centers)

        # 合并代价
        cost_matrix = (
            self.w_position * norm_dist
            + self.w_validity * validity_penalty[np.newaxis, :]
            + self.w_topology * topology_penalty
        )

        # 超出最大匹配距离的格子设为 INF，确保匈牙利算法不会选择这些分配
        cost_matrix[out_of_range] = np.inf

        return cost_matrix

    def _get_qr_center(self, qr: Dict) -> Tuple[float, float]:
        """
        从 QR 检测 dict 中提取中心点坐标。

        支持多种键名：center、hbb、box。

        Args:
            qr: QR 检测 dict

        Returns:
            (cx, cy) 中心点坐标
        """
        if 'center' in qr:
            return tuple(qr['center'][:2])

        # 从水平外接矩形计算中心点
        box = qr.get('hbb') or qr.get('box')
        if box is not None and len(box) >= 4:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            return (cx, cy)

        # 从旋转框提取中心点
        rotated = qr.get('rotated_box')
        if rotated is not None and len(rotated) >= 2:
            return (float(rotated[0]), float(rotated[1]))

        # 默认返回原点（不应发生）
        print(f"TopologyMatcher: 无法从 QR 检测中提取中心点，使用 (0, 0)")
        return (0.0, 0.0)

    def _compute_topology_penalty(
        self,
        egg_centers: List[Tuple[float, float]],
        qr_centers: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        计算拓扑惩罚矩阵。

        拓扑约束：蛋和 QR 的水平排列顺序应保持一致。
        若蛋 i 在蛋 j 左侧，则其匹配的 QR 也应在 QR j 左侧。
        违反此约束时施加惩罚。

        Args:
            egg_centers: 种蛋中心点列表
            qr_centers:  QR 中心点列表

        Returns:
            拓扑惩罚矩阵，shape (n_eggs, n_qrs)，值域 [0, 1]
        """
        n_eggs = len(egg_centers)
        n_qrs  = len(qr_centers)

        if n_eggs <= 1 or n_qrs <= 1:
            return np.zeros((n_eggs, n_qrs), dtype=np.float64)

        # 蛋和 QR 的 x 坐标排名（归一化到 [0, 1]）
        egg_x = np.array([c[0] for c in egg_centers], dtype=np.float64)
        qr_x  = np.array([c[0] for c in qr_centers],  dtype=np.float64)

        egg_x_norm = (egg_x - egg_x.min()) / (egg_x.max() - egg_x.min() + 1e-8)
        qr_x_norm  = (qr_x  - qr_x.min())  / (qr_x.max()  - qr_x.min()  + 1e-8)

        # 拓扑惩罚 = |蛋归一化 x - QR 归一化 x|（位置不一致时惩罚大）
        penalty = np.abs(
            egg_x_norm[:, np.newaxis] - qr_x_norm[np.newaxis, :]
        )  # (n_eggs, n_qrs)

        return penalty

    # ------------------------------------------------------------------
    # 步骤 3+4：匈牙利求解 + 过滤超距分配
    # ------------------------------------------------------------------

    def _solve_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        对代价矩阵执行匈牙利算法，并过滤掉超距（inf）分配。

        改进：先剔除"全 inf 行/列"，对剩余子矩阵做分配，
        避免 1e9 大数干扰最优解。

        Args:
            cost_matrix: shape (n_eggs, n_qrs)，含 inf 的代价矩阵

        Returns:
            有效分配列表 [(egg_idx, qr_idx), ...]
        """
        n_eggs, n_qrs = cost_matrix.shape

        # 找出至少有一个有限代价的行和列
        finite_mask = np.isfinite(cost_matrix)
        valid_rows = np.where(finite_mask.any(axis=1))[0]
        valid_cols = np.where(finite_mask.any(axis=0))[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return []

        # 提取子矩阵
        sub_cost = cost_matrix[np.ix_(valid_rows, valid_cols)]
        # 将子矩阵中残余的 inf 替换为大数（此时不会影响有限代价的最优解）
        sub_finite = np.where(np.isinf(sub_cost), 1e9, sub_cost)

        row_ind, col_ind = linear_sum_assignment(sub_finite)

        # 映射回原始索引并过滤 inf
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
        过滤 INF 代价的分配（保留向后兼容）。

        Args:
            row_ind:     匈牙利算法返回的行索引（蛋索引）
            col_ind:     匈牙利算法返回的列索引（QR 索引）
            cost_matrix: 代价矩阵

        Returns:
            有效分配列表 [(egg_idx, qr_idx), ...]
        """
        valid_assignments = [
            (int(r), int(c)) for r, c in zip(row_ind, col_ind)
            if not np.isinf(cost_matrix[r, c])
        ]
        return valid_assignments

    # ------------------------------------------------------------------
    # 步骤 5：更新笼位时序状态
    # ------------------------------------------------------------------

    def _update_cage_states(
        self,
        assignments: List[Tuple[int, int]],
        valid_qrs: List[Dict],
        egg_centers: List[Tuple[float, float]],
        frame: Optional[np.ndarray],
    ) -> None:
        """
        根据当前帧的匹配结果更新每个笼位的时序状态。

        状态机转换：
          empty     → occupied:  本帧检测到蛋
          occupied  → confirmed: appear_num >= min_appear_frames
          occupied  → uncertain: 连续 2 帧未检测到蛋
          uncertain → empty:     连续 3 帧未检测到蛋
          confirmed → occupied:  蛋消失（appear_num 重置）

        Args:
            assignments: 有效分配列表 [(egg_idx, qr_idx), ...]
            valid_qrs:   有效 QR 列表
            egg_centers: 种蛋中心点列表
            frame:       当前帧图像（可为 None）
        """
        # 收集本帧有匹配的 cage_id
        matched_cage_ids = set()

        for egg_idx, qr_idx in assignments:
            qr = valid_qrs[qr_idx]
            cage_id = self._get_cage_id(qr)
            if cage_id is None:
                continue

            matched_cage_ids.add(cage_id)
            egg_center = egg_centers[egg_idx]

            # 读取本蛋的 class_id（0=egg, 1=invalidegg）
            egg_meta_i = (
                self._current_egg_meta[egg_idx]
                if (hasattr(self, '_current_egg_meta')
                    and egg_idx < len(self._current_egg_meta))
                else {'class_id': 0, 'is_invalid': False, 'score': 1.0}
            )

            if cage_id not in self._cage_states:
                self._cage_states[cage_id] = self._init_cage_state(cage_id)

            state = self._cage_states[cage_id]
            state['appear_num']    += 1
            state['miss_frames']    = 0
            state['last_egg_center'] = egg_center
            state['last_frame_count'] = self._frame_count

            # 记录最近一次观测到的蛋类别（用于上报时区分好/坏蛋）
            state['last_egg_class_id'] = int(egg_meta_i.get('class_id', 0))
            state['last_is_invalid']   = bool(egg_meta_i.get('is_invalid', False))
            # 若曾观测到 invalidegg，则置 True 并保留（即使后续帧又只看到完整蛋）
            if state['last_is_invalid']:
                state['ever_invalid'] = True

            # 保存帧图片（仅在首次确认时）
            if frame is not None and state['frame'] is None:
                state['frame'] = frame.copy()

            # 状态转换
            if state['status'] == STATE_EMPTY:
                state['status'] = STATE_OCCUPIED
            elif state['status'] == STATE_OCCUPIED:
                if state['appear_num'] >= self.min_appear_frames:
                    state['status'] = STATE_CONFIRMED
                    # 保存确认帧
                    if frame is not None:
                        state['frame'] = frame.copy()
                        state['record_time'] = time.strftime(
                            '%Y-%m-%d %H:%M:%S', time.localtime()
                        )
            elif state['status'] == STATE_UNCERTAIN:
                state['status'] = STATE_OCCUPIED
            elif state['status'] == STATE_CONFIRMED:
                pass  # 保持 confirmed

        # 更新未匹配到蛋的笼位状态
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
                    # 确认后蛋消失，重置为 occupied 等待再次确认
                    if state['miss_frames'] >= 5:
                        state['status'] = STATE_OCCUPIED
                        state['appear_num'] = 0
                        state['reported'] = False  # 复位上报标志，允许下次重新上报
                        state['frame'] = None
                        state['ever_invalid'] = False
                        state['last_is_invalid'] = False
                        state['last_egg_class_id'] = 0

    def _update_states_no_eggs(self, valid_qrs: List[Dict]) -> None:
        """
        当本帧无蛋检测时，更新所有已知笼位的 miss_frames。

        Args:
            valid_qrs: 有效 QR 列表（用于初始化新发现的笼位）
        """
        for cage_id, state in self._cage_states.items():
            state['miss_frames'] += 1
            if state['status'] in (STATE_OCCUPIED, STATE_CONFIRMED):
                if state['miss_frames'] >= 2:
                    state['status'] = STATE_UNCERTAIN
            elif state['status'] == STATE_UNCERTAIN:
                if state['miss_frames'] >= 3:
                    state['status'] = STATE_EMPTY
                    state['appear_num'] = 0

    def _init_cage_state(self, cage_id: str) -> dict:
        """
        初始化笼位状态字典。

        Args:
            cage_id: 笼位 ID

        Returns:
            初始状态 dict
        """
        return {
            'cage_id':         cage_id,
            'status':          STATE_EMPTY,
            'appear_num':      0,
            'miss_frames':     0,
            'last_egg_center': None,
            'last_frame_count': self._frame_count,
            'frame':           None,
            'record_time':     None,
            'reported':        False,   # 是否已上报
            'last_egg_class_id': 0,     # 最近一次观测到的蛋类别（0=egg, 1=invalidegg）
            'last_is_invalid':   False,
            'ever_invalid':      False, # 该笼位本次确认期间是否曾出现过 invalidegg
        }

    def _get_cage_id(self, qr: Dict) -> Optional[str]:
        """
        从 QR 检测 dict 中提取笼位 ID。

        支持多种键名：cage_id、decode_id、qr_id。

        Args:
            qr: QR 检测 dict

        Returns:
            笼位 ID 字符串，若无法提取则返回 None
        """
        for key in ('cage_id', 'decode_id', 'qr_id'):
            val = qr.get(key)
            if val is not None:
                return str(val)
        return None

    # ------------------------------------------------------------------
    # 步骤 6：收集已确认结果
    # ------------------------------------------------------------------

    def _collect_confirmed_results(self) -> List[Dict]:
        """
        收集所有状态为 confirmed 且尚未上报的笼位匹配结果。

        Returns:
            匹配结果列表，格式与上传管道兼容：
            [{'cage_id', 'egg_num', 'record_time', 'frame_path', 'appear_num'}]
        """
        results = []

        for cage_id, state in self._cage_states.items():
            if state['status'] == STATE_CONFIRMED and not state['reported']:
                record_time = state.get('record_time') or time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime()
                )

                # 保存帧图片
                frame_path = ''
                if state['frame'] is not None:
                    try:
                        import cv2
                        fname = f"{cage_id}_{int(time.time() * 1000)}.jpg"
                        frame_path = os.path.join(self.picture_recognition_path, fname)
                        cv2.imwrite(frame_path, state['frame'])
                    except Exception as e:
                        print(f"TopologyMatcher: 保存帧图片失败: {e}")

                result = {
                    'cage_id':     cage_id,
                    'egg_num':     1,           # 每个笼位对应一个蛋
                    'record_time': record_time,
                    'frame_path':  frame_path,
                    'appear_num':  state['appear_num'],
                    # 质量信息：供后续产蛋质量统计 / 论文实验
                    'egg_class_id': int(state.get('last_egg_class_id', 0)),
                    'is_invalid':   bool(state.get('ever_invalid', False)),
                    'egg_class':    'invalidegg' if state.get('ever_invalid', False) else 'egg',
                }
                results.append(result)
                state['reported'] = True

        return results

    # ------------------------------------------------------------------
    # 公共工具方法
    # ------------------------------------------------------------------

    def get_cage_state(self, cage_id: str) -> Optional[dict]:
        """
        获取指定笼位的当前状态。

        Args:
            cage_id: 笼位 ID

        Returns:
            状态 dict，若笼位不存在则返回 None
        """
        return self._cage_states.get(cage_id)

    def get_all_cage_states(self) -> Dict[str, dict]:
        """
        获取所有笼位的当前状态。

        Returns:
            {cage_id: state_dict} 字典
        """
        return dict(self._cage_states)

    def reset(self) -> None:
        """
        重置所有笼位状态和帧计数器。
        """
        self._cage_states.clear()
        self._frame_count = 0
        print("TopologyMatcher: 状态已重置")
