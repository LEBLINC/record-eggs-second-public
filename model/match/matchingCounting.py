# coding=utf-8
"""
    部署版本，去除二维码辅助推断
    @project: EGGRECORDQT
    @Author：wjt
    @file： main.py
    @date：2024/1/10 19:12
"""
import cv2
import numpy as np
import time
from model.track.matchUtils import *
from pyzbar.pyzbar import decode, ZBarSymbol
from collections import deque
import os
import re
from typing import Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from model.utils.path_utils import get_app_root
import urllib.request


class MatchingCounting:
    # OpenCV/WeChat QR 解码器在多线程下可能出现堆损坏，统一串行化解码
    _cv_decode_lock = threading.Lock()
    # WeChatQRCode 模型文件下载仅尝试一次（避免4路摄像头各自阻塞/重复下载）
    _wechat_model_download_lock = threading.Lock()
    _wechat_model_download_attempted = False

    def __init__(self, cfg):

        self.qr_dist = {}  # 用于记录二维码信息，键值位目标跟踪id
        self.qr_id = 1  # 代表记录的二维码数量
        self.egg_dist = {}  # 用于记录蛋信息，键值位目标跟踪id
        self.egg_id = 1  # 代表记录的蛋数量

        self.count = 0

        # 用于提前向UI上报已识别但未汇总的二维码事件
        self.early_results = []
        self.early_emit = None  # 可选：直接回调到上层以立即通知UI
        try:
            self.camera_idx = int(cfg.get('camera_idx', -1)) if isinstance(cfg, dict) else -1
        except Exception:
            self.camera_idx = -1

        # ----------------------------
        # 二维码解码增强配置（默认值即可直接用）
        # ----------------------------
        qr_decode_cfg = cfg.get('qr_decode', {}) if isinstance(cfg, dict) else {}
        # 重要：实时场景下不要“每帧多次解码”，否则会非常卡；默认做强节流 + 异步后台解码
        self.qr_use_async = bool(qr_decode_cfg.get('use_async', True))
        self.qr_max_workers = int(qr_decode_cfg.get('max_workers', 1))  # 每路摄像头建议 1
        # 提高解码频率（你当前摄像头只有 640x480，移动巡检时“可解窗口”很短，需要更频繁尝试）
        # 仍采用异步后台解码，避免阻塞主匹配线程导致画面卡顿。
        self.qr_global_interval_ms = int(qr_decode_cfg.get('global_interval_ms', 250))  # 每路摄像头全局解码间隔
        self.qr_try_interval_ms = int(qr_decode_cfg.get('try_interval_ms', 600))  # 同一track两次解码最小间隔
        self.qr_max_tries = int(qr_decode_cfg.get('max_tries', 30))  # 单个track最多尝试次数，避免无限消耗CPU

        self.qr_min_pad_px = int(qr_decode_cfg.get('min_pad_px', 20))          # ROI最小外扩像素（保证静区）
        self.qr_pad_ratios = qr_decode_cfg.get('pad_ratios', [0.35, 0.70])     # ROI按框大小外扩比例（可多档尝试）
        # 低分辨率场景下二维码框会很小，不能卡太严；真正“太小”由缩放上采样兜底解决
        self.qr_min_box_size = int(qr_decode_cfg.get('min_box_size', 14))      # 二维码框最小边（太小基本解不出）
        self.qr_target_min_side = int(qr_decode_cfg.get('target_min_side', 360))  # 将ROI缩放到至少这么大再解码
        # 仅保留画面上方二维码（避免旧二维码干扰）
        try:
            self.qr_filter_enabled = bool(qr_decode_cfg.get('qr_filter_enabled', True))
        except Exception:
            self.qr_filter_enabled = True
        try:
            self.qr_top_ratio = float(qr_decode_cfg.get('qr_top_ratio', 0.45))
        except Exception:
            self.qr_top_ratio = 0.45
        self.qr_top_ratio = max(0.05, min(0.95, self.qr_top_ratio))
        # 二维码短暂丢检时的保活时间（秒）
        try:
            self.qr_keep_seconds = float(qr_decode_cfg.get('qr_keep_seconds', 0.4))
        except Exception:
            self.qr_keep_seconds = 0.4
        self.qr_keep_seconds = max(0.0, self.qr_keep_seconds)

        # 兼容旧配置（仍然允许传 scales，但默认不用“多尺度狂扫”）
        self.qr_max_scales = qr_decode_cfg.get('scales', [1, 2])               # 小码放大尝试（轻量）
        self.qr_debug_save = bool(qr_decode_cfg.get('debug_save', False))      # 失败样本落盘（默认关闭）
        self.qr_debug_max = int(qr_decode_cfg.get('debug_max', 50))
        self.qr_debug_dir = str(qr_decode_cfg.get('debug_dir', ''))            # 为空则默认放到 picture_recognition_path/qr_debug
        self._qr_debug_saved = 0
        self._qr_lock = threading.Lock()
        self._qr_inflight = set()
        self._qr_last_global_decode_ms = 0
        self._qr_pool = ThreadPoolExecutor(max_workers=max(1, self.qr_max_workers)) if self.qr_use_async else None
        # 二维码“正对面”抓拍配置（用于复核）
        try:
            self.qr_capture_delay_after_detect_s = float(qr_decode_cfg.get('qr_capture_delay_after_detect_s', 5.0))
        except Exception:
            self.qr_capture_delay_after_detect_s = 5.0
        try:
            self.qr_capture_delay_after_decode_s = float(qr_decode_cfg.get('qr_capture_delay_after_decode_s', 3.5))
        except Exception:
            self.qr_capture_delay_after_decode_s = 3.5
        try:
            self.qr_capture_window_s = float(qr_decode_cfg.get('qr_capture_window_s', 6.0))
        except Exception:
            self.qr_capture_window_s = 6.0
        try:
            self.qr_capture_pad_ratio = float(qr_decode_cfg.get('qr_capture_pad_ratio', 0.35))
        except Exception:
            self.qr_capture_pad_ratio = 0.35
        self.qr_image_results = []
        # 外部（例如Track线程）早期解码的兜底注入：track_id -> cage_id
        self._pending_external_decodes = {}

        # WeChatQRCode（最强解码器）：显式加载 resources/wechat 下的4个模型文件
        self.wechat_detector = None
        self._init_wechat_detector(cfg)

        # 轻量级OpenCV二维码检测器，作为pyzbar失败或多结果时的兜底
        try:
            self.cv_qr_detector = cv2.QRCodeDetector()
        except Exception:
            self.cv_qr_detector = None
        self.edge_threshold = int(cfg['width'] * 0.05)
        self.egg_diff_num = 0
        self.qr_diff_num = 0
        self.width = int(cfg['width'] * 0.05)

        # 初始化队列
        self.egg_appear_nums_queue = deque(maxlen=10)
        self.qr_appear_nums_queue = deque(maxlen=10)

        self.match_center = cfg['width'] // 2
        self.match_range = cfg['width'] // 2 - self.edge_threshold

        # 读取匹配增强配置（若缺省则使用默认值）
        matching_cfg = cfg.get('matching', {}) if isinstance(cfg, dict) else {}
        self.alpha_x = float(matching_cfg.get('alpha_x', 0.45))  # 自适应横向阈值比例
        self.beta_edge = float(matching_cfg.get('beta_edge', 0.12))  # 走廊边界忽略比例
        self.switch_n = int(matching_cfg.get('switch_n', 5))
        self.switch_margin_ratio = float(matching_cfg.get('switch_margin_ratio', 0.2))
        self.stable_T = int(matching_cfg.get('stable_T', 10))
        # 当新二维码明显更近时，允许“快速切换”（0~1，越小越容易切换）
        try:
            self.switch_min_ratio = float(matching_cfg.get('switch_min_ratio', 0.7))
        except Exception:
            self.switch_min_ratio = 0.7
        self.switch_min_ratio = max(0.2, min(0.95, self.switch_min_ratio))
        try:
            self.use_force_on_timeout = bool(matching_cfg.get('use_force_on_timeout', True))
        except Exception:
            self.use_force_on_timeout = True
        # 蛋检测置信度门槛（用于匹配阶段过滤误检，不影响二维码）
        try:
            self.egg_detect_min_conf = float(matching_cfg.get('egg_detect_min_conf', 0.30))
        except Exception:
            self.egg_detect_min_conf = 0.30
        # 更早锁定阈值（减少同一颗蛋被重复匹配到下一个二维码）
        try:
            self.lock_T = int(matching_cfg.get('lock_T', max(3, int(self.stable_T * 0.5))))
        except Exception:
            self.lock_T = max(3, int(self.stable_T * 0.5))
        self.lock_T = max(1, self.lock_T)
        try:
            self.switch_requires_prev_missing = bool(matching_cfg.get('switch_requires_prev_missing', True))
        except Exception:
            self.switch_requires_prev_missing = True
        # 蛋的置信度过滤（抑制“闪一下就被计数”的误检）
        # - egg_min_conf：低于该阈值的“egg”不会参与最终计数（仍可用于跟踪/显示框）
        # - egg_conf_window：对单个 track 的置信度做滑窗平均，避免单帧抖动
        try:
            self.egg_min_conf = float(matching_cfg.get('egg_min_conf', 0.15))
        except Exception:
            self.egg_min_conf = 0.15
        try:
            self.egg_conf_window = int(matching_cfg.get('egg_conf_window', 8))
        except Exception:
            self.egg_conf_window = 8
        self.egg_conf_window = max(1, self.egg_conf_window)

        # 停止巡检时的“强制汇总”最低稳定帧门槛（避免 stop 时把闪检也算进去）
        # 默认：max(3, stable_T/2)
        try:
            self.force_stable_T = int(matching_cfg.get('force_stable_T', max(3, int(self.stable_T * 0.5))))
        except Exception:
            self.force_stable_T = max(3, int(self.stable_T * 0.5))
        self.force_stable_T = max(1, self.force_stable_T)

        # 已计数蛋的冷却期：防止同一颗蛋在移动到下一个笼位时被重复匹配/重复计数
        try:
            self.egg_reuse_cooldown_s = float(matching_cfg.get('egg_reuse_cooldown_s', 120.0))
        except Exception:
            self.egg_reuse_cooldown_s = 120.0
        self.egg_reuse_cooldown_s = max(0.0, self.egg_reuse_cooldown_s)
        self._used_egg_track_ids = {}  # egg_track_id -> last_count_time(time.time())
        # 蛋track_id跳变去重（同一蛋短时间被重检）
        try:
            self.egg_reid_dedupe_seconds = float(matching_cfg.get('egg_reid_dedupe_seconds', 8.0))
        except Exception:
            self.egg_reid_dedupe_seconds = 8.0
        try:
            self.egg_reid_iou_thresh = float(matching_cfg.get('egg_reid_iou_thresh', 0.6))
        except Exception:
            self.egg_reid_iou_thresh = 0.6
        try:
            self.egg_reid_dist_ratio = float(matching_cfg.get('egg_reid_dist_ratio', 0.6))
        except Exception:
            self.egg_reid_dist_ratio = 0.6
        self._recent_cage_eggs = {}
        vcfg = matching_cfg.get('vertical_roi', {}) if isinstance(matching_cfg, dict) else {}
        self.vertical_roi_enabled = bool(vcfg.get('enabled', True))
        self.vertical_roi_offset_ratio = float(vcfg.get('qr_offset_ratio', 0.2))
        self.vertical_roi_bottom_ratio = float(vcfg.get('bottom_ratio', 0.95))

        # 每帧构建的二维码走廊上下文：{qr_track_id: {left,right,center_x,center_y,height,s_px,box}}
        self.qr_context_map = {}
        self.frame_height = None

        self.picture_recognition_path = cfg['picture_recognition_path']
        if not os.path.exists(self.picture_recognition_path):
            os.makedirs(self.picture_recognition_path)

        self.color_map = {}

    def ingest_external_qr_decode(self, track_id: int, cage_id: str) -> None:
        """
        接收外部来源（例如 Track 线程的早期解码）得到的笼号，并写回到本匹配器的 qr_dist。
        这是解决“巡检时UI能看到笼号，但结束汇总产蛋数=0”的关键：
        - UI早期解码不参与计数
        - 计数/汇总必须依赖 qr_dist[track_id]['decode_id']
        """
        try:
            if cage_id is None:
                return
            cage_id = str(cage_id).strip()
            if not cage_id:
                return
            tid = int(track_id)
        except Exception:
            return

        try:
            with self._qr_lock:
                if tid in self.qr_dist:
                    if self.qr_dist[tid].get('decode_id') is None:
                        self.qr_dist[tid]['decode_id'] = cage_id
                        if self.qr_dist[tid].get('decode_ts') is None:
                            self.qr_dist[tid]['decode_ts'] = time.time()
                        print(f"[QR_SUCCESS] {cage_id} by Track-Early-Inject")
                        # 允许后续 match() 自动补发 early 事件（ui_reported 仍为 False）
                else:
                    # 可能Track线程先解码、Match线程稍后才创建qr_dist条目，先暂存
                    self._pending_external_decodes[tid] = cage_id
        except Exception:
            pass

    def _archive_egg_to_qr(self, egg_track_id: int, egg_info: dict) -> None:
        """
        将即将被清理（超时/离开画面）的蛋信息“归档”到其最终匹配到的二维码记录中。

        目的：解决“巡检时能看到框/连线，但镜头移开/结束巡检后汇总蛋数=0”的问题。
        关键点：
        - 统计不应只依赖当前仍在 self.egg_dist 的目标（它们可能已经超时被删）
        - 归档保留该蛋在离开前的 stable_frames，用于后续汇总时判定是否计数
        """
        try:
            if not isinstance(egg_info, dict):
                return
            qr_tid = egg_info.get('min_qr_track_id')
            if qr_tid is None:
                return
            if qr_tid not in self.qr_dist:
                return
            # 仅归档“确实有过匹配”的蛋
            try:
                stable_frames = int(egg_info.get('stable_frames', 0) or 0)
            except Exception:
                stable_frames = 0
            if stable_frames <= 0:
                return

            qr_info = self.qr_dist.get(qr_tid)
            if not isinstance(qr_info, dict):
                return
            archive = qr_info.get('egg_archive')
            if not isinstance(archive, dict):
                archive = {}
                qr_info['egg_archive'] = archive

            # 保留该蛋历史最大 stable_frames（防止多次归档覆盖更大的值）
            prev_frames = 0
            prev_conf = None
            try:
                old_val = archive.get(egg_track_id)
                if isinstance(old_val, dict):
                    prev_frames = int(old_val.get('stable_frames', old_val.get('frames', 0)) or 0)
                    if old_val.get('conf_max') is not None:
                        prev_conf = float(old_val.get('conf_max'))
                    elif old_val.get('conf') is not None:
                        prev_conf = float(old_val.get('conf'))
                else:
                    prev_frames = int(old_val or 0)
            except Exception:
                prev_frames = 0
                prev_conf = None

            # 置信度信息（可选）：用于过滤“闪检误计数”
            cur_conf = None
            try:
                if isinstance(egg_info, dict):
                    if egg_info.get('conf_max') is not None:
                        cur_conf = float(egg_info.get('conf_max'))
            except Exception:
                cur_conf = None

            archive[egg_track_id] = {
                'stable_frames': max(int(prev_frames), int(stable_frames)),
                'conf_max': max(float(prev_conf) if prev_conf is not None else 0.0,
                                float(cur_conf) if cur_conf is not None else 0.0)
            }
        except Exception:
            # 归档失败不应影响主流程
            return

    def _purge_used_eggs(self, now_ts: float) -> None:
        """清理过期的“已计数蛋track_id”，避免长期增长。"""
        try:
            if float(self.egg_reuse_cooldown_s) <= 0:
                return
            expire_before = float(now_ts) - float(self.egg_reuse_cooldown_s)
            if expire_before <= 0:
                return
            for tid, ts in list(self._used_egg_track_ids.items()):
                try:
                    if float(ts) < expire_before:
                        del self._used_egg_track_ids[tid]
                except Exception:
                    try:
                        del self._used_egg_track_ids[tid]
                    except Exception:
                        pass
        except Exception:
            return

    def _is_used_egg(self, egg_track_id: int, now_ts: Optional[float] = None) -> bool:
        """判断该蛋track是否已计数（在冷却期内则忽略，避免重复匹配到后续笼位）。"""
        try:
            if float(self.egg_reuse_cooldown_s) <= 0:
                return False
        except Exception:
            return False
        if now_ts is None:
            now_ts = time.time()
        try:
            now_ts = float(now_ts)
        except Exception:
            now_ts = time.time()
        try:
            self._purge_used_eggs(now_ts)
        except Exception:
            pass
        try:
            tid = int(egg_track_id)
        except Exception:
            return False
        ts = self._used_egg_track_ids.get(tid)
        if ts is None:
            return False
        try:
            return (now_ts - float(ts)) < float(self.egg_reuse_cooldown_s)
        except Exception:
            return True

    def _mark_used_eggs(self, egg_track_ids: set) -> None:
        """将已计数的蛋track加入黑名单，并从当前匹配状态里移除。"""
        if not egg_track_ids:
            return
        now_ts = time.time()
        try:
            now_ts = float(now_ts)
        except Exception:
            now_ts = time.time()
        # 记录到“已计数”集合
        for tid0 in list(egg_track_ids):
            try:
                tid = int(tid0)
            except Exception:
                continue
            self._used_egg_track_ids[tid] = now_ts
            # 从 egg_dist 移除，避免后续被重新分配到新二维码
            try:
                if tid in self.egg_dist:
                    del self.egg_dist[tid]
            except Exception:
                pass
        # 从各二维码的 egg_dist 中移除
        try:
            for _qr_tid, qr_info in list(self.qr_dist.items()):
                m = qr_info.get('egg_dist')
                if not isinstance(m, dict):
                    continue
                for tid0 in egg_track_ids:
                    try:
                        m.pop(int(tid0), None)
                    except Exception:
                        pass
        except Exception:
            pass

    @staticmethod
    def _calc_iou(box_a, box_b) -> float:
        try:
            ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
            bx1, by1, bx2, by2 = [float(x) for x in box_b]
        except Exception:
            return 0.0
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return float(inter / denom)

    def _dedupe_stable_eggs_by_recent(self, cage_id: str, stable_ids: set, egg_boxes_map: dict) -> set:
        """按位置/时间去重，抑制 track_id 跳变导致的重复计数。"""
        if not cage_id or not stable_ids:
            return stable_ids
        try:
            window_s = float(self.egg_reid_dedupe_seconds)
        except Exception:
            window_s = 0.0
        if window_s <= 0:
            return stable_ids
        now_ts = time.time()
        recents = self._recent_cage_eggs.get(cage_id, [])
        # 清理过期
        fresh = []
        for r in recents:
            try:
                if (now_ts - float(r.get('ts', 0.0))) <= window_s:
                    fresh.append(r)
            except Exception:
                pass
        recents = fresh
        kept = set()
        for tid in list(stable_ids):
            try:
                box = egg_boxes_map.get(tid)
            except Exception:
                box = None
            if box is None:
                kept.add(tid)
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in box]
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
            except Exception:
                kept.add(tid)
                continue
            duplicate = False
            for r in recents:
                try:
                    iou = self._calc_iou(box, r.get('box'))
                except Exception:
                    iou = 0.0
                if iou >= float(self.egg_reid_iou_thresh):
                    duplicate = True
                    break
                try:
                    dx = cx - float(r.get('cx', cx))
                    dy = cy - float(r.get('cy', cy))
                    dist = (dx * dx + dy * dy) ** 0.5
                except Exception:
                    dist = 1e9
                if dist <= float(self.egg_reid_dist_ratio) * min(w, h):
                    duplicate = True
                    break
            if not duplicate:
                kept.add(tid)
                recents.append({
                    'ts': now_ts,
                    'cx': cx,
                    'cy': cy,
                    'box': [x1, y1, x2, y2],
                })
        # 限制缓存长度
        if len(recents) > 64:
            recents = recents[-64:]
        self._recent_cage_eggs[cage_id] = recents
        return kept

    def _maybe_emit_qr_snapshot(self, qr_track_id: int, qr_info: dict) -> None:
        """在二维码正对面时输出抓拍（避免频繁保存）。"""
        try:
            if not isinstance(qr_info, dict):
                return
            if qr_info.get('capture_saved'):
                return
            cage_id = qr_info.get('decode_id')
            if cage_id is None:
                return
            first_ts = qr_info.get('first_seen_ts')
            decode_ts = qr_info.get('decode_ts')
            if first_ts is None or decode_ts is None:
                return
            now_ts = time.time()
            if (now_ts - float(first_ts)) < float(self.qr_capture_delay_after_detect_s):
                return
            if (now_ts - float(decode_ts)) < float(self.qr_capture_delay_after_decode_s):
                return
            roi = qr_info.get('best_capture_roi') or qr_info.get('best_decode_roi_tight') or qr_info.get('best_decode_roi')
            if roi is None or getattr(roi, "size", 0) == 0:
                return
            record_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))
            self.qr_image_results.append({
                'id_code': str(cage_id),
                'image': roi.copy(),
                'record_time': record_time,
                'camera_idx': int(self.camera_idx) if isinstance(self.camera_idx, int) else -1,
            })
            qr_info['capture_saved'] = True
        except Exception:
            return

    def drain_qr_image_results(self):
        """返回待保存的二维码抓拍结果并清空。"""
        try:
            if not self.qr_image_results:
                return []
            out = self.qr_image_results
            self.qr_image_results = []
            return out
        except Exception:
            self.qr_image_results = []
            return []

    def _egg_conf_avg(self, egg_info: dict) -> Optional[float]:
        """计算蛋track的置信度滑窗平均（若无hist则回退到conf_max）。"""
        if not isinstance(egg_info, dict):
            return None
        hist = egg_info.get('conf_hist')
        if hist is not None:
            try:
                if len(hist) > 0:
                    return float(sum(hist) / float(len(hist)))
            except Exception:
                pass
        try:
            if egg_info.get('conf_max') is not None:
                return float(egg_info.get('conf_max'))
        except Exception:
            pass
        return None

    def _egg_conf_ok(self, egg_info: dict) -> bool:
        """是否满足计数置信度门槛。若无置信度信息则不拦截（兼容历史/异常数据）。"""
        try:
            if float(self.egg_min_conf) <= 0:
                return True
        except Exception:
            return True
        avg = self._egg_conf_avg(egg_info)
        if avg is None:
            return True
        try:
            return float(avg) >= float(self.egg_min_conf)
        except Exception:
            return True

    @staticmethod
    def _archive_value_to_frames_conf(v) -> tuple[int, Optional[float]]:
        """兼容 egg_archive 老/新格式：int 或 dict({stable_frames, conf_max})。"""
        frames = 0
        conf = None
        if isinstance(v, dict):
            try:
                frames = int(v.get('stable_frames', v.get('frames', 0)) or 0)
            except Exception:
                frames = 0
            try:
                if v.get('conf_max') is not None:
                    conf = float(v.get('conf_max'))
                elif v.get('conf') is not None:
                    conf = float(v.get('conf'))
            except Exception:
                conf = None
        else:
            try:
                frames = int(v or 0)
            except Exception:
                frames = 0
        return frames, conf

    def _project_root(self) -> str:
        """返回项目根目录（用于定位 resources/wechat 模型文件）。"""
        return get_app_root()

    def _init_wechat_detector(self, cfg: dict):
        """初始化 WeChatQRCode 解码器（比pyzbar/opencv自带更稳）。"""
        try:
            # opencv-python 不包含 wechat_qrcode 模块；只有 opencv-contrib-python 才有该接口。
            if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
                self.wechat_detector = None
                print("[QR][WARN] 当前 OpenCV 未包含 wechat_qrcode 模块（你很可能安装的是 opencv-python）。"
                      "若要启用 WeChatQRCode 提升移动/倾斜/反光场景解码成功率，请安装 opencv-contrib-python。")
                return
            qr_decode_cfg = cfg.get('qr_decode', {}) if isinstance(cfg, dict) else {}
            model_dir = qr_decode_cfg.get('wechat_model_dir', os.path.join('resources', 'wechat'))
            if not isinstance(model_dir, str) or not model_dir:
                model_dir = os.path.join('resources', 'wechat')
            # 允许相对路径（相对项目根目录）
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(self._project_root(), model_dir)
            det_proto = os.path.join(model_dir, 'detect.prototxt')
            det_model = os.path.join(model_dir, 'detect.caffemodel')
            sr_proto = os.path.join(model_dir, 'sr.prototxt')
            sr_model = os.path.join(model_dir, 'sr.caffemodel')

            # 若缺少模型文件，尝试自动下载（默认开启；失败则继续使用轻量解码器兜底）
            auto_dl = True
            try:
                auto_dl = bool(qr_decode_cfg.get('auto_download_wechat_models', True))
            except Exception:
                auto_dl = True
            try:
                # 默认超时设短一些，避免无网环境启动卡顿
                timeout_s = float(qr_decode_cfg.get('wechat_download_timeout_s', 3.0))
            except Exception:
                timeout_s = 3.0

            default_base = "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/wechat_qrcode/models"
            urls_cfg = qr_decode_cfg.get('wechat_model_urls', {}) if isinstance(qr_decode_cfg, dict) else {}
            if not isinstance(urls_cfg, dict):
                urls_cfg = {}
            urls = {
                'detect.prototxt': urls_cfg.get('detect.prototxt', f"{default_base}/detect.prototxt"),
                'sr.prototxt': urls_cfg.get('sr.prototxt', f"{default_base}/sr.prototxt"),
                'detect.caffemodel': urls_cfg.get('detect.caffemodel', f"{default_base}/detect.caffemodel"),
                'sr.caffemodel': urls_cfg.get('sr.caffemodel', f"{default_base}/sr.caffemodel"),
            }

            def _download_if_missing(dst_path: str, url: str) -> bool:
                try:
                    if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
                        return True
                except Exception:
                    pass
                if not auto_dl or not url:
                    return False
                try:
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                except Exception:
                    pass
                tmp_path = f"{dst_path}.tmp{int(time.time() * 1000)}"
                try:
                    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                        data = resp.read()
                    # 简单校验：caffemodel 通常为数MB；prototxt 也应非空
                    if not data or len(data) < 512:
                        return False
                    with open(tmp_path, 'wb') as f:
                        f.write(data)
                    os.replace(tmp_path, dst_path)
                    return True
                except Exception:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
                    return False

            # 补齐缺失文件（只要缺 caffemodel，解码成功率会明显下降）
            # 注意：仅尝试一次，避免4路摄像头各自阻塞下载。
            try:
                need_files = (det_proto, det_model, sr_proto, sr_model)
                need_download = not all(os.path.isfile(p) for p in need_files)
            except Exception:
                need_download = True
            if auto_dl and need_download:
                try:
                    with MatchingCounting._wechat_model_download_lock:
                        if not MatchingCounting._wechat_model_download_attempted:
                            MatchingCounting._wechat_model_download_attempted = True
                            _download_if_missing(det_proto, urls.get('detect.prototxt', ''))
                            _download_if_missing(sr_proto, urls.get('sr.prototxt', ''))
                            _download_if_missing(det_model, urls.get('detect.caffemodel', ''))
                            _download_if_missing(sr_model, urls.get('sr.caffemodel', ''))
                except Exception:
                    pass

            if all(os.path.isfile(p) for p in (det_proto, det_model, sr_proto, sr_model)):
                self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode(det_proto, det_model, sr_proto, sr_model)
                print(f"[QR] WeChatQRCode loaded from: {model_dir}")
            else:
                # 兜底：尝试无参初始化（部分环境可能仍可用，但一般不如显式模型稳定）
                self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
                print(f"[QR][WARN] WeChatQRCode model files not found under: {model_dir} (fallback init)")
        except Exception as e:
            self.wechat_detector = None
            print(f"[QR][WARN] WeChatQRCode init failed: {e}")

    @staticmethod
    def _pick_best_text(texts: list[str]) -> Optional[str]:
        """优先挑选符合 4位-5位 格式的笼号，其次返回第一个非空字符串。"""
        if not texts:
            return None
        pattern_full = re.compile(r'^\d{4}-\d{5}$')
        pattern_any = re.compile(r'\d{4}-\d{5}')
        for t in texts:
            if not isinstance(t, str):
                continue
            s = t.strip()
            if not s:
                continue
            if pattern_full.match(s):
                return s
            m = pattern_any.search(s)
            if m:
                return m.group(0)
        for t in texts:
            if isinstance(t, str) and t.strip():
                return t.strip()
        return None

    def _crop_with_padding(self, frame: np.ndarray, box, pad_ratio: float) -> np.ndarray:
        """按框大小自适应外扩，保留二维码静区（对解码成功率非常关键）。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = max(self.qr_min_pad_px, int(bw * float(pad_ratio)))
        pad_y = max(self.qr_min_pad_px, int(bh * float(pad_ratio)))
        xx1 = max(0, x1 - pad_x)
        yy1 = max(0, y1 - pad_y)
        xx2 = min(w, x2 + pad_x)
        yy2 = min(h, y2 + pad_y)
        return frame[yy1:yy2, xx1:xx2]

    @staticmethod
    def _resize_min_side(img: np.ndarray, target_min_side: int) -> np.ndarray:
        """将图像最短边放大到 target_min_side（不缩小），用于提高解码成功率。"""
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        mn = min(h, w)
        if mn <= 0:
            return img
        if mn >= int(target_min_side):
            return img
        scale = float(target_min_side) / float(mn)
        try:
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        except Exception:
            return img

    @staticmethod
    def _sharpen_image(img: np.ndarray) -> np.ndarray:
        """图像锐化，增强边缘以对抗运动模糊。"""
        if img is None or img.size == 0:
            return img
        try:
            # 标准锐化核
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(img, -1, kernel)
        except Exception:
            return img

    @staticmethod
    def _focus_measure(img: np.ndarray) -> float:
        """
        计算清晰度指标（Laplacian 方差），用于优先选取“更清晰”的二维码ROI。
        - 运动模糊场景下，ROI越大不一定越可解；加入清晰度可显著提升移动解码成功率。
        """
        if img is None or getattr(img, "size", 0) == 0:
            return 0.0
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        except Exception:
            gray = img
        try:
            h, w = gray.shape[:2]
            mn = min(h, w)
            # 下采样到约 220px 的最短边，降低计算开销且提升指标稳定性
            if mn > 220:
                scale = 220.0 / float(mn)
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            return float(lap.var())
        except Exception:
            return 0.0

    def _decode_with_wechat(self, bgr: np.ndarray) -> Optional[str]:
        """WeChatQRCode 解码（通常对倾斜/反光/模糊更稳）。"""
        if self.wechat_detector is None:
            return None
        try:
            # OpenCV/WeChatQRCode 在多线程下不稳定，强制串行化
            with MatchingCounting._cv_decode_lock:
                img = np.ascontiguousarray(bgr)
                texts, _points = self.wechat_detector.detectAndDecode(img)
            if isinstance(texts, (list, tuple)):
                texts = [str(t).strip() for t in texts if t is not None]
            elif isinstance(texts, str):
                texts = [texts.strip()]
            else:
                texts = []
            res = self._pick_best_text(texts)
            if res:
                # 打印日志确认 WeChat 生效
                print(f"[QR_SUCCESS] {res} by WeChatQRCode")
            return res
        except Exception:
            return None

    def _decode_with_opencv(self, img: np.ndarray) -> Optional[str]:
        """OpenCV QRCodeDetector 解码（轻量兜底）。"""
        if self.cv_qr_detector is None:
            return None
        try:
            # OpenCV QRCodeDetector 在多线程下可能堆损坏，强制串行化
            with MatchingCounting._cv_decode_lock:
                src = np.ascontiguousarray(img)
                res, _points, _ = self.cv_qr_detector.detectAndDecode(src)
            if isinstance(res, str) and res.strip():
                final = self._pick_best_text([res.strip()])
                if final:
                    print(f"[QR_SUCCESS] {final} by OpenCV")
                return final
        except Exception:
            pass
        return None

    def _decode_with_pyzbar(self, img: np.ndarray) -> Optional[str]:
        """pyzbar 解码（对部分打印码很有效，但容易受静区/模糊影响）。"""
        try:
            # 只解 QRCode，避免 zbar 的 pdf417 解码器在非目标纹理上触发大量 assert 警告
            qr_codes = decode(img, symbols=[ZBarSymbol.QRCODE])
            texts = []
            for c in qr_codes:
                try:
                    texts.append(c.data.decode('utf-8', 'ignore').strip())
                except Exception:
                    pass
            res = self._pick_best_text(texts)
            if res:
                print(f"[QR_SUCCESS] {res} by PyZbar")
            return res
        except Exception:
            return None

    def _decode_qr_robust(self, bgr: np.ndarray) -> Optional[str]:
        """
        组合解码策略：
        - WeChatQRCode（强）
        - OpenCV QRCodeDetector（轻）
        - pyzbar（兼容）
        - 锐化/CLAHE/小码多尺度放大 + 灰度/阈值/反色等预处理
        """
        if bgr is None or bgr.size == 0:
            return None

        # 0) 尝试锐化（对抗运动模糊）
        try:
            sharpened = self._sharpen_image(bgr)
            txt = self._decode_with_wechat(sharpened)
            if txt:
                return txt
        except Exception:
            sharpened = bgr

        # 1) 先用 WeChat（最稳）- 原图
        txt = self._decode_with_wechat(bgr)
        if txt:
            return txt

        # 1.5) 尝试 CLAHE 增强后给 WeChat（对抗暗光/光照不均）
        # 既然上游已经做了 Gamma 提亮，这里 CLAHE 作为备选，仅在原图失败时尝试
        # 并且为了减少卡顿，可以判断一下是否真的需要（比如均值低）
        try:
            # 简单抽样判断亮度，避免对亮图也做 CLAHE 浪费算力
            mean_v = np.mean(bgr[::4, ::4])
            if mean_v < 100:  # 只有比较暗的时候才跑 CLAHE，减少 CPU 争抢
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)
                enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
                txt = self._decode_with_wechat(enhanced_bgr)
                if txt:
                    print(f"[QR_SUCCESS] {txt} by WeChatQRCode (CLAHE Enhanced)")
                    return txt
        except Exception:
            pass

        # 2) 再用 OpenCV / pyzbar（原图）
        txt = self._decode_with_opencv(bgr) or self._decode_with_pyzbar(bgr)
        if txt:
            return txt

        # 3) 预处理（灰度/CLAHE/阈值/反色）后给 OpenCV/ZBar
        try:
            if 'gray' not in locals() or gray is None:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = None
        
        variants = []
        if gray is not None:
            variants.append(gray)
            try:
                # 刚才生成的 CLAHE 图
                if 'enhanced_gray' in locals():
                    variants.append(enhanced_gray)
            except Exception:
                pass
            try:
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                variants.append(th)
                variants.append(255 - th)
            except Exception:
                pass
            try:
                variants.append(255 - gray)
            except Exception:
                pass

        for v in variants:
            txt = self._decode_with_opencv(v) or self._decode_with_pyzbar(v)
            if txt:
                return txt

        # 4) 小码放大后再试（只对较小ROI启用，避免性能抖动）
        h, w = bgr.shape[:2]
        min_hw = min(h, w)
        if min_hw <= 260:
            for s in self.qr_max_scales:
                try:
                    s = float(s)
                except Exception:
                    continue
                if s <= 1:
                    continue
                try:
                    enlarged = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                except Exception:
                    continue
                
                # 放大后也尝试锐化
                try:
                    enlarged_sharp = self._sharpen_image(enlarged)
                except Exception:
                    enlarged_sharp = enlarged

                txt = self._decode_with_wechat(enlarged) or \
                      self._decode_with_wechat(enlarged_sharp) or \
                      self._decode_with_opencv(enlarged) or \
                      self._decode_with_pyzbar(enlarged)
                if txt:
                    return txt
                # 放大后再做一次灰度阈值尝试
                try:
                    g2 = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
                    _, th2 = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    txt = self._decode_with_opencv(th2) or self._decode_with_pyzbar(th2)
                    if txt:
                        return txt
                except Exception:
                    pass

        return None

    def _decode_qr_fast(self, bgr: np.ndarray) -> Optional[str]:
        """
        实时友好的解码策略（尽量快）：
        1) 尝试锐化后 WeChatQRCode
        2) 放大到固定最小边长后 WeChatQRCode
        3) 灰度+Otsu 后 OpenCV/pyzbar 兜底
        """
        if bgr is None or bgr.size == 0:
            return None

        # 尝试锐化（Fast模式也加一次锐化，因为开销很小）
        try:
            sharpened = self._sharpen_image(bgr)
        except Exception:
            sharpened = bgr

        txt = self._decode_with_wechat(sharpened)
        if txt:
            return txt
        
        # 如果锐化没解出来，试原图（有时候锐化过度反而坏事）
        if sharpened is not bgr:
            txt = self._decode_with_wechat(bgr)
            if txt:
                return txt

        big = self._resize_min_side(bgr, self.qr_target_min_side)
        if big is not bgr:
            txt = self._decode_with_wechat(big)
            if txt:
                return txt

        # 轻量预处理兜底
        try:
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if big.ndim == 3 else big
            # 轻微降噪
            try:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
            except Exception:
                pass
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            txt = self._decode_with_opencv(th) or self._decode_with_pyzbar(th)
            if txt:
                return txt
            # 反色再试一次（有些场景反光导致黑白反转）
            inv = 255 - th
            return self._decode_with_opencv(inv) or self._decode_with_pyzbar(inv)
        except Exception:
            return None

    def _decode_qr_multi(self, rois: list[np.ndarray], use_robust: bool = False) -> Optional[str]:
        """
        多ROI解码：
        - 先对每个ROI走 _decode_qr_fast（快）
        - 若 use_robust=True，则对第一个ROI追加一次 _decode_qr_robust（更强但更慢）
        """
        if not rois:
            return None
        # 1) 快速策略：多ROI依次尝试
        for r in rois:
            try:
                if r is None or getattr(r, "size", 0) == 0:
                    continue
                txt = self._decode_qr_fast(r)
                if txt:
                    return txt
            except Exception:
                continue
        # 2) 强力兜底：仅对首个ROI做 robust，避免性能抖动
        if use_robust:
            try:
                r0 = rois[0]
                if r0 is not None and getattr(r0, "size", 0) != 0:
                    return self._decode_qr_robust(r0)
            except Exception:
                return None
        return None

    def _maybe_schedule_decode(self, track_id: int):
        """将解码任务异步丢到后台线程，避免阻塞主匹配线程导致卡顿。"""
        if self._qr_pool is None:
            return
        if track_id not in self.qr_dist:
            return
        info = self.qr_dist[track_id]
        if info.get('decode_id') is not None:
            return

        now_ms = int(time.time() * 1000)
        # 全局节流：每路摄像头不要同时解多个二维码
        # 增加判断：如果当前有未完成任务，且该任务耗时超过1s，则强制清理，避免“卡死”在某个难解的帧上
        if self._qr_inflight:
            # 检查是否有任务“超时”
            if (now_ms - int(self._qr_last_global_decode_ms or 0)) > 1000:
                with self._qr_lock:
                    self._qr_inflight.clear()
            else:
                # 确实正在忙，且未超时，则跳过
                return

        if now_ms - int(self._qr_last_global_decode_ms or 0) < int(self.qr_global_interval_ms):
            return
        # 单track节流
        last_ms = int(info.get('last_decode_try_ms', 0) or 0)
        if now_ms - last_ms < int(self.qr_try_interval_ms):
            return
        tries = int(info.get('decode_try_count', 0) or 0)
        if tries >= int(self.qr_max_tries):
            return
        if track_id in self._qr_inflight:
            return

        # 尺寸太小基本解不出来：只缓存最优ROI，等变大后再解
        bw, bh = info.get('best_decode_box_size', (0, 0))
        try:
            bw = int(bw); bh = int(bh)
        except Exception:
            bw = bh = 0
        if min(bw, bh) > 0 and min(bw, bh) < int(self.qr_min_box_size):
            return

        # 组装多ROI候选：
        # - tight(小padding, 背景更少，通常更快/更稳)
        # - loose(大padding, 静区更足，用于tight失败时兜底)
        rois: list[np.ndarray] = []
        roi_loose = info.get('best_decode_roi', None)
        roi_tight = info.get('best_decode_roi_tight', None)
        if roi_tight is not None and getattr(roi_tight, "size", 0) != 0:
            rois.append(roi_tight.copy())
        if roi_loose is not None and getattr(roi_loose, "size", 0) != 0:
            rois.append(roi_loose.copy())
        if not rois:
            return

        # 记录尝试
        info['last_decode_try_ms'] = now_ms
        try_no = tries + 1
        info['decode_try_count'] = try_no
        self._qr_last_global_decode_ms = now_ms
        self._qr_inflight.add(track_id)

        # 决策：移动模糊/反光场景下，前几次用 fast；若多次失败或ROI很清晰/很大，则补一次 robust
        try:
            focus = float(info.get('best_decode_focus', 0.0) or 0.0)
        except Exception:
            focus = 0.0
        min_side = min(int(bw or 0), int(bh or 0)) if bw and bh else 0
        
        # 优化策略：如果已经尝试超过3次还没解出来，说明这个ROI可能就是解不出来的（运动模糊/遮挡），
        # 此时应该降低尝试频率，甚至暂停尝试，等待更清晰的帧进来更新 best_decode_roi。
        # 这里通过提高 robust 门槛来实现：只对“非常清晰”的大图启用 robust，否则只用 fast 快速试错。
        use_robust = False
        if min_side >= 60 and focus >= 60.0:
             # 大且清晰，值得一试
             use_robust = True
        elif try_no >= 5:
             # 试了很多次了，除非图真的很好，否则别浪费算力做 robust
             use_robust = False

        # 传入多ROI，优先 fast；必要时对首个ROI加 robust
        try:
            future = self._qr_pool.submit(self._decode_qr_multi, rois, use_robust)
        except Exception:
            with self._qr_lock:
                self._qr_inflight.discard(track_id)
            return

        def _done_cb(fut):
            txt = None
            try:
                txt = fut.result()
            except Exception:
                txt = None
            save_debug = False
            with self._qr_lock:
                self._qr_inflight.discard(track_id)
                # track 已被清理则忽略
                if track_id not in self.qr_dist:
                    return
                if txt and self.qr_dist[track_id].get('decode_id') is None:
                    self.qr_dist[track_id]['decode_id'] = txt
                    if self.qr_dist[track_id].get('decode_ts') is None:
                        self.qr_dist[track_id]['decode_ts'] = time.time()
                else:
                    # 解码失败：按里程碑保存少量样本，便于排查（静区/模糊/反光/尺寸）
                    try:
                        if self.qr_debug_save and (try_no in (1, 3, 5, 10, 20, 30, 40, 50) or try_no >= int(self.qr_max_tries)):
                            save_debug = True
                    except Exception:
                        save_debug = False

            if save_debug:
                try:
                    # rois[0] 为 loose ROI（通常静区更足），更适合作为失败样本
                    self._maybe_save_qr_debug(rois[0], track_id)
                except Exception:
                    pass

        try:
            future.add_done_callback(_done_cb)
        except Exception:
            # 极端情况下不支持callback则同步取一次（仍然不会抛到主线程）
            _done_cb(future)

    def _maybe_save_qr_debug(self, bgr: np.ndarray, track_id: int):
        """可选：将失败样本落盘，便于定位是静区/模糊/反光/尺寸问题。"""
        if not self.qr_debug_save:
            return
        if self._qr_debug_saved >= self.qr_debug_max:
            return
        try:
            out_dir = self.qr_debug_dir
            if not out_dir:
                out_dir = os.path.join(self.picture_recognition_path, 'qr_debug')
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, f"qr_fail_tid{track_id}_{int(time.time()*1000)}.jpg")
            cv2.imwrite(p, bgr)
            self._qr_debug_saved += 1
        except Exception:
            pass

    def match(self, results, frame, draw_flag=True):
        egg_current_detects = {}
        qr_current_detects = []

        height, width = frame.shape[:2]
        self.frame_height = height

        if width != self.width:
            self.match_center = width // 2
            self.match_range = width // 2 - self.edge_threshold
            self.edge_threshold = int(width * 0.05)
            self.width = width

        names, qr_boxes, qr_track_ids, egg_boxes, egg_track_ids, egg_confs, qr_confs = unpack_results(results)

        # 仅保留画面上方二维码（过滤旧二维码）
        if self.qr_filter_enabled:
            try:
                kept_qr_boxes = []
                kept_qr_ids = []
                kept_qr_confs = []
                top_y = float(height) * float(self.qr_top_ratio)
                for box, tid, conf in zip(qr_boxes, qr_track_ids, qr_confs):
                    try:
                        cy = (float(box[1]) + float(box[3])) / 2.0
                    except Exception:
                        continue
                    if cy <= top_y:
                        kept_qr_boxes.append(box)
                        kept_qr_ids.append(tid)
                        kept_qr_confs.append(conf)
                qr_boxes = kept_qr_boxes
                qr_track_ids = kept_qr_ids
                qr_confs = kept_qr_confs
            except Exception:
                pass

        # 先预处理图片中所有的二维码信息
        for index, (box, track_id) in enumerate(zip(qr_boxes, qr_track_ids)):
            qr_current_detect = self._process_qr_codes(frame, track_id, box, width)
            qr_current_detects.append(qr_current_detect)

        # 二维码短暂丢检时保活（使用上一帧位置，避免匹配闪烁）
        if self.qr_keep_seconds > 0:
            try:
                current_ids = {q['track_id'] for q in qr_current_detects if q is not None and 'track_id' in q}
                now_ts = time.time()
                for qr_track_id, qr_info in list(self.qr_dist.items()):
                    if qr_track_id in current_ids:
                        continue
                    qr_box = qr_info.get('qr_box')
                    if qr_box is None:
                        continue
                    try:
                        if (now_ts - float(qr_info.get('record_time', 0))) > float(self.qr_keep_seconds):
                            continue
                    except Exception:
                        continue
                    try:
                        mid = calculate_mid(qr_box, width)
                        aspect_ratio = calculate_aspect_ratio(qr_box)
                    except Exception:
                        continue
                    qr_current_detects.append({
                        'box': qr_box,
                        'qr_id': qr_info.get('qr_id', qr_track_id),
                        'track_id': qr_track_id,
                        'aspect_ratio': aspect_ratio,
                        'mid': mid
                    })
                    current_ids.add(qr_track_id)
            except Exception:
                pass

        # 基于二维码中心计算“走廊”分区与自适应横向尺度 S_px
        self.qr_context_map = self._compute_qr_context_map(qr_current_detects, width, height)
        current_qr_track_ids = {q['track_id'] for q in qr_current_detects if q is not None and 'track_id' in q}

        # 预处理图片中所有的鸭蛋信息，以蛋为核心
        now_ts = time.time()
        for index, (box, track_id, egg_conf) in enumerate(zip(egg_boxes, egg_track_ids, egg_confs)):
            try:
                if egg_conf is not None and float(egg_conf) < float(self.egg_detect_min_conf):
                    continue
            except Exception:
                pass
            # 已计数的蛋：忽略，避免被后续二维码重复匹配
            if self._is_used_egg(track_id, now_ts):
                continue
            egg_current_detects[track_id] = box
            self._process_egg_detection(box, track_id, egg_conf, qr_current_detects)

        # 查找最小平均距离进行蛋二维码匹配（加入切换滞后与稳定帧计数）
        for box, track_id, egg_conf in zip(egg_boxes, egg_track_ids, egg_confs):
            try:
                if egg_conf is not None and float(egg_conf) < float(self.egg_detect_min_conf):
                    continue
            except Exception:
                pass
            if self._is_used_egg(track_id, now_ts):
                continue
            if track_id not in self.egg_dist:
                continue
            # 一旦达到稳定阈值（可计数），锁定该蛋与笼位的归属，防止移动到下一个笼位时“把同一颗蛋又匹配过去”
            try:
                if bool(self.egg_dist[track_id].get('locked')) and self.egg_dist[track_id].get('min_qr_track_id') is not None:
                    prev_key_locked = self.egg_dist[track_id].get('min_qr_track_id')
                    # 若蛋已经明显离开上一二维码走廊，则解除锁定，允许重新匹配
                    try:
                        center_x1 = (box[0] + box[2]) / 2.0
                    except Exception:
                        center_x1 = None
                    prev_ctx = self.qr_context_map.get(prev_key_locked)
                    if prev_ctx is not None and center_x1 is not None:
                        if not (prev_ctx.get('left', -1e9) <= center_x1 <= prev_ctx.get('right', 1e9)):
                            self.egg_dist[track_id]['locked'] = False
                        else:
                            if prev_key_locked in self.qr_dist:
                                self.qr_dist[prev_key_locked]['egg_dist'][track_id] = track_id
                            continue
                    else:
                        if prev_key_locked in self.qr_dist:
                            self.qr_dist[prev_key_locked]['egg_dist'][track_id] = track_id
                        continue
            except Exception:
                pass

            if bool(self.egg_dist[track_id]['qr_dist']):
                # 选出distance最小与第二小的候选
                items = list(self.egg_dist[track_id]['qr_dist'].items())
                items.sort(key=lambda kv: kv[1]['distance'])
                min_key = items[0][0]
                min_dist = items[0][1]['distance']
                second_dist = items[1][1]['distance'] if len(items) > 1 else float('inf')

                prev_key = self.egg_dist[track_id].get('min_qr_track_id')
                try:
                    center_x1 = (box[0] + box[2]) / 2.0
                except Exception:
                    center_x1 = None
                # 计算切换门槛（基于目标候选走廊的 S_px）
                s_px = self.qr_context_map.get(min_key, {}).get('s_px', width * 0.25)
                margin_px = self.switch_margin_ratio * s_px

                # 初始化稳定/切换计数器
                if 'stable_frames' not in self.egg_dist[track_id]:
                    self.egg_dist[track_id]['stable_frames'] = 0
                if 'switch_target' not in self.egg_dist[track_id]:
                    self.egg_dist[track_id]['switch_target'] = None
                if 'switch_streak' not in self.egg_dist[track_id]:
                    self.egg_dist[track_id]['switch_streak'] = 0

                apply_switch = False
                if prev_key is None:
                    apply_switch = True
                elif min_key == prev_key:
                    # 稳定匹配，增加稳定帧计数
                    self.egg_dist[track_id]['stable_frames'] = self.egg_dist[track_id].get('stable_frames', 0) + 1
                    self.egg_dist[track_id]['switch_target'] = None
                    self.egg_dist[track_id]['switch_streak'] = 0
                    # 达到锁定阈值后锁定，避免后续被切换到其他二维码
                    try:
                        stable_frames = int(self.egg_dist[track_id].get('stable_frames', 0) or 0)
                        if stable_frames >= int(self.lock_T):
                            self.egg_dist[track_id]['locked'] = True
                    except Exception:
                        pass
                else:
                    # 上一个二维码仍在画面中时，优先保持不切换（防止重复匹配）
                    if self.switch_requires_prev_missing and prev_key in current_qr_track_ids:
                        prev_ctx = self.qr_context_map.get(prev_key)
                        min_ctx = self.qr_context_map.get(min_key)
                        allow_switch_now = False
                        try:
                            if prev_ctx and min_ctx and center_x1 is not None:
                                midline = (float(prev_ctx.get('center_x', 0.0)) + float(min_ctx.get('center_x', 0.0))) / 2.0
                                # 若蛋已经跨过中线并更接近新二维码一侧，则允许切换
                                if center_x1 >= midline:
                                    allow_switch_now = True
                        except Exception:
                            allow_switch_now = False
                        # 若新二维码明显更近，也允许切换
                        try:
                            prev_dist_fast = self.egg_dist[track_id]['qr_dist'].get(prev_key, {}).get('distance', float('inf'))
                            if min_dist < prev_dist_fast * float(self.switch_min_ratio):
                                allow_switch_now = True
                        except Exception:
                            pass
                        if not allow_switch_now:
                            self.egg_dist[track_id]['stable_frames'] = self.egg_dist[track_id].get('stable_frames', 0) + 1
                            self.egg_dist[track_id]['switch_target'] = None
                            self.egg_dist[track_id]['switch_streak'] = 0
                            try:
                                stable_frames = int(self.egg_dist[track_id].get('stable_frames', 0) or 0)
                                if stable_frames >= int(self.lock_T):
                                    self.egg_dist[track_id]['locked'] = True
                            except Exception:
                                pass
                            continue
                    # 只有当优势超过 margin_px 并且持续 N 帧才允许切换
                    # 估计 prev 的当前距离（若存在）
                    prev_dist = self.egg_dist[track_id]['qr_dist'].get(prev_key, {}).get('distance', float('inf'))
                    advantage = prev_dist - min_dist
                    # 新二维码明显更近时，直接切换
                    if min_dist < prev_dist * float(self.switch_min_ratio):
                        apply_switch = True
                    elif advantage >= margin_px:
                        if self.egg_dist[track_id]['switch_target'] == min_key:
                            self.egg_dist[track_id]['switch_streak'] += 1
                        else:
                            self.egg_dist[track_id]['switch_target'] = min_key
                            self.egg_dist[track_id]['switch_streak'] = 1
                        if self.egg_dist[track_id]['switch_streak'] >= self.switch_n:
                            apply_switch = True
                    else:
                        # 优势不足，清空切换尝试
                        self.egg_dist[track_id]['switch_target'] = None
                        self.egg_dist[track_id]['switch_streak'] = 0

                if apply_switch:
                    # 从旧二维码下移除
                    if prev_key and prev_key in self.qr_dist and track_id in self.qr_dist[prev_key]['egg_dist']:
                        del self.qr_dist[prev_key]['egg_dist'][track_id]
                    # 加到新二维码下
                    self.qr_dist[min_key]['egg_dist'][track_id] = track_id
                    self.egg_dist[track_id]['min_qr_track_id'] = min_key
                    self.egg_dist[track_id]['record_time'] = time.time()
                    # 切换后重置稳定帧计数
                    self.egg_dist[track_id]['stable_frames'] = 1
                    # 切换后解除锁定（重新积累稳定帧）
                    self.egg_dist[track_id]['locked'] = False

        # 更新二维码对应的eggBox，上传以二维码为核心
        for qr_track_id, qr_info in self.qr_dist.items():
            if qr_info['flag']:
                self.qr_dist[qr_track_id]['egg_boxs'] = egg_current_detects
                self.qr_dist[qr_track_id]['flag'] = False

        if draw_flag:
            for index, (box, track_id) in enumerate(zip(qr_boxes, qr_track_ids)):
                self._draw_rectangle(box, frame, 'qr', track_id)

            for index, (box, track_id) in enumerate(zip(egg_boxes, egg_track_ids)):
                self._draw_rectangle(box, frame, 'egg', track_id)
                try:
                    if track_id in self.egg_dist and bool(self.egg_dist[track_id].get('qr_dist')) and self.egg_dist[track_id].get('min_qr_track_id') is not None:
                        self._draw_lines(track_id, qr_track_ids, qr_boxes, box, frame)
                except Exception:
                    pass

            if len(self.qr_appear_nums_queue) != 0:
                qr_appear_nums = [self.qr_dist[temp_track_id]['appear_num'] for temp_track_id in
                                  self.qr_appear_nums_queue if temp_track_id in self.qr_dist.keys()]
                if len(qr_appear_nums) > 0:
                    qr_mean = np.mean(qr_appear_nums)
                    self.qr_diff_num = qr_mean / 2

            if len(self.egg_appear_nums_queue) != 0:
                egg_appear_nums = [self.egg_dist[temp_track_id]['appear_num'] for temp_track_id in
                                   self.egg_appear_nums_queue if temp_track_id in self.egg_dist.keys()]
                if len(egg_appear_nums) > 0:
                    egg_mean = np.mean(egg_appear_nums)
                    self.egg_diff_num = egg_mean / 1.5
            # 绘制左边缘线
            cv2.line(frame, (self.edge_threshold, 0), (self.edge_threshold, height), (0, 0, 255, 128), 1,
                     lineType=cv2.LINE_AA)

            # 绘制右边缘线
            cv2.line(frame, (width - self.edge_threshold, 0), (width - self.edge_threshold, height),
                     (0, 0, 255, 128),
                     1,
                     lineType=cv2.LINE_AA)

    def _process_qr_codes(self, frame, track_id, box, width):
        """
        :param width:
        :param frame:
        :param track_id: 跟踪过程的id，汇集了蛋与二维码
        :param box: XYXY样式的矩形框坐标
        :return: 用于计算二维码变形程度与识别二维码
        """
        if track_id not in self.qr_dist.keys():
            # 初始化二维码记录
            self.qr_dist[track_id] = {
                'qr_id': self.qr_id,
                'record_time': time.time(),
                'egg_num': 0,  # 对应在该二维码下蛋的数量
                'egg_track_ids': [],
                'flag': True,  # 匹配更新判断
                'frame': None,
                'qr_box': None,
                'min_mid': 10000,
                'egg_boxs': None,
                'aspect_ratio': None,
                'appear_num': 1,  # 该二维码出现次数
                'decode_id': None,
                'diff_num': self.qr_diff_num,
                'count': self.count,
                'egg_dist': {},
                # 归档已离开画面的蛋（egg_track_id -> stable_frames），用于停止巡检/镜头移开后仍能正确汇总
                'egg_archive': {},
                'ui_reported': False,  # 是否已向UI上报早期识别事件
                'first_seen_ts': time.time(),
                'decode_ts': None,
                'capture_saved': False,
                'best_capture_score': None,
                'best_capture_roi': None,
                'best_capture_box': None,
                'best_capture_ts': None,
            }
            self.qr_id += 1
            # 为每个 track_id 生成随机的背景颜色
            self.color_map[track_id] = generate_random_color()
            self.qr_appear_nums_queue.append(track_id)
            # 若Track线程已提前解码出笼号，这里立刻写回（关键：保证 stop 汇总能统计到蛋数）
            try:
                with self._qr_lock:
                    pending = self._pending_external_decodes.pop(track_id, None)
                    if pending and self.qr_dist[track_id].get('decode_id') is None:
                        self.qr_dist[track_id]['decode_id'] = str(pending).strip()
                        if self.qr_dist[track_id].get('decode_ts') is None:
                            self.qr_dist[track_id]['decode_ts'] = time.time()
            except Exception:
                pass
        else:
            # 更新二维码记录
            self.qr_dist[track_id]['record_time'] = time.time()
            self.qr_dist[track_id]['appear_num'] += 1
            self.qr_dist[track_id]['diff_num'] = self.qr_diff_num
            self.qr_dist[track_id]['count'] = self.count

        mid = calculate_mid(box, width)
        # 计算并更新图片形变程度
        aspect_ratio = calculate_aspect_ratio(box)
        self.qr_dist[track_id]['aspect_ratio'] = aspect_ratio
        # 仅在已解码后才积累“正对面”抓拍候选，避免额外开销
        try:
            if self.qr_dist[track_id].get('decode_id') is not None:
                now_ts = time.time()
                if self.qr_dist[track_id].get('decode_ts') is None:
                    self.qr_dist[track_id]['decode_ts'] = now_ts
                decode_ts = float(self.qr_dist[track_id].get('decode_ts') or now_ts)
                if (now_ts - decode_ts) <= float(self.qr_capture_window_s):
                    roi = self._crop_with_padding(frame, box, float(self.qr_capture_pad_ratio))
                    if roi is not None and getattr(roi, "size", 0) != 0:
                        try:
                            focus = self._focus_measure(roi)
                        except Exception:
                            focus = 0.0
                        try:
                            base = abs(float(aspect_ratio) - 1.0)
                        except Exception:
                            base = 0.0
                        # 越接近正方形、越清晰，score越低
                        try:
                            score = float(base) - min(1.5, float(focus) / 200.0)
                        except Exception:
                            score = float(base)
                        best_score = self.qr_dist[track_id].get('best_capture_score')
                        if best_score is None or score < float(best_score):
                            self.qr_dist[track_id]['best_capture_score'] = float(score)
                            self.qr_dist[track_id]['best_capture_roi'] = roi.copy()
                            self.qr_dist[track_id]['best_capture_box'] = box
                            self.qr_dist[track_id]['best_capture_ts'] = now_ts
        except Exception:
            pass

        qr_current_detect = {
            'box': box,
            'qr_id': self.qr_dist[track_id]['qr_id'],
            'track_id': track_id,
            'aspect_ratio': aspect_ratio,
            'mid': mid
        }

        # 用于保存上传图像
        if mid < self.qr_dist[track_id]['min_mid']:
            self.qr_dist[track_id]['min_mid'] = mid
            self.qr_dist[track_id]['frame'] = frame.copy()
            self.qr_dist[track_id]['qr_box'] = box
            self.qr_dist[track_id]['flag'] = True

        # 为解码挑选“最清晰/最大”的ROI样本（只存ROI，避免整帧拷贝带来的额外开销）
        try:
            bw = max(1, int(box[2]) - int(box[0]))
            bh = max(1, int(box[3]) - int(box[1]))
            area = float(bw * bh)
            # aspect_ratio越接近1越像正视二维码
            sq_penalty = abs((float(bw) / float(bh + 1e-6)) - 1.0)
            mid_norm = float(mid) / max(1.0, float(width) / 2.0)
            base_score = area / (1.0 + 2.0 * sq_penalty) / (1.0 + mid_norm * mid_norm)
            best_score = float(self.qr_dist[track_id].get('best_decode_score', -1.0) or -1.0)

            # 生成两档ROI：tight（较小padding）与 loose（较大padding）
            pr_loose = float(self.qr_pad_ratios[-1]) if self.qr_pad_ratios else 0.7
            pr_tight = float(self.qr_pad_ratios[0]) if self.qr_pad_ratios else pr_loose
            roi_loose = self._crop_with_padding(frame, box, pr_loose)
            roi_tight = None
            try:
                if pr_tight != pr_loose:
                    roi_tight = self._crop_with_padding(frame, box, pr_tight)
            except Exception:
                roi_tight = None

            # 清晰度（运动模糊场景关键）：优先用 tight ROI 评估，避免 padding 背景稀释指标
            focus = 0.0
            try:
                if roi_tight is not None and getattr(roi_tight, "size", 0) > 0:
                    focus = float(self._focus_measure(roi_tight))
                elif roi_loose is not None and getattr(roi_loose, "size", 0) > 0:
                    focus = float(self._focus_measure(roi_loose))
            except Exception:
                focus = 0.0

            # 将清晰度融入评分：最多加成 3x（focus>=300 时封顶），避免只追求“最大ROI但糊”
            try:
                focus_factor = 1.0 + min(2.0, max(0.0, focus) / 150.0)
            except Exception:
                focus_factor = 1.0
            score = float(base_score) * float(focus_factor)

            if score > best_score:
                if roi_loose is not None and getattr(roi_loose, "size", 0) > 0:
                    self.qr_dist[track_id]['best_decode_roi'] = roi_loose.copy()
                if roi_tight is not None and getattr(roi_tight, "size", 0) > 0:
                    self.qr_dist[track_id]['best_decode_roi_tight'] = roi_tight.copy()
                else:
                    self.qr_dist[track_id]['best_decode_roi_tight'] = None
                self.qr_dist[track_id]['best_decode_focus'] = float(focus)
                self.qr_dist[track_id]['best_decode_score'] = float(score)
                self.qr_dist[track_id]['best_decode_box_size'] = (bw, bh)
        except Exception:
            pass

        # 判断二维码是否识别，未识别则使用zbar与微信库进行识别
        if self.qr_dist[track_id]['decode_id'] is None:
            # 实时场景：将解码放到后台线程，避免阻塞匹配主线程导致卡顿
            try:
                self._maybe_schedule_decode(track_id)
            except Exception:
                pass
        else:
            # 已解码但可能尚未上报过早期事件的情况（补报一次，避免错过首次触发）
            if not self.qr_dist[track_id]['ui_reported']:
                try:
                    print(f"[EARLY_PREP_RETRY] track={track_id} cage_id={self.qr_dist[track_id]['decode_id']} t={int(time.time()*1000)}")
                except Exception:
                    pass
                early_data = {
                    'cage_id': self.qr_dist[track_id]['decode_id'],
                    'egg_num': 0,
                    'track_id': track_id,
                    'frame_path': None,
                    'appear_num': self.qr_dist[track_id]['appear_num'],
                    'record_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                    'early': True,
                    'delivered': False
                }
                self.early_results.append(early_data)
                self.qr_dist[track_id]['ui_reported'] = True
                # 直接上报到UI（若已设置回调）
                try:
                    if self.early_emit is not None:
                        self.early_emit([early_data])
                        early_data['delivered'] = True
                except Exception:
                    pass
            # 已识别情况下无需重复解码

        return qr_current_detect

    def _compute_qr_context_map(self, qr_current_detects, width, height):
        """基于二维码中心构建走廊分区与自适应横向尺度S_px。"""
        if not qr_current_detects:
            return {}
        # 计算中心并排序
        qr_entries = []  # [(track_id, center_x, center_y, height, box)]
        for q in qr_current_detects:
            if q is None:
                continue
            box = q['box']
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            h = max(1, box[3] - box[1])
            qr_entries.append((q['track_id'], cx, cy, h, box))
        if not qr_entries:
            return {}
        qr_entries.sort(key=lambda t: t[1])

        ctx = {}
        for idx, (tid, cx, cy, h, box) in enumerate(qr_entries):
            # 左右分界
            if idx == 0:
                left = self.edge_threshold
            else:
                left = (qr_entries[idx - 1][1] + cx) / 2.0
            if idx == len(qr_entries) - 1:
                right = width - self.edge_threshold
            else:
                right = (cx + qr_entries[idx + 1][1]) / 2.0

            # 横向邻距 S_px
            left_dx = abs(cx - qr_entries[idx - 1][1]) if idx > 0 else float('inf')
            right_dx = abs(qr_entries[idx + 1][1] - cx) if idx < len(qr_entries) - 1 else float('inf')
            s_px = min(left_dx, right_dx)
            if not np.isfinite(s_px) or s_px == float('inf'):
                s_px = max(1.0, right - left)

            ctx[tid] = {
                'left': left,
                'right': right,
                'center_x': cx,
                'center_y': cy,
                'height': h,
                's_px': s_px,
                'box': box,
            }
        return ctx

    def drain_early_results(self):
        """
        将已收集到的早期识别结果返回并清空，以便上层线程尽快推送到UI。
        该结果仅用于UI高亮，不应计入上传与蛋数统计。
        """
        # 仅返回尚未通过直接回调送达的早期结果
        undelivered = [r for r in self.early_results if not r.get('delivered')]
        # 清空缓存，已送达的丢弃，未送达的由上层接管
        self.early_results = []
        return undelivered

    def set_early_emit(self, fn):
        """设置早期结果的直接上报回调（可选）。"""
        self.early_emit = fn

    def _process_egg_detection(self, box, track_id, egg_conf, qr_current_detects):
        """
        :param box: XYXY样式的矩形框坐标
        :param track_id: 跟踪过程中的id
        :param egg_conf: 该蛋框的置信度（用于后续计数过滤）
        :param qr_current_detects: 当前帧检测到的二维码信息
        :return: 用于处理蛋数据，与蛋的上一次匹配到的所有二维码进行计算，更新匹配
        """
        # 已计数的蛋：不再参与匹配（防止被后续二维码重复匹配/重复计数）
        try:
            if self._is_used_egg(track_id, time.time()):
                return
        except Exception:
            pass

        # 置信度记录（滑窗 + 最大值）
        conf_val = None
        try:
            if egg_conf is not None:
                conf_val = float(egg_conf)
        except Exception:
            conf_val = None

        if track_id not in self.egg_dist.keys():
            # 初始化蛋检测记录
            self.egg_dist[track_id] = {
                'egg_id': self.egg_id,
                'qr_dist': {},
                'min_qr_track_id': None,
                'record_time': time.time(),
                'appear_num': 1,  # 该蛋出现次数
                'diff_num': self.egg_diff_num,
                'count': self.count,
                'stable_frames': 0,
                'locked': False,
                'conf_hist': deque([conf_val] if conf_val is not None else [], maxlen=self.egg_conf_window),
                'conf_max': float(conf_val) if conf_val is not None else 0.0
            }
            self.color_map[track_id] = generate_random_color()
            self.egg_id += 1
            self.egg_appear_nums_queue.append(track_id)
        else:
            # 更新最新记录
            self.egg_dist[track_id]['record_time'] = time.time()
            self.egg_dist[track_id]['appear_num'] += 1
            self.egg_dist[track_id]['diff_num'] = self.egg_diff_num
            self.egg_dist[track_id]['count'] = self.count
            # 更新置信度滑窗
            if conf_val is not None:
                try:
                    hist = self.egg_dist[track_id].get('conf_hist')
                    if not isinstance(hist, deque):
                        hist = deque(maxlen=self.egg_conf_window)
                        self.egg_dist[track_id]['conf_hist'] = hist
                    hist.append(conf_val)
                except Exception:
                    pass
                try:
                    prev_max = float(self.egg_dist[track_id].get('conf_max', 0.0) or 0.0)
                    self.egg_dist[track_id]['conf_max'] = max(prev_max, float(conf_val))
                except Exception:
                    pass

        egg_qr_dist = self.egg_dist[track_id]['qr_dist']
        x1_1, y1_1, x2_1, y2_1 = box[0], box[1], box[2], box[3]
        center_x1 = (x1_1 + x2_1) / 2.0
        center_y1 = (y1_1 + y2_1) / 2.0
        # 全局横向范围快速过滤（保留原逻辑）
        if center_x1 < self.match_center - self.match_range or center_x1 > self.match_center + self.match_range:
            return

        for i, qr_detect in enumerate(qr_current_detects):
            if qr_detect is None:
                continue
            mid = qr_detect['mid']
            qr_box = qr_detect['box']
            qr_id = qr_detect['qr_id']
            qr_track_id = qr_detect['track_id']

            # 走廊与阈值上下文
            ctx = self.qr_context_map.get(qr_track_id)
            if not ctx:
                continue

            left_b, right_b = ctx['left'], ctx['right']
            s_px = ctx['s_px']
            qr_cx = ctx['center_x']
            qr_cy = ctx['center_y']
            qr_h = ctx['height']

            # 蛋必须落在该二维码所属走廊内
            if not (left_b <= center_x1 <= right_b):
                continue

            # 边界忽略：离分界太近的不计
            edge_margin = self.beta_edge * s_px
            if center_x1 - left_b < edge_margin or right_b - center_x1 < edge_margin:
                continue

            # 垂直ROI约束：蛋在二维码下方带状区域
            if self.vertical_roi_enabled and self.frame_height is not None:
                y_low = qr_cy + self.vertical_roi_offset_ratio * qr_h
                y_high = self.frame_height * self.vertical_roi_bottom_ratio
                if center_y1 < y_low or center_y1 > y_high:
                    continue

            # 自适应横向阈值：|cx_egg − cx_qr| ≤ alpha_x * s_px
            x_delta = abs(center_x1 - qr_cx)
            if x_delta > self.alpha_x * s_px:
                continue

            # 横向优先代价：使用x距离作为代价
            distance = x_delta

            # 二维码id在蛋的匹配记录里
            if qr_track_id in egg_qr_dist.keys():
                distances = self.egg_dist[track_id]['qr_dist'][qr_track_id]['aspect_ratio']
                distances.append(distance)
                self.egg_dist[track_id]['qr_dist'][qr_track_id]['distance'] = np.mean(distances)
            else:
                self.egg_dist[track_id]['qr_dist'][qr_track_id] = {
                    'aspect_ratio': [distance],
                    'mid': mid,
                    'distance': distance,
                    'qr_id': qr_id
                }

    def update_and_delete_records(self):
        """
        :return: 用于汇集蛋当前帧匹配的结果，蛋大于10秒则将该蛋与二维码做最终匹配记录，后删除
        """
        del_qr_ids = []
        del_egg_ids = []
        result = []
        self.count += 1

        for egg_track_id, egg_info in list(self.egg_dist.items()):
            min_qr_track_id = self.egg_dist[egg_track_id].get('min_qr_track_id')  # 蛋对应的二维码track_id
            if self.count - egg_info.get('count', 0) > 30:
                # 关键修复：在删除蛋之前，把它最后的稳定信息归档到对应二维码里
                self._archive_egg_to_qr(egg_track_id, egg_info)

                appear_num = self.egg_dist[egg_track_id].get('appear_num', 0)
                diff_num = self.egg_dist[egg_track_id].get('diff_num', 0)
                # 删除在二维码匹配记录中出现次数较少的记录（保留原逻辑）
                try:
                    if min_qr_track_id in self.qr_dist.keys() and appear_num < diff_num:
                        if egg_track_id in self.qr_dist[min_qr_track_id].get('egg_dist', {}):
                            del self.qr_dist[min_qr_track_id]['egg_dist'][egg_track_id]
                except Exception:
                    pass
                del_egg_ids.append(egg_track_id)

        for qr_track_id, qr_info in list(self.qr_dist.items()):
            try:
                self._maybe_emit_qr_snapshot(qr_track_id, qr_info)
            except Exception:
                pass
            flag = True
            for egg_track_id in self.qr_dist[qr_track_id]['egg_dist'].keys():
                if egg_track_id in self.egg_dist.keys():
                    flag = False
            if self.count - qr_info['count'] > 30 and flag:
                # 统计“稳定蛋”：包含当前仍在跟踪的 + 已归档（已离开画面但曾稳定匹配）的
                try:
                    stable_threshold = int(self.stable_T)
                    if self.use_force_on_timeout:
                        stable_threshold = min(int(self.stable_T), int(self.force_stable_T))
                except Exception:
                    stable_threshold = int(self.stable_T)
                stable_ids = set()
                try:
                    for egg_id in self.qr_dist[qr_track_id].get('egg_dist', {}).keys():
                        if egg_id in self.egg_dist:
                            try:
                                sf = int(self.egg_dist[egg_id].get('stable_frames', 0) or 0)
                            except Exception:
                                sf = 0
                            if sf >= stable_threshold and self._egg_conf_ok(self.egg_dist.get(egg_id, {})):
                                stable_ids.add(int(egg_id))
                except Exception:
                    pass
                try:
                    archive = self.qr_dist[qr_track_id].get('egg_archive', {}) or {}
                    if isinstance(archive, dict):
                        for egg_id, v in archive.items():
                            sf, conf = self._archive_value_to_frames_conf(v)
                            # 归档蛋若带有置信度信息，则同样应用过滤阈值
                            conf_ok = True
                            try:
                                if conf is not None and float(self.egg_min_conf) > 0:
                                    conf_ok = float(conf) >= float(self.egg_min_conf)
                            except Exception:
                                conf_ok = True
                            if sf >= stable_threshold and conf_ok:
                                try:
                                    stable_ids.add(int(egg_id))
                                except Exception:
                                    pass
                except Exception:
                    pass

                cage_id = self.qr_dist[qr_track_id]['decode_id']
                egg_boxes_map = self.qr_dist[qr_track_id].get('egg_boxs', {}) or {}
                try:
                    if cage_id is not None:
                        stable_ids = self._dedupe_stable_eggs_by_recent(str(cage_id), stable_ids, egg_boxes_map)
                except Exception:
                    pass
                egg_num = len(stable_ids)
                record_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                qr_box = self.qr_dist[qr_track_id]['qr_box']

                # 条件：必须已识别到二维码，且该二维码下存在蛋
                if cage_id is not None and egg_num > 0:
                    # 先在帧上渲染二维码框与文本
                    cv2.rectangle(self.qr_dist[qr_track_id]['frame'], (qr_box[0], qr_box[1]), (qr_box[2], qr_box[3]),
                                  (0, 255, 0), 2)
                    text = f"QR:{cage_id}"
                    cv2.putText(
                        self.qr_dist[qr_track_id]['frame'],
                        text,
                        (qr_box[0] - 10, qr_box[1] - 10),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    # 再渲染被匹配到的蛋框与连接线（仅渲染稳定的蛋）
                    for egg_track_id, egg_box in self.qr_dist[qr_track_id]['egg_boxs'].items():
                        if egg_track_id in stable_ids:
                            cv2.rectangle(self.qr_dist[qr_track_id]['frame'], (egg_box[0], egg_box[1]),
                                          (egg_box[2], egg_box[3]), (0, 255, 0), 2)
                            text = f"egg"
                            cv2.putText(
                                self.qr_dist[qr_track_id]['frame'],
                                text,
                                (egg_box[0] - 10, egg_box[1] - 10),
                                cv2.FONT_HERSHEY_TRIPLEX,
                                1,
                                (255, 255, 255),
                                2,
                            )
                            center1 = ((qr_box[0] + qr_box[2]) // 2, (qr_box[1] + qr_box[3]) // 2)
                            center2 = ((egg_box[0] + egg_box[2]) // 2, (egg_box[1] + egg_box[3]) // 2)
                            # 在图像上绘制连接线
                            cv2.line(self.qr_dist[qr_track_id]['frame'], center1, center2, (0, 0, 255), 4)

                    # appear_num 阈值用于去抖，保留原有逻辑
                    appear_num = self.qr_dist[qr_track_id]['appear_num']
                    diff_num = self.qr_dist[qr_track_id]['diff_num']
                    
                    # 构建符合需求的图片路径: camera_X/match_{cage}/{wb}_{time}_{egg_num}eggs.jpg
                    frame_path = None
                    try:
                        # 解析 cage_id (e.g. "2920/14143")
                        cage_str = "unknown"
                        wb_str = "unknown"
                        safe_id = str(cage_id).replace('\\', '/')
                        if '/' in safe_id:
                            parts = safe_id.split('/')
                            if len(parts) >= 2:
                                cage_str = parts[0]
                                wb_str = parts[1]
                            else:
                                cage_str = safe_id
                        else:
                            cage_str = safe_id
                        
                        # 创建子目录 match_{cage}
                        sub_dir_name = f"match_{cage_str}"
                        save_dir = os.path.join(self.picture_recognition_path, sub_dir_name)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        
                        # 构建文件名
                        # 格式: {wb}_{time}_{egg_num}eggs.jpg
                        # time格式: YYYYMMDD_HHMMSS
                        file_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
                        file_name = f"{wb_str}_{file_time_str}_{egg_num}eggs.jpg"
                        
                        frame_path = os.path.join(save_dir, file_name)
                        
                        # 保存图片
                        if self.qr_dist[qr_track_id]['frame'] is not None:
                            cv2.imwrite(frame_path, self.qr_dist[qr_track_id]['frame'])
                    except Exception as e:
                        print(f"图片保存失败: {e}")
                        frame_path = None

                    send_data = {
                        'cage_id': cage_id,
                        'record_time': record_time,
                        'egg_num': egg_num,
                        'track_id': qr_track_id,
                        'appear_num': appear_num,
                        'frame_path': frame_path
                    }
                    result.append(send_data)
                    del_qr_ids.append(qr_track_id)
                    # 关键修复：将已计数蛋加入黑名单，避免下一个笼位再次匹配到同一颗蛋
                    try:
                        self._mark_used_eggs(stable_ids)
                    except Exception:
                        pass
                else:
                    # 无可用结果的二维码条目也应适当清理，避免长期堆积
                    # - 若 egg_num>0 但 cage_id 仍为空，可能仍有异步解码在途：给一点缓冲时间
                    try:
                        if egg_num <= 0:
                            del_qr_ids.append(qr_track_id)
                        else:
                            # 有蛋但未解码：延迟更久再清理
                            if self.count - qr_info.get('count', 0) > 90:
                                del_qr_ids.append(qr_track_id)
                    except Exception:
                        pass

        # 删除大于10秒的条目
        for del_egg_id in del_egg_ids:
            try:
                if del_egg_id in self.egg_dist:
                    del self.egg_dist[del_egg_id]
            except Exception:
                pass
        # 删除大于10秒的条目
        for del_qr_id in del_qr_ids:
            try:
                if del_qr_id in self.qr_dist:
                    del self.qr_dist[del_qr_id]
            except Exception:
                pass

        return result

    def finalize_all_results(self, force: bool = False):
        """
        在流程结束/巡检停止时，强制汇总当前仍在跟踪但尚未触发超时条件的匹配结果。
        - 默认仍要求稳定帧数 >= self.stable_T；
        - 当 force=True 时，使用更保守的阈值 force_stable_T（避免 stop 时“闪一下”的误检被计数）。
        - 汇总完成后，会清理已输出的二维码条目，避免二次统计。
        """
        del_qr_ids = []
        result = []

        for qr_track_id, qr_info in list(self.qr_dist.items()):
            try:
                cage_id = qr_info.get('decode_id')
                if cage_id is None:
                    continue

                # 统计当前二维码下已稳定的蛋：包含当前仍在跟踪的 + 已归档（已离开画面）的
                stable_ids = set()
                for egg_id in qr_info.get('egg_dist', {}).keys():
                    if egg_id in self.egg_dist:
                        try:
                            stable_frames = int(self.egg_dist[egg_id].get('stable_frames', 0) or 0)
                        except Exception:
                            stable_frames = 0
                        try:
                            conf_ok = self._egg_conf_ok(self.egg_dist.get(egg_id, {}))
                        except Exception:
                            conf_ok = True
                        if conf_ok and ((stable_frames >= self.stable_T) or (force and stable_frames >= self.force_stable_T)):
                            try:
                                stable_ids.add(int(egg_id))
                            except Exception:
                                pass
                # 归档蛋
                archive = qr_info.get('egg_archive', {}) or {}
                if isinstance(archive, dict):
                    for egg_id, v in archive.items():
                        stable_frames, conf = self._archive_value_to_frames_conf(v)
                        conf_ok = True
                        try:
                            if conf is not None and float(self.egg_min_conf) > 0:
                                conf_ok = float(conf) >= float(self.egg_min_conf)
                        except Exception:
                            conf_ok = True
                        if conf_ok and ((stable_frames >= self.stable_T) or (force and stable_frames >= self.force_stable_T)):
                            try:
                                stable_ids.add(int(egg_id))
                            except Exception:
                                pass

                try:
                    egg_boxes_map = qr_info.get('egg_boxs', {}) or {}
                    stable_ids = self._dedupe_stable_eggs_by_recent(str(cage_id), stable_ids, egg_boxes_map)
                except Exception:
                    pass
                egg_num = len(stable_ids)
                if egg_num <= 0:
                    continue

                record_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                qr_box = qr_info.get('qr_box')

                # 可选渲染与存图
                frame_path = None
                try:
                    frame = qr_info.get('frame')
                    if frame is not None and qr_box is not None:
                        # 渲染二维码框
                        cv2.rectangle(frame, (qr_box[0], qr_box[1]), (qr_box[2], qr_box[3]), (0, 255, 0), 2)
                        text = f"QR:{cage_id}"
                        cv2.putText(frame, text, (qr_box[0] - 10, qr_box[1] - 10),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

                        # 渲染匹配到的蛋框（仅渲染稳定的）
                        egg_boxes_map = qr_info.get('egg_boxs', {}) or {}
                        for egg_track_id, egg_box in egg_boxes_map.items():
                            if egg_track_id in stable_ids:
                                cv2.rectangle(frame, (egg_box[0], egg_box[1]), (egg_box[2], egg_box[3]), (0, 255, 0), 2)
                                cv2.putText(frame, 'egg', (egg_box[0] - 10, egg_box[1] - 10),
                                            cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

                        # 写图
                        # 构建符合需求的图片路径: camera_X/match_{cage}/{wb}_{time}_{egg_num}eggs.jpg
                        try:
                            # 解析 cage_id (e.g. "2920/14143")
                            cage_str = "unknown"
                            wb_str = "unknown"
                            safe_id = str(cage_id).replace('\\', '/')
                            if '/' in safe_id:
                                parts = safe_id.split('/')
                                if len(parts) >= 2:
                                    cage_str = parts[0]
                                    wb_str = parts[1]
                                else:
                                    cage_str = safe_id
                            else:
                                cage_str = safe_id
                            
                            # 创建子目录 match_{cage}
                            sub_dir_name = f"match_{cage_str}"
                            save_dir = os.path.join(self.picture_recognition_path, sub_dir_name)
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir, exist_ok=True)
                            
                            # 构建文件名
                            file_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
                            file_name = f"{wb_str}_{file_time_str}_{egg_num}eggs.jpg"
                            
                            frame_path = os.path.join(save_dir, file_name)
                            
                            cv2.imwrite(frame_path, frame)
                        except Exception as e:
                            print(f"图片保存失败: {e}")
                            frame_path = None
                except Exception:
                    # 渲染失败不影响统计
                    frame_path = None

                send_data = {
                    'cage_id': cage_id,
                    'record_time': record_time,
                    'egg_num': egg_num,
                    'track_id': qr_track_id,
                    'appear_num': qr_info.get('appear_num', 0),
                    'frame_path': frame_path
                }
                result.append(send_data)
                del_qr_ids.append(qr_track_id)
            except Exception:
                # 单条异常不影响其他条目
                continue

        # 清理已输出的二维码条目
        for del_qr_id in del_qr_ids:
            try:
                del self.qr_dist[del_qr_id]
            except Exception:
                pass

        return result

    def _draw_lines(self, track_id, track_ids, boxes, box, frame):
        index = find_index_of_id(self.egg_dist[track_id]['min_qr_track_id'], track_ids)
        if index != -1:
            # 提取矩形框的中心点坐标
            center1 = (
                (boxes[index][0] + boxes[index][2]) // 2, (boxes[index][1] + boxes[index][3]) // 2)
            center2 = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            # 在图像上绘制连接线
            cv2.line(frame, center1, center2, (0, 0, 255), 4)

    def _draw_rectangle(self, box, frame, name, track_id):
        if name == 'qr':
            if self.qr_dist[track_id]['decode_id'] is not None:
                str_display = 'Identify: ' + str(self.qr_dist[track_id]['decode_id'])
            else:
                str_display = 'wait: ' + str(self.qr_dist[track_id]['qr_id'])
        else:
            str_display = ''

        # 获取 track_id 对应的背景颜色
        bg_color = self.color_map.get(track_id, (0, 0, 0))  # 默认为黑色背景

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text = f"{name}{str_display}"

        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 2)

        # 计算背景矩形的位置
        bg_rect_x = box[0] - 10
        bg_rect_y = box[1] - 35
        bg_rect_width = text_width + 5
        bg_rect_height = text_height + 10

        # 绘制背景矩形
        cv2.rectangle(frame, (bg_rect_x, bg_rect_y), (bg_rect_x + bg_rect_width, bg_rect_y + bg_rect_height), bg_color,
                      -1)

        cv2.putText(
            frame,
            text,
            (box[0] - 10, box[1] - 10),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (255, 255, 255),
            2,
        )
