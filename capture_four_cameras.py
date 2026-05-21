# coding=utf-8
"""
    四摄像头同步采集脚本（独立运行）
    使用方式：
      - python capture_four_cameras.py
      - 选择保存目录，点击 “开始采集” 即可同时录制四路视频
    说明：
      - 摄像头索引优先读取 configs/config.yaml 中 camera_0..camera_3 的 video 字段
      - 支持读取 camera_i.backend/width/height/fps/enabled 等配置（与六路采集脚本一致的思路）
      - 若未配置，则回退为 [0,1,2,3]
      - 每路保存至保存目录下的 camera_i 子目录，文件名为时间戳.mp4

    修改说明（本次）：
      - 默认分辨率改为 1920x1080（当 config.yaml 未指定 width/height 时生效）
      - 增加读取全局 fps（cfg.get("fps")），camera_i.fps 仍可覆盖
"""

import os
import sys
import time
import datetime
import traceback
import yaml
import cv2

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QLineEdit, QGridLayout, QMessageBox
)

# ====== 默认采集分辨率（你要的 1080p）======
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
# 默认 fps 这里不强制（保持原逻辑），如需强制可改为 30.0
DEFAULT_FPS = None


def load_config():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(base_dir, 'configs', 'config.yaml')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _to_int(v, default=None):
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _to_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _parse_device(v, fallback_idx: int):
    """
    兼容两类配置：
    - int / 可转int字符串：按摄像头索引打开
    - 其它字符串：按URL/文件路径打开（不建议在本脚本中使用，但不强行限制）
    """
    if v is None or v == "":
        return fallback_idx
    if isinstance(v, int):
        return v
    try:
        return int(v)
    except Exception:
        return str(v)


def get_camera_settings(cfg):
    """
    返回长度为4的配置列表，每项包含：
      device_index, backend, width, height, fps, enabled, display_name, camera_controls, fourcc
    """
    settings = []
    if not isinstance(cfg, dict):
        cfg = {}

    # ====== 改动点：默认给 1080p（未配置时生效）======
    global_w = _to_int(cfg.get("width"), DEFAULT_WIDTH)
    global_h = _to_int(cfg.get("height"), DEFAULT_HEIGHT)

    # 可选：支持全局 fps（camera_i.fps 仍可覆盖）
    global_fps = _to_float(cfg.get("fps"), DEFAULT_FPS)
    if global_fps is not None and global_fps <= 0:
        global_fps = None

    global_ctrl = cfg.get("camera_controls", {}) if isinstance(cfg.get("camera_controls", {}), dict) else {}

    for i in range(4):
        cam_cfg = cfg.get(f"camera_{i}", {})
        if not isinstance(cam_cfg, dict):
            cam_cfg = {}

        enabled = cam_cfg.get("enabled", True) is not False
        device_index = _parse_device(cam_cfg.get("video", None), i)
        backend = _to_int(cam_cfg.get("backend"), None)

        desired_w = _to_int(cam_cfg.get("width"), global_w)
        desired_h = _to_int(cam_cfg.get("height"), global_h)

        target_fps = _to_float(cam_cfg.get("fps"), global_fps)
        if target_fps is not None and target_fps <= 0:
            target_fps = None

        display_name = cam_cfg.get("display_name")
        # camera_controls：允许在 camera_i.camera_controls 覆盖全局 camera_controls
        cam_ctrl = cam_cfg.get("camera_controls", {}) if isinstance(cam_cfg.get("camera_controls", {}), dict) else {}
        merged_ctrl = {}
        try:
            merged_ctrl.update(global_ctrl)
            merged_ctrl.update(cam_ctrl)
        except Exception:
            merged_ctrl = cam_ctrl or global_ctrl or {}

        # fourcc：可选（例如 MJPG/H264/YUY2/avc1）。支持字符串或列表。
        fourcc = cam_cfg.get("fourcc", None)
        settings.append(
            {
                "logical_idx": i,
                "device_index": device_index,
                "backend": backend,
                "width": desired_w,
                "height": desired_h,
                "fps": target_fps,
                "enabled": enabled,
                "display_name": display_name,
                "camera_controls": merged_ctrl,
                "fourcc": fourcc,
            }
        )
    return settings


def ensure_output_dirs(root_dir):
    mapping = {}
    os.makedirs(root_dir, exist_ok=True)
    for logical_idx in range(4):
        sub_dir = os.path.join(root_dir, f'camera_{logical_idx}')
        os.makedirs(sub_dir, exist_ok=True)
        mapping[logical_idx] = sub_dir
    return mapping


def build_output_path(dir_path):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(dir_path, f'{ts}.mp4')


class CameraRecorderThread(QThread):
    error_signal = pyqtSignal(int, str)
    status_signal = pyqtSignal(int, str)
    preview_signal = pyqtSignal(int, object)  # (logical_idx, QImage)

    def __init__(
        self,
        logical_idx,
        device_index,
        output_path,
        desired_width=None,
        desired_height=None,
        backend=None,
        target_fps=None,
        camera_controls=None,
        fourcc=None,
        startup_delay_s=0.0,
    ):
        super().__init__()
        self.logical_idx = logical_idx
        self.device_index = device_index
        self.output_path = output_path
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.backend = backend
        self.target_fps = target_fps
        self.camera_controls = camera_controls if isinstance(camera_controls, dict) else {}
        self.fourcc = fourcc
        self.startup_delay_s = float(startup_delay_s or 0.0)
        self._running = False
        self._cap = None
        self._writer = None
        self._last_preview_t = 0.0
        self._writer_size = None  # (w, h)
        self._backend_used = None
        # 记录“上一次成功打开”的backend/profile，重连时优先尝试，避免反复在高规格模式上失败
        self._preferred_backend = None
        self._preferred_profile = None  # (w, h, fps)
        # 极暗画面监测（用于自动恢复自动曝光）
        self._dark_streak = 0
        self._last_exp_restore_t = 0.0

    def _apply_camera_controls(self):
        """可选的相机控制：自动对焦/自动曝光/曝光（尽量不强行覆盖）。"""
        if not self._cap or not self.camera_controls:
            return

        # autofocus: null=不改；1=开；0=关
        af = self.camera_controls.get("autofocus", None)
        if af is not None:
            try:
                self._cap.set(cv2.CAP_PROP_AUTOFOCUS, int(af))
            except Exception:
                pass

        manage_exposure = self.camera_controls.get("manage_exposure", False) is True
        if manage_exposure:
            ae_list = self.camera_controls.get("auto_exposure", None)
            if not isinstance(ae_list, (list, tuple)):
                ae_list = [0.75]
            applied = False
            first = None
            for v in ae_list:
                if first is None:
                    first = v
                try:
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(v))
                    gv = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    if gv and abs(float(gv) - float(v)) < 0.05:
                        applied = True
                        break
                except Exception:
                    pass
            if (not applied) and (first is not None):
                try:
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(first))
                except Exception:
                    pass

        exposure = self.camera_controls.get("exposure", None)
        if exposure is not None:
            try:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            except Exception:
                pass

    def _configure_capture(self, width=None, height=None, fps=None):
        """尽量降低带宽/延迟，减少多路同时采集时黑屏/掉帧风险。"""
        if not self._cap:
            return

        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if width and height:
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            except Exception:
                pass

        if fps and fps > 0:
            try:
                self._cap.set(cv2.CAP_PROP_FPS, float(fps))
            except Exception:
                pass

        # 尝试设置编码：优先使用配置fourcc；缺省按主程序逻辑优先 MJPG（更兼容/更省带宽）
        fourcc_candidates = []
        if isinstance(self.fourcc, (list, tuple)):
            fourcc_candidates = [str(x).strip() for x in self.fourcc if x is not None]
        elif self.fourcc is not None:
            fourcc_candidates = [str(self.fourcc).strip()]
        if not fourcc_candidates:
            fourcc_candidates = ["MJPG"]
        if "MJPG" not in [c.upper() for c in fourcc_candidates]:
            fourcc_candidates.append("MJPG")
        for code in fourcc_candidates:
            try:
                code = str(code).strip()
                if len(code) != 4:
                    continue
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
            except Exception:
                pass

        self._apply_camera_controls()

    @staticmethod
    def _frame_is_valid(frame) -> bool:
        if frame is None:
            return False
        try:
            if getattr(frame, "size", 0) == 0:
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _to_bgr(frame):
        """VideoWriter 通常要求BGR三通道；兼容灰度/四通道输入。"""
        try:
            if frame is None:
                return None
            if len(frame.shape) == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception:
            return frame
        return frame

    @staticmethod
    def _sample_mean_std(frame):
        """轻量采样均值/方差（参考主程序多线程采集逻辑）。"""
        try:
            sample = frame[::32, ::32]
            m = float(sample.mean())
            s = float(sample.std())
            return m, s, sample
        except Exception:
            try:
                m = float(frame.mean()) if hasattr(frame, "mean") else 0.0
            except Exception:
                m = 0.0
            try:
                s = float(frame.std()) if hasattr(frame, "std") else 0.0
            except Exception:
                s = 0.0
            return m, s, None

    def _is_hard_black(self, frame) -> bool:
        try:
            m, s, sample = self._sample_mean_std(frame)
            hard_black = (m < 1.0 and s < 1.0)
            if (not hard_black) and (sample is not None):
                try:
                    zero_ratio = float((sample < 2).mean())
                    if zero_ratio > 0.98 and m < 3.0:
                        hard_black = True
                except Exception:
                    pass
            return bool(hard_black)
        except Exception:
            return False

    def _try_recover_from_hard_black(self):
        if not self._cap:
            return None
        try:
            for _ in range(8):
                self._cap.grab()
        except Exception:
            pass
        try:
            ret_r, frm_r = self._cap.read()
        except Exception:
            ret_r, frm_r = False, None
        if ret_r and frm_r is not None and (not self._is_hard_black(frm_r)):
            return frm_r
        return None

    def _get_cap_fourcc_str(self) -> str:
        try:
            if not self._cap:
                return ""
            fourcc_v = int(self._cap.get(cv2.CAP_PROP_FOURCC) or 0)
            s = "".join([chr((fourcc_v >> (8 * i)) & 0xFF) for i in range(4)])
            return s.strip()
        except Exception:
            return ""

    def _get_auto_exposure_try_list(self):
        ctrl = self.camera_controls if isinstance(self.camera_controls, dict) else {}
        try_list = []
        try:
            ae = ctrl.get("auto_exposure") if isinstance(ctrl, dict) else None
            if ae is not None:
                if isinstance(ae, (list, tuple)):
                    try_list.extend(list(ae))
                else:
                    try_list.append(ae)
        except Exception:
            try_list = []
        if not try_list:
            try_list = [0.75, 3.0]
        return try_list

    def _restore_auto_exposure_once(self, mean_v=None):
        if not self._cap:
            return None
        try:
            try_list = self._get_auto_exposure_try_list()
            best_v = None
            best_mean = float(mean_v) if mean_v is not None else 0.0
            best_frame = None
            for v in try_list:
                try:
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(v))
                except Exception:
                    continue
                try:
                    for _ in range(5):
                        self._cap.grab()
                except Exception:
                    pass
                ret_t, frm_t = self._cap.read()
                if not ret_t or frm_t is None:
                    continue
                try:
                    smp_t = frm_t[::32, ::32]
                    m = float(smp_t.mean())
                except Exception:
                    try:
                        m = float(frm_t.mean())
                    except Exception:
                        m = best_mean
                if m > best_mean:
                    best_mean = m
                    best_v = v
                    best_frame = frm_t

            if best_v is not None:
                try:
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(best_v))
                except Exception:
                    pass

            try:
                for _ in range(2):
                    self._cap.grab()
            except Exception:
                pass
            ret2, frm2 = self._cap.read()
            if ret2 and frm2 is not None:
                return frm2
            return best_frame
        except Exception:
            return None

    def _restore_auto_exposure_if_needed(self, frame):
        try:
            if frame is None:
                return
            m, _s, _ = self._sample_mean_std(frame)
            if m >= 8.0:
                self._dark_streak = 0
                return
            self._dark_streak += 1
            if self._dark_streak < 10:
                return
            now = time.time()
            if (now - float(self._last_exp_restore_t or 0.0)) < 5.0:
                self._dark_streak = 0
                return
            self._last_exp_restore_t = now
            self._dark_streak = 0
            self.status_signal.emit(self.logical_idx, f'画面极暗(均值≈{m:.2f})，尝试恢复自动曝光…')
            self._restore_auto_exposure_once(mean_v=m)
        except Exception:
            return

    def _open_capture(self):
        is_int_index = isinstance(self.device_index, int)

        backends = []
        seen = set()
        for b in [self.backend, self._preferred_backend, cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
            if (not is_int_index) and (b is not None):
                continue
            if b in seen:
                continue
            seen.add(b)
            backends.append(b)

        profiles = []
        prof_seen = set()

        def _add_profile(w, h, fps):
            key = (int(w) if isinstance(w, int) else w,
                   int(h) if isinstance(h, int) else h,
                   float(fps) if isinstance(fps, (int, float)) else fps)
            if key in prof_seen:
                return
            prof_seen.add(key)
            profiles.append((w, h, fps))

        w0 = int(self.desired_width) if self.desired_width else None
        h0 = int(self.desired_height) if self.desired_height else None
        fps0 = float(self.target_fps) if self.target_fps else None

        if isinstance(self._preferred_profile, (list, tuple)) and len(self._preferred_profile) == 3:
            try:
                _add_profile(self._preferred_profile[0], self._preferred_profile[1], self._preferred_profile[2])
            except Exception:
                pass

        _add_profile(w0, h0, fps0)
        _add_profile(None, None, fps0)
        if fps0 and fps0 > 15:
            _add_profile(w0, h0, 15.0)
        _add_profile(1920, 1080, fps0 if fps0 else None)
        _add_profile(1280, 720, fps0 if fps0 else None)
        _add_profile(1280, 720, 15.0)
        _add_profile(640, 480, 15.0)

        last_err = None
        for backend in backends:
            for (w_set, h_set, fps_set) in profiles:
                cap = None
                try:
                    if is_int_index:
                        cap = cv2.VideoCapture(self.device_index) if backend is None else cv2.VideoCapture(self.device_index, backend)
                    else:
                        cap = cv2.VideoCapture(str(self.device_index))
                    if not cap or not cap.isOpened():
                        raise RuntimeError(f"open failed (backend={backend})")

                    self._cap = cap
                    self._backend_used = backend
                    self._configure_capture(w_set, h_set, fps_set if fps_set is not None else self.target_fps)

                    try:
                        for _ in range(2):
                            self._cap.grab()
                    except Exception:
                        pass
                    ret, frame = self._cap.read()
                    if not ret or (not self._frame_is_valid(frame)):
                        raise RuntimeError("quick test read failed")

                    if self._is_hard_black(frame):
                        recovered = self._try_recover_from_hard_black()
                        if recovered is not None:
                            frame = recovered
                        else:
                            raise RuntimeError("hard black frame on init")

                    try:
                        m, _s, _ = self._sample_mean_std(frame)
                        if m < 8.0:
                            frm2 = self._restore_auto_exposure_once(mean_v=m)
                            if frm2 is not None:
                                frame = frm2
                    except Exception:
                        pass

                    fps = self._cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 1 or fps > 240:
                        fps = fps_set or self.target_fps or 15.0

                    self._preferred_backend = backend
                    self._preferred_profile = (w_set, h_set, fps_set)
                    return frame, float(fps)
                except Exception as e:
                    last_err = e
                    try:
                        if cap:
                            cap.release()
                    except Exception:
                        pass
                    self._cap = None
                    continue

        raise RuntimeError(f"摄像头 {self.device_index} 无法打开/读帧: {last_err}")

    def _open_writer(self, frame_size, fps):
        mp4_candidates = ["avc1", "H264", "mp4v"]
        for code in mp4_candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*code)
                writer = cv2.VideoWriter(self.output_path, fourcc, fps, frame_size)
                if writer is not None and writer.isOpened():
                    return writer
            except Exception:
                pass

        alt_path = os.path.splitext(self.output_path)[0] + ".avi"
        avi_candidates = ["XVID", "MJPG"]
        for code in avi_candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*code)
                writer = cv2.VideoWriter(alt_path, fourcc, fps, frame_size)
                if writer is not None and writer.isOpened():
                    self.output_path = alt_path
                    return writer
            except Exception:
                pass

        raise RuntimeError("无法打开视频写入器（MP4/AVI 编码器均失败）")

    def run(self):
        self._running = True
        try:
            if self.startup_delay_s and self.startup_delay_s > 0:
                self.status_signal.emit(self.logical_idx, f'等待启动 {self.startup_delay_s:.2f}s…')
                t_end = time.time() + float(self.startup_delay_s)
                while self._running and time.time() < t_end:
                    time.sleep(0.05)
                if not self._running:
                    return

            first_frame, fps = self._open_capture()
            first_frame = self._to_bgr(first_frame)
            h0, w0 = first_frame.shape[:2]
            self._writer_size = (w0, h0)
            self._writer = self._open_writer(self._writer_size, fps)
            backend_info = f"后端{self._backend_used}" if self._backend_used is not None else "后端默认"
            cap_fourcc = self._get_cap_fourcc_str()
            cap_fourcc_info = f", FOURCC={cap_fourcc}" if cap_fourcc else ""
            self.status_signal.emit(
                self.logical_idx,
                f'录制中 {w0}x{h0}@{fps:.1f} ({backend_info}{cap_fourcc_info}) → {os.path.basename(self.output_path)}',
            )

            fail_count = 0
            last_success_t = time.time()
            read_timeout_s = 30.0
            t0 = time.time()
            frames = 0

            try:
                self._writer.write(first_frame)
                frames += 1
            except Exception:
                pass

            while self._running:
                if (time.time() - last_success_t) > read_timeout_s:
                    self.status_signal.emit(self.logical_idx, '长时间无有效帧，尝试重新打开摄像头…')
                    fail_count = 6

                ok, frame = self._cap.read()
                if not ok or (not self._frame_is_valid(frame)):
                    fail_count += 1
                    if fail_count >= 3:
                        self.status_signal.emit(self.logical_idx, f'读取失败({fail_count})，恢复中…')
                    if fail_count > 5:
                        self.status_signal.emit(self.logical_idx, '读取失败过多，释放并重连摄像头…')
                        try:
                            try:
                                if self._cap is not None:
                                    self._cap.release()
                            except Exception:
                                pass
                            self._cap = None
                            t_end = time.time() + 2.0
                            while self._running and time.time() < t_end:
                                time.sleep(0.05)
                            if not self._running:
                                return

                            new_first, _fps2 = self._open_capture()
                            new_first = self._to_bgr(new_first)
                            if self._writer_size:
                                ww, hh = self._writer_size
                                if new_first.shape[1] != ww or new_first.shape[0] != hh:
                                    new_first = cv2.resize(new_first, (ww, hh), interpolation=cv2.INTER_LINEAR)
                            try:
                                self._writer.write(new_first)
                                frames += 1
                            except Exception:
                                pass
                            fail_count = 0
                            last_success_t = time.time()
                            continue
                        except Exception as e:
                            raise RuntimeError(f'读取失败且重连失败: {e}')
                    time.sleep(0.01)
                    continue

                if self._is_hard_black(frame):
                    recovered = self._try_recover_from_hard_black()
                    if recovered is not None:
                        frame = recovered
                    else:
                        fail_count += 1
                        if fail_count >= 3:
                            self.status_signal.emit(self.logical_idx, f'硬黑屏({fail_count})，恢复中…')
                        if fail_count > 5:
                            self.status_signal.emit(self.logical_idx, '硬黑屏过多，释放并重连摄像头…')
                            try:
                                try:
                                    if self._cap is not None:
                                        self._cap.release()
                                except Exception:
                                    pass
                                self._cap = None
                                t_end = time.time() + 2.0
                                while self._running and time.time() < t_end:
                                    time.sleep(0.05)
                                if not self._running:
                                    return

                                new_first, _fps2 = self._open_capture()
                                new_first = self._to_bgr(new_first)
                                if self._writer_size:
                                    ww, hh = self._writer_size
                                    if new_first.shape[1] != ww or new_first.shape[0] != hh:
                                        new_first = cv2.resize(new_first, (ww, hh), interpolation=cv2.INTER_LINEAR)
                                try:
                                    self._writer.write(new_first)
                                    frames += 1
                                except Exception:
                                    pass
                                fail_count = 0
                                last_success_t = time.time()
                                continue
                            except Exception as e:
                                raise RuntimeError(f'硬黑屏重连失败: {e}')
                        time.sleep(0.02)
                        continue

                try:
                    self._restore_auto_exposure_if_needed(frame)
                except Exception:
                    pass

                fail_count = 0
                last_success_t = time.time()
                frame = self._to_bgr(frame)
                if self._writer_size:
                    ww, hh = self._writer_size
                    if frame.shape[1] != ww or frame.shape[0] != hh:
                        frame = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_LINEAR)
                self._writer.write(frame)
                frames += 1

                now_t = time.time()
                if (now_t - self._last_preview_t) >= 0.1:
                    try:
                        preview_frame = frame
                        try:
                            max_w = 640
                            if preview_frame is not None and preview_frame.shape[1] > max_w:
                                scale = max_w / float(preview_frame.shape[1])
                                new_h = max(1, int(preview_frame.shape[0] * scale))
                                preview_frame = cv2.resize(preview_frame, (max_w, new_h), interpolation=cv2.INTER_AREA)
                        except Exception:
                            preview_frame = frame

                        frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                        h2, w2, ch = frame_rgb.shape
                        qimg = QImage(frame_rgb.data, w2, h2, ch * w2, QImage.Format_RGB888).copy()
                        self.preview_signal.emit(self.logical_idx, qimg)
                        self._last_preview_t = now_t
                    except Exception:
                        pass

                if frames % 60 == 0:
                    elapsed = time.time() - t0
                    self.status_signal.emit(self.logical_idx, f'录制中 {frames} 帧 / {int(elapsed)} 秒')

        except Exception as e:
            msg = f'摄像头{self.logical_idx} 异常: {e}'
            self.error_signal.emit(self.logical_idx, msg)
        finally:
            try:
                if self._writer is not None:
                    self._writer.release()
            except Exception:
                pass
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
            self.status_signal.emit(self.logical_idx, '已停止')

    def stop(self):
        self._running = False


class CaptureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('四摄像头采集')
        self.resize(700, 280)

        self.cfg = load_config()
        self.cam_settings = get_camera_settings(self.cfg)

        self.threads = []
        self.output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'capture')

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(self.output_root)
        self.path_edit.setReadOnly(True)
        btn_choose = QPushButton('选择保存目录')
        btn_choose.clicked.connect(self._choose_dir)
        path_row.addWidget(QLabel('保存目录:'))
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(btn_choose)
        layout.addLayout(path_row)

        preview_grid = QGridLayout()
        preview_grid.setHorizontalSpacing(10)
        preview_grid.setVerticalSpacing(10)
        self.preview_labels = []
        for i in range(4):
            lbl = QLabel(f'预览 {i}')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet('background-color: #000; color: #FFF;')
            lbl.setMinimumWidth(240)
            lbl.setMinimumHeight(180)
            lbl.setMaximumHeight(420)
            r = i // 2
            c = i % 2
            preview_grid.addWidget(lbl, r, c)
            self.preview_labels.append(lbl)
        layout.addLayout(preview_grid)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.status_labels = []
        for i in range(4):
            s = self.cam_settings[i]
            name = s.get("display_name") or str(i)
            l_title = QLabel(f'摄像头 {i} ({name}) / 设备 {s.get("device_index")}')
            l_status = QLabel('就绪')
            grid.addWidget(l_title, i, 0)
            grid.addWidget(l_status, i, 1)
            self.status_labels.append(l_status)
        layout.addLayout(grid)

        ctrl = QHBoxLayout()
        self.btn_start = QPushButton('开始采集')
        self.btn_start.clicked.connect(self._toggle_record)
        self.btn_stop = QPushButton('停止并退出')
        self.btn_stop.clicked.connect(self._stop_and_exit)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        layout.addLayout(ctrl)

        self.setLayout(layout)

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, '选择保存目录', self.output_root)
        if d:
            self.output_root = d
            self.path_edit.setText(self.output_root)

    def _toggle_record(self):
        if not self.threads:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        try:
            mapping = ensure_output_dirs(self.output_root)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.threads = []

            max_pixels = 0
            for s in self.cam_settings:
                try:
                    if s.get("enabled", True) is False:
                        continue
                    w = s.get("width") or 0
                    h = s.get("height") or 0
                    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                        max_pixels = max(max_pixels, int(w) * int(h))
                except Exception:
                    pass
            if max_pixels >= (3840 * 2160):
                base_delay = 1.0
            elif max_pixels >= (1920 * 1080):
                base_delay = 0.7
            elif max_pixels >= (1280 * 720):
                base_delay = 0.5
            else:
                base_delay = 0.35

            for logical_idx in range(4):
                s = self.cam_settings[logical_idx]
                if s.get("enabled", True) is False:
                    self.status_labels[logical_idx].setText('已跳过（配置禁用）')
                    continue
                out_dir = mapping[logical_idx]
                out_file = os.path.join(out_dir, f'{timestamp}.mp4')
                t = CameraRecorderThread(
                    logical_idx=logical_idx,
                    device_index=s.get("device_index"),
                    output_path=out_file,
                    desired_width=s.get("width"),
                    desired_height=s.get("height"),
                    backend=s.get("backend"),
                    target_fps=s.get("fps"),
                    camera_controls=s.get("camera_controls"),
                    fourcc=s.get("fourcc"),
                    startup_delay_s=float(base_delay) * float(logical_idx),
                )
                t.error_signal.connect(self._on_thread_error)
                t.status_signal.connect(self._on_thread_status)
                t.preview_signal.connect(self._on_preview)
                t.start()
                self.threads.append(t)

            if self.threads:
                self.btn_start.setText('停止采集')
            else:
                QMessageBox.information(self, '提示', '没有可用摄像头：全部被禁用或未配置。')
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, '错误', f'启动采集失败：{e}')

    def _stop_record(self):
        for t in self.threads:
            try:
                t.stop()
                t.wait(5000)
            except Exception:
                pass
        self.threads = []
        self.btn_start.setText('开始采集')

    def _stop_and_exit(self):
        self._stop_record()
        self.close()

    def _on_thread_error(self, logical_idx, msg):
        if 0 <= logical_idx < len(self.status_labels):
            self.status_labels[logical_idx].setText(msg)

    def _on_thread_status(self, logical_idx, msg):
        if 0 <= logical_idx < len(self.status_labels):
            self.status_labels[logical_idx].setText(msg)

    def _on_preview(self, logical_idx, qimg):
        try:
            if 0 <= logical_idx < len(self.preview_labels):
                lbl = self.preview_labels[logical_idx]
                pix = QPixmap.fromImage(qimg)
                if not pix.isNull():
                    lbl_w = lbl.width()
                    lbl_h = lbl.height()
                    pix = pix.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    lbl.setPixmap(pix)
                    lbl.setText('')
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self._stop_record()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = CaptureWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
