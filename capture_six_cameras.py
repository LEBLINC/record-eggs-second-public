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
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QLineEdit,
    QGridLayout,
    QMessageBox,
)


def load_config():
    """加载配置文件，可选设置摄像头索引与分辨率。"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(base_dir, "configs", "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def get_camera_indices(cfg):
    """
    优先读取 camera_0..camera_5 的 video 字段，缺省回退自身索引。
    如果配置 camera_i.enabled 为 False，将跳过该路，避免不断尝试无效设备。
    """
    indices = []
    for i in range(6):
        key = f"camera_{i}"
        if isinstance(cfg, dict) and key in cfg and isinstance(cfg[key], dict):
            if cfg[key].get("enabled") is False:
                indices.append(None)
                continue
            v = cfg[key].get("video")
            if v is not None:
                try:
                    indices.append(int(v))
                    continue
                except Exception:
                    pass
        indices.append(i)
    return indices


def ensure_output_dirs(root_dir):
    """为六路摄像头创建输出目录映射。"""
    mapping = {}
    os.makedirs(root_dir, exist_ok=True)
    for logical_idx in range(6):
        sub_dir = os.path.join(root_dir, f"camera_{logical_idx}")
        os.makedirs(sub_dir, exist_ok=True)
        mapping[logical_idx] = sub_dir
    return mapping


def build_output_path(dir_path):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(dir_path, f"{ts}.mp4")


def _try_open_once(index, backend, desired_width=None, desired_height=None, target_fps=None):
    """尝试用指定后端打开并读一帧，成功返回 (True, (w,h,fps))，否则 (False, error_str)。"""
    cap = None
    try:
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if not cap or not cap.isOpened():
            raise RuntimeError("open failed")

        # 优先减少缓冲 & 设置分辨率/帧率，避免探测阶段就失败
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if desired_width and desired_height:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(desired_width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(desired_height))
            except Exception:
                pass
        if target_fps and target_fps > 0:
            try:
                cap.set(cv2.CAP_PROP_FPS, float(target_fps))
            except Exception:
                pass
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        except Exception:
            pass

        # 预热多帧，部分设备首帧为空
        warm_ok = False
        for _ in range(6):
            ok, frame = cap.read()
            if ok and frame is not None:
                warm_ok = True
                break
            time.sleep(0.05)
        if not warm_ok:
            raise RuntimeError("read failed")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 120:
            fps = target_fps or 15.0
        return True, (w, h, float(fps))
    except Exception as e:
        return False, str(e)
    finally:
        try:
            if cap:
                cap.release()
        except Exception:
            pass


def probe_device(index, backend=None, desired_width=None, desired_height=None, target_fps=None):
    """按多后端尝试，返回 (ok, info/failure_msg)。"""
    # 将用户配置的backend优先放在列表首位
    backend_order = []
    seen = set()
    for b in [backend, cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
        if b in seen:
            continue
        seen.add(b)
        backend_order.append(b)

    last_err = ""
    for b in backend_order:
        ok, info = _try_open_once(index, b, desired_width, desired_height, target_fps)
        if ok:
            return True, info
        last_err = f"后端{b}: {info}"
    return False, last_err


class CameraRecorderThread(QThread):
    error_signal = pyqtSignal(int, str)
    status_signal = pyqtSignal(int, str)
    preview_signal = pyqtSignal(int, object)

    def __init__(
        self,
        logical_idx,
        device_index,
        output_path,
        desired_width=None,
        desired_height=None,
        backend=None,
        target_fps=None,
    ):
        super().__init__()
        self.logical_idx = logical_idx
        self.device_index = device_index
        self.output_path = output_path
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.backend = backend
        self.target_fps = target_fps
        self._running = False
        self._cap = None
        self._writer = None
        self._last_preview_t = 0.0

    def _open_capture(self):
        """多后端尝试打开摄像头，并做简短预热，避免部分设备打不开。"""
        # 用户配置的backend优先，其次DirectShow/MSMF/默认
        backends = []
        seen = set()
        for b in [self.backend, cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
            if b in seen:
                continue
            seen.add(b)
            backends.append(b)

        last_err = None
        for backend in backends:
            try:
                if backend is None:
                    self._cap = cv2.VideoCapture(self.device_index)
                else:
                    self._cap = cv2.VideoCapture(self.device_index, backend)
                if not self._cap or not self._cap.isOpened():
                    raise RuntimeError(f"后端 {backend} 打开失败")

                # 减少延迟/带宽占用
                try:
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if self.desired_width and self.desired_height:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.desired_width))
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.desired_height))
                if self.target_fps and self.target_fps > 0:
                    try:
                        self._cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                    except Exception:
                        pass
                try:
                    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
                except Exception:
                    pass

                # 读几帧预热，部分设备第一次返回空帧
                warm_ok = False
                for _ in range(8):
                    ok, frame = self._cap.read()
                    if ok and frame is not None:
                        warm_ok = True
                        break
                    time.sleep(0.05)
                if not warm_ok:
                    raise RuntimeError("预热读取失败")

                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 1 or fps > 120:
                    fps = self.target_fps or 15.0
                return w, h, float(fps)
            except Exception as e:
                last_err = e
                try:
                    if self._cap:
                        self._cap.release()
                except Exception:
                    pass
                self._cap = None
                continue
        raise RuntimeError(f"摄像头 {self.device_index} 无法打开: {last_err}")

    def _open_writer(self, frame_size, fps):
        """尝试 mp4v，失败回退 XVID。"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, frame_size)
        if writer is not None and writer.isOpened():
            return writer
        alt_path = os.path.splitext(self.output_path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(alt_path, fourcc, fps, frame_size)
        if writer is not None and writer.isOpened():
            self.output_path = alt_path
            return writer
        raise RuntimeError("无法打开视频写入器（mp4v/XVID均失败）")

    def run(self):
        self._running = True
        try:
            if self.device_index is None:
                raise RuntimeError("该路被禁用或未配置索引")
            w, h, fps = self._open_capture()
            self._writer = self._open_writer((w, h), fps)
            self.status_signal.emit(
                self.logical_idx,
                f"录制中 {w}x{h}@{fps:.1f} → {os.path.basename(self.output_path)}",
            )
            fail_count = 0
            t0 = time.time()
            frames = 0
            while self._running:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    fail_count += 1
                    if fail_count > 50:
                        raise RuntimeError("连续读取失败")
                    time.sleep(0.02)
                    continue
                fail_count = 0
                self._writer.write(frame)
                frames += 1
                now_t = time.time()
                if (now_t - self._last_preview_t) >= 0.1:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h2, w2, ch = frame_rgb.shape
                        qimg = QImage(
                            frame_rgb.data, w2, h2, ch * w2, QImage.Format_RGB888
                        ).copy()
                        self.preview_signal.emit(self.logical_idx, qimg)
                        self._last_preview_t = now_t
                    except Exception:
                        pass
                if frames % 60 == 0:
                    elapsed = time.time() - t0
                    self.status_signal.emit(
                        self.logical_idx, f"录制中 {frames} 帧 / {int(elapsed)} 秒"
                    )
        except Exception as e:
            self.error_signal.emit(self.logical_idx, f"摄像头{self.logical_idx} 异常: {e}")
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
            self.status_signal.emit(self.logical_idx, "已停止")

    def stop(self):
        self._running = False


class CaptureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("六摄像头采集")
        self.resize(900, 420)

        self.cfg = load_config()
        self.cam_indices = get_camera_indices(self.cfg)
        self.desired_width = None
        self.desired_height = None
        try:
            w = int(self.cfg.get("width", 0))
            h = int(self.cfg.get("height", 0))
            if w > 0 and h > 0:
                self.desired_width = w
                self.desired_height = h
        except Exception:
            pass

        self.threads = []
        self.output_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "capture_six"
        )

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(self.output_root)
        self.path_edit.setReadOnly(True)
        btn_choose = QPushButton("选择保存目录")
        btn_choose.clicked.connect(self._choose_dir)
        path_row.addWidget(QLabel("保存目录:"))
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(btn_choose)
        layout.addLayout(path_row)

        preview_grid = QGridLayout()
        preview_grid.setHorizontalSpacing(10)
        preview_grid.setVerticalSpacing(10)
        self.preview_labels = []
        for i in range(6):
            lbl = QLabel(f"预览 {i}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: #000; color: #FFF;")
            lbl.setMinimumWidth(200)
            lbl.setMinimumHeight(150)
            lbl.setMaximumHeight(360)
            r = i // 3
            c = i % 3
            preview_grid.addWidget(lbl, r, c)
            self.preview_labels.append(lbl)
        layout.addLayout(preview_grid)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.status_labels = []
        for i in range(6):
            grid.addWidget(QLabel(f"摄像头 {i} (设备 {self.cam_indices[i]})"), i, 0)
            l_status = QLabel("就绪")
            grid.addWidget(l_status, i, 1)
            self.status_labels.append(l_status)
        layout.addLayout(grid)

        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("开始采集")
        self.btn_start.clicked.connect(self._toggle_record)
        self.btn_stop = QPushButton("停止并退出")
        self.btn_stop.clicked.connect(self._stop_and_exit)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        layout.addLayout(ctrl)

        self.setLayout(layout)

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存目录", self.output_root)
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.threads = []
            for logical_idx in range(6):
                cam_idx = self.cam_indices[logical_idx]
                cam_cfg = self.cfg.get(f"camera_{logical_idx}", {}) if isinstance(self.cfg, dict) else {}
                enabled = cam_cfg.get("enabled", True) is not False
                backend = cam_cfg.get("backend")
                try:
                    if backend is not None:
                        backend = int(backend)
                except Exception:
                    pass
                desired_w = self.desired_width
                desired_h = self.desired_height
                target_fps = cam_cfg.get("fps", None)
                try:
                    if cam_cfg.get("width"):
                        desired_w = int(cam_cfg.get("width"))
                    if cam_cfg.get("height"):
                        desired_h = int(cam_cfg.get("height"))
                except Exception:
                    pass
                try:
                    if target_fps is not None:
                        target_fps = float(target_fps)
                        if target_fps <= 0:
                            target_fps = None
                except Exception:
                    target_fps = None

                if cam_idx is None or not enabled:
                    self.status_labels[logical_idx].setText("已跳过（配置禁用）")
                    continue

                # 启动前先探测该索引是否可用（探测失败也继续尝试录制）
                ok, info = probe_device(cam_idx, backend, desired_w, desired_h, target_fps)
                if ok:
                    w, h, fps = info
                    self.status_labels[logical_idx].setText(f"检测通过 {w}x{h}@{fps:.1f}，准备录制")
                else:
                    self.status_labels[logical_idx].setText(f"探测失败，尝试录制: {info}")

                out_dir = mapping[logical_idx]
                out_file = os.path.join(out_dir, f"{timestamp}.mp4")
                t = CameraRecorderThread(
                    logical_idx=logical_idx,
                    device_index=cam_idx,
                    output_path=out_file,
                    desired_width=desired_w,
                    desired_height=desired_h,
                    backend=backend,
                    target_fps=target_fps,
                )
                t.error_signal.connect(self._on_thread_error)
                t.status_signal.connect(self._on_thread_status)
                t.preview_signal.connect(self._on_preview)
                t.start()
                self.threads.append(t)
            self.btn_start.setText("停止采集")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"启动采集失败：{e}")

    def _stop_record(self):
        for t in self.threads:
            try:
                t.stop()
                t.wait(3000)
            except Exception:
                pass
        self.threads = []
        self.btn_start.setText("开始采集")

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
                    pix = pix.scaled(
                        lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    lbl.setPixmap(pix)
                    lbl.setText("")
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


if __name__ == "__main__":
    main()

