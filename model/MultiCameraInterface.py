# coding=utf-8
"""
    多摄像头接口，提供给QT访问（完整改进版）
    @project: EGGRECORDQT
    @Author：lzy
    @file： MultiCameraInterface.py
"""
import os
import subprocess
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from model.track.yoloTrack import YOLOTrack
from model.match.matchingCounting import MatchingCounting
import cv2
import queue
from PyQt5.QtGui import QImage
import threading
import time
import datetime
import numpy as np
from model.communication.SendHttp import SendHttp
from model.communication.SaveToMySQL import SaveToMySQL
from model.utils.exception import exception_handler
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import psutil

from model.utils.gpu_manager import GPUManager, GPUBatchProcessor
import re
from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol
from model.utils.preprocess import _apply_gamma_u8
from model.inference.model_factory import create_detector, create_matcher


class CameraDetector:
    """优化的摄像头检测和管理类 - 完全兼容原接口"""

    def __init__(self):
        self.available_cameras = []
        self.camera_info = {}

    def detect_available_cameras(self, max_cameras=10):
        """快速检测系统中可用的摄像头"""
        print("正在快速检测可用摄像头...")
        self.available_cameras.clear()
        self.camera_info.clear()

        # 抑制OpenCV错误日志（兼容版本）
        original_log_level = None
        try:
            original_log_level = cv2.getLogLevel()
            if hasattr(cv2, 'LOG_LEVEL_ERROR'):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
            elif hasattr(cv2, 'logging') and hasattr(cv2.logging, 'LOG_LEVEL_ERROR'):
                cv2.setLogLevel(cv2.logging.LOG_LEVEL_ERROR)
            else:
                cv2.setLogLevel(3)  # ERROR级别
        except:
            pass

        try:
            # 直接返回固定的摄像头ID列表（默认4路：左右各两路）
            fixed_camera_ids = [0, 1, 2, 3]
            print(f"使用固定摄像头ID: {fixed_camera_ids}")
            return fixed_camera_ids

        except Exception as e:
            # 静默处理异常，减少日志输出
            pass
        finally:
            # 恢复原始日志级别
            try:
                if original_log_level is not None:
                    cv2.setLogLevel(original_log_level)
            except:
                pass

        print(f"快速检测完成，找到 {len(self.available_cameras)} 个可用摄像头: {self.available_cameras}")
        return self.available_cameras

    def _get_backend_name(self, backend):
        """获取backend的名称"""
        backend_names = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_ANY: "Auto"
        }
        return backend_names.get(backend, f"Unknown({backend})")

    def get_optimal_backend(self, camera_idx):
        """获取摄像头的最佳backend"""
        if camera_idx in self.camera_info:
            return self.camera_info[camera_idx]['backend']
        return cv2.CAP_DSHOW  # 默认使用DirectShow

class ResourceMonitor:
    """资源监控类"""

    def __init__(self):
        self.last_check_time = time.time()

    def check_resources(self):
        current_time = time.time()
        if current_time - self.last_check_time > 10:  # 每10秒检查一次
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent

                if cpu_percent > 85:
                    print(f"警告: CPU使用率过高 {cpu_percent}%")
                if memory_percent > 85:
                    print(f"警告: 内存使用率过高 {memory_percent}%")

                self.last_check_time = current_time
            except Exception as e:
                print(f"资源监控异常: {e}")


class BaseThread(QThread):
    def __init__(self):
        super().__init__()
        self.run_flag = True
        self.paused = threading.Event()
        self.paused.set()  # 开始时不暂停

    def stop(self):
        self.run_flag = False
        self.resume()  # 确保在停止线程前恢复线程，防止线程阻塞

    def pause(self):
        self.paused.clear()  # 暂停线程

    def resume(self):
        self.paused.set()  # 恢复线程


class MultiCameraFrameThread(BaseThread):
    def __init__(self, camera_configs, frame_queues):
        """
        改进的摄像头读取线程
        """
        super().__init__()
        self.camera_configs = camera_configs
        self.frame_queues = frame_queues
        self.caps = [None] * len(camera_configs)
        self.cap_false_counts = [0] * len(camera_configs)
        self.camera_detector = CameraDetector()
        self.available_cameras = []

        # 摄像头健康状态监控
        self.last_successful_reads = [0] * len(camera_configs)
        self.camera_health_check_interval = 15  # 每15秒检查一次
        self.last_health_check = time.time()
        self.read_timeout = 30  # 30秒无成功读取视为摄像头掉线

        # 设置最优分辨率和帧率
        self.optimal_width = 640
        self.optimal_height = 480
        self.optimal_fps = 15
        self.fast_init = True  # 快速初始化：仅尝试一次打开，减少等待

        # 文件源播放控制（避免读文件“跑飞”导致高倍速与EOF重连）
        self._is_file_source = [False] * len(camera_configs)
        self._file_source_fps = [0.0] * len(camera_configs)
        self._last_file_read_time = [0.0] * len(camera_configs)

        # 添加资源监控
        self.resource_monitor = ResourceMonitor()
        # 捕获线程
        self.capture_workers = []
        # 设备源节流：避免采集过快导致USB带宽拥堵
        self._last_device_read_time = [0.0] * len(camera_configs)

    def _normalize_device_source(self, src):
        """将设备名/路径规范化为可被DirectShow识别的字符串。"""
        try:
            s = str(src).strip()
        except Exception:
            return None
        if not s:
            return None
        s_lower = s.lower()
        if (
            s_lower.startswith("video=")
            or s_lower.startswith("@device:")
            or s_lower.startswith("@device_pnp_")
            or s_lower.startswith("@device_pnp:")
            or s_lower.startswith("dshow:")
        ):
            return s
        return f"video={s}"

    def _is_device_source(self, src):
        try:
            s = str(src).strip().lower()
        except Exception:
            return False
        return (
            s.startswith("video=")
            or s.startswith("@device:")
            or s.startswith("@device_pnp_")
            or s.startswith("@device_pnp:")
            or s.startswith("dshow:")
        )

    def _looks_like_device_path(self, src):
        try:
            s = str(src).strip().lower()
        except Exception:
            return False
        if not s:
            return False
        if s.startswith("video="):
            s = s[6:]
        return s.startswith("@device_pnp_") or s.startswith("@device_pnp:") or s.startswith("@device:")

    def _normalize_alt_name(self, text):
        try:
            s = str(text).strip()
        except Exception:
            return ""
        if s.lower().startswith("video="):
            s = s[6:]
        s = s.strip('"').strip("'")
        # 统一转义形式，便于匹配
        while "\\\\" in s:
            s = s.replace("\\\\", "\\")
        return s.lower()

    def _find_ffmpeg_path(self):
        # 优先读取配置中的 ffmpeg_path
        try:
            for cfg in self.camera_configs:
                if isinstance(cfg, dict) and cfg.get("ffmpeg_path"):
                    return str(cfg.get("ffmpeg_path")).strip()
        except Exception:
            pass
        return "ffmpeg"

    def _list_dshow_devices_via_ffmpeg(self):
        """通过 ffmpeg -list_devices 获取 DirectShow 设备列表（含 Alternative name）"""
        ffmpeg_path = self._find_ffmpeg_path()
        try:
            result = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True,
                text=True,
                check=False
            )
        except FileNotFoundError:
            print("未找到 ffmpeg，可在配置中设置 ffmpeg_path 或加入 PATH")
            return []
        except Exception as e:
            print(f"读取 ffmpeg 设备列表失败: {e}")
            return []

        text = ""
        try:
            text = (result.stderr or "") + "\n" + (result.stdout or "")
        except Exception:
            text = ""

        devices = []
        current_name = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            name_match = re.search(r'\"(.+?)\"\s+\(video\)', line)
            if name_match:
                current_name = name_match.group(1)
                continue
            alt_match = re.search(r'Alternative name\s+\"(.+?)\"', line)
            if alt_match:
                alt = alt_match.group(1)
                devices.append({
                    "name": current_name,
                    "alt": alt,
                })
        return devices

    def _map_device_sources_to_indices(self, sources):
        """将 device_path 映射为 DirectShow 索引，避免 OpenCV 无法按名称打开。"""
        if not any(self._looks_like_device_path(s) for s in sources):
            return sources
        devices = self._list_dshow_devices_via_ffmpeg()
        if not devices:
            print("未获取到 DirectShow 设备列表，无法进行 device_path->index 映射")
            return sources

        alt_list = [d.get("alt") for d in devices if d.get("alt")]
        norm_alt_list = [self._normalize_alt_name(a) for a in alt_list]
        alt_token_list = []
        for nalt in norm_alt_list:
            token = None
            try:
                m = re.search(r'#([^#]+)#\{', nalt)
                if m:
                    token = m.group(1)
            except Exception:
                token = None
            alt_token_list.append(token or "")

        mapped = []
        for src in sources:
            if not self._looks_like_device_path(src):
                mapped.append(src)
                continue
            nsrc = self._normalize_alt_name(src)
            mapped_idx = None
            for idx, nalt in enumerate(norm_alt_list):
                if nalt == nsrc:
                    mapped_idx = idx
                    break
            if mapped_idx is None:
                # 尝试按 InstanceId 片段进行弱匹配（如 "7&15e42829&0&0000"）
                token = None
                try:
                    m = re.search(r'#([^#]+)#\{', nsrc)
                    if m:
                        token = m.group(1)
                except Exception:
                    token = None
                if token:
                    for idx, alt_token in enumerate(alt_token_list):
                        if alt_token and alt_token.lower() == token.lower():
                            mapped_idx = idx
                            break
            if mapped_idx is None:
                print(f"未找到设备路径对应索引，保留原值: {src}")
                mapped.append(src)
            else:
                print(f"设备路径映射为索引: {src} -> {mapped_idx}")
                mapped.append(mapped_idx)
        return mapped

    @exception_handler
    def init_cameras(self):
        """初始化所有摄像头（改进版）"""
        print("开始检测和初始化多摄像头...")

        # 首先根据传入配置派生可用“视频源”（支持索引或文件路径）
        try:
            derived_sources = []
            for cfg in self.camera_configs:
                # 1) 设备名/路径（稳定绑定）
                dev_src = cfg.get('device_source') or cfg.get('device_name') or cfg.get('device_path')
                dev_src = self._normalize_device_source(dev_src) if dev_src else None
                if dev_src:
                    derived_sources.append(dev_src)
                    continue

                # 2) 文件源
                vfile = cfg.get('video_file')
                if isinstance(vfile, str) and len(vfile) > 0:
                    derived_sources.append(vfile)
                    continue

                # 3) 设备索引
                idx = cfg.get('actual_video_idx', cfg.get('video'))
                if idx is not None:
                    derived_sources.append(int(idx))
            derived_sources = self._map_device_sources_to_indices(derived_sources)
            self.available_cameras = derived_sources if derived_sources else [0, 1, 2, 3, 4, 5]
        except Exception:
            self.available_cameras = [0, 1, 2, 3, 4, 5]

        if not self.available_cameras:
            print("错误：未检测到任何可用摄像头！")
            return False

        # 调整摄像头数量为实际可用数量
        actual_camera_count = min(len(self.camera_configs), len(self.available_cameras))
        print(f"将初始化 {actual_camera_count} 个摄像头，可用摄像头索引: {self.available_cameras[:actual_camera_count]}")

        # 逐个初始化，避免资源冲突
        success_count = 0

        for i in range(actual_camera_count):
            print(f"正在初始化摄像头 {i}...")
            if self.init_camera(i):
                success_count += 1
                # 去掉等待时间，加快初始化速度
            else:
                print(f"摄像头 {i} 初始化失败，跳过")

        print(f"摄像头初始化完成，成功: {success_count}/{actual_camera_count}")
        return success_count > 0

    @exception_handler
    def init_camera(self, camera_idx):
        """初始化单个摄像头（改进版）"""
        if camera_idx >= len(self.available_cameras):
            print(f"摄像头索引 {camera_idx} 超出可用范围")
            return False

        source = self.available_cameras[camera_idx]

        # 设备名/路径（DirectShow）
        if isinstance(source, str) and self._is_device_source(source):
            try:
                if self.caps[camera_idx]:
                    self.caps[camera_idx].release()
                    time.sleep(0.1)
                    self.caps[camera_idx] = None

                cam_cfg = self.camera_configs[camera_idx] if camera_idx < len(self.camera_configs) else {}
                try:
                    backend = int(cam_cfg.get('backend', cv2.CAP_DSHOW))
                except Exception:
                    backend = cv2.CAP_DSHOW
                print(f"摄像头 {camera_idx} 使用设备名/路径: {source}")

                self.caps[camera_idx] = cv2.VideoCapture(source, backend)
                if not self.caps[camera_idx].isOpened():
                    # 备选backend
                    alt_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
                    for alt_backend in alt_backends:
                        if alt_backend == backend:
                            continue
                        self.caps[camera_idx] = cv2.VideoCapture(source, alt_backend)
                        if self.caps[camera_idx].isOpened():
                            backend = alt_backend
                            print(f"摄像头 {camera_idx} 使用备选backend {self._get_backend_name(alt_backend)} 成功")
                            break

                if not self.caps[camera_idx].isOpened():
                    print(f"摄像头 {camera_idx} 设备源无法打开: {source}")
                    return False

                self._is_file_source[camera_idx] = False
                self._file_source_fps[camera_idx] = 0.0
                self._last_file_read_time[camera_idx] = 0.0

                # 配置摄像头参数
                if not self._configure_camera_params(camera_idx):
                    print(f"摄像头 {camera_idx} 参数配置失败")
                    return False

                # 快速验证
                if self._quick_camera_test(camera_idx):
                    print(f"摄像头 {camera_idx} 初始化成功（设备名/路径）")
                    self.last_successful_reads[camera_idx] = time.time()
                    return True
                print(f"摄像头 {camera_idx} 验证失败（设备名/路径）")
                return False
            except Exception as e:
                print(f"摄像头 {camera_idx} 设备源初始化异常: {e}")
                return False

        # 文件源处理：直接按路径打开
        if isinstance(source, str):
            try:
                if self.caps[camera_idx]:
                    self.caps[camera_idx].release()
                    time.sleep(0.1)
                    self.caps[camera_idx] = None
                self.caps[camera_idx] = cv2.VideoCapture(source)
                if not self.caps[camera_idx].isOpened():
                    print(f"摄像头 {camera_idx} 文件源无法打开: {source}")
                    return False
                # 记录文件源信息
                self._is_file_source[camera_idx] = True
                try:
                    fps = float(self.caps[camera_idx].get(cv2.CAP_PROP_FPS) or 0.0)
                    self._file_source_fps[camera_idx] = fps if fps > 0 else 0.0
                except Exception:
                    self._file_source_fps[camera_idx] = 0.0
                self._last_file_read_time[camera_idx] = 0.0

                # 快速验证读取一帧
                ret, frame = self.caps[camera_idx].read()
                if not ret or frame is None:
                    print(f"摄像头 {camera_idx} 文件源读帧失败: {source}")
                    return False
                # 回退一帧到起始（如有需要）
                try:
                    self.caps[camera_idx].set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception:
                    pass
                print(f"摄像头 {camera_idx} 使用文件源初始化成功: {source}")
                self.last_successful_reads[camera_idx] = time.time()
                return True
            except Exception as e:
                print(f"摄像头 {camera_idx} 文件源初始化异常: {e}")
                return False

        actual_camera_idx = source
        print(f"初始化摄像头 {camera_idx} (实际设备索引: {actual_camera_idx})")
        # 设备源
        self._is_file_source[camera_idx] = False

        # 最多尝试3次（若启用fast_init仅1次）
        max_attempts = 1 if self.fast_init else 3
        for attempt in range(max_attempts):
            success = False
            try:
                # 先释放现有连接
                if self.caps[camera_idx]:
                    self.caps[camera_idx].release()
                    time.sleep(0.1)  # 减少等待时间
                    self.caps[camera_idx] = None

                # 获取最佳backend
                backend = self.camera_detector.get_optimal_backend(actual_camera_idx)
                print(f"摄像头 {camera_idx} 尝试使用backend: {self._get_backend_name(backend)} (第{attempt + 1}次)")

                # 创建VideoCapture对象（设备源）
                self.caps[camera_idx] = cv2.VideoCapture(actual_camera_idx, backend)

                if not self.caps[camera_idx].isOpened():
                    print(f"摄像头 {camera_idx} 无法打开，尝试其他backend...")

                    # 直接尝试DirectShow作为备选
                    alternative_backends = [cv2.CAP_DSHOW] if backend != cv2.CAP_DSHOW else [cv2.CAP_MSMF]

                    for alt_backend in alternative_backends:
                        self.caps[camera_idx].release()
                        time.sleep(0.1)  # 减少等待时间
                        self.caps[camera_idx] = cv2.VideoCapture(actual_camera_idx, alt_backend)
                        if self.caps[camera_idx].isOpened():
                            print(f"摄像头 {camera_idx} 使用备选backend {self._get_backend_name(alt_backend)} 成功")
                            backend = alt_backend
                            break

                    if not self.caps[camera_idx].isOpened():
                        continue

                # 配置摄像头参数
                success = self._configure_camera_params(camera_idx)
                if not success:
                    print(f"摄像头 {camera_idx} 参数配置失败")
                    continue

                # 快速验证摄像头可用性（只读取一帧）
                success = self._quick_camera_test(camera_idx)
                if success:
                    print(f"摄像头 {camera_idx} 初始化成功")
                    self.last_successful_reads[camera_idx] = time.time()
                    return True
                else:
                    print(f"摄像头 {camera_idx} 验证失败，尝试重新初始化...")

            except Exception as e:
                print(f"摄像头 {camera_idx} 初始化异常 (尝试 {attempt + 1}): {e}")

            finally:
                if attempt < 2 and not success:  # 如果不是最后一次尝试且失败了
                    if self.caps[camera_idx]:
                        self.caps[camera_idx].release()
                        self.caps[camera_idx] = None
                    time.sleep(0.2)  # 减少重试等待时间

        # 所有尝试都失败
        print(f"摄像头 {camera_idx} 初始化完全失败")
        if self.caps[camera_idx]:
            self.caps[camera_idx].release()
            self.caps[camera_idx] = None
        return False

    def _get_backend_name(self, backend):
        """获取backend的名称"""
        backend_names = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_ANY: "Auto",
            700: "DirectShow",
            1400: "MSMF"
        }
        return backend_names.get(backend, f"Unknown({backend})")

    def _configure_camera_params(self, camera_idx):
        """配置摄像头参数"""
        try:
            cap = self.caps[camera_idx]
            cam_cfg = self.camera_configs[camera_idx] if camera_idx < len(self.camera_configs) else {}

            # 目标分辨率/帧率：优先使用每路摄像头配置 camera_i.width/height/fps
            # 说明：二维码解码对像素密度很敏感，640x480 往往“能检出但解不出来”。
            # 因此这里正式启用配置文件里的 width/height/fps（之前版本固定 640x480）。
            try:
                target_w = int(cam_cfg.get("width", self.optimal_width) or self.optimal_width)
            except Exception:
                target_w = int(self.optimal_width)
            try:
                target_h = int(cam_cfg.get("height", self.optimal_height) or self.optimal_height)
            except Exception:
                target_h = int(self.optimal_height)
            try:
                target_fps = float(cam_cfg.get("fps", self.optimal_fps) or self.optimal_fps)
            except Exception:
                target_fps = float(self.optimal_fps)

            # 设置缓冲区大小为1（减少延迟）
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

            # 设置帧率
            if target_fps > 0:
                cap.set(cv2.CAP_PROP_FPS, target_fps)

            # 尝试设置MJPEG编码（更兼容）
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except:
                pass  # 如果不支持就跳过

            # 相机曝光与亮度控制交给设备默认自动模式，不做程序端干预

            # 获取实际设置的参数
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"摄像头 {camera_idx} 实际参数: {actual_width}x{actual_height}@{actual_fps}fps")
            # 提示：若二维码解码长期为0，优先把摄像头分辨率提高到 1280x720 或 1920x1080
            try:
                if actual_width <= 640 and actual_height <= 480:
                    print(f"[QR][HINT] 摄像头 {camera_idx} 当前分辨率较低({actual_width}x{actual_height})，二维码可能检测到但解码困难。"
                          f"建议在 configs/config.yaml 将 camera_{camera_idx}.width/height 提升到 1280x720 或 1920x1080。")
            except Exception:
                pass
            return True

        except Exception as e:
            print(f"摄像头 {camera_idx} 参数配置异常: {e}")
            return False

    def _quick_camera_test(self, camera_idx):
        """快速验证摄像头可用性（只读取一帧）"""
        try:
            cap = self.caps[camera_idx]
            
            # 清空缓冲区（只清空2次，减少等待）
            for _ in range(2):
                cap.grab()

            # 只读取一帧验证可用性
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"摄像头 {camera_idx} 验证失败，无法读取帧")
                return False

            print(f"摄像头 {camera_idx} 验证成功，成功读取 1 帧")
            return True

        except Exception as e:
            print(f"摄像头 {camera_idx} 验证异常: {e}")
            return False

    @exception_handler
    def reset_problematic_camera(self, camera_idx):
        """重置有问题的摄像头"""
        print(f"尝试重置摄像头 {camera_idx}")

        # 释放摄像头
        if self.caps[camera_idx]:
            try:
                self.caps[camera_idx].release()
            except:
                pass
            self.caps[camera_idx] = None

        # 等待系统释放资源（减少等待时间）
        time.sleep(0.5)

        # 重新初始化
        return self.init_camera(camera_idx)

    @exception_handler
    def handle_capture_failure(self, camera_idx):
        """处理摄像头捕获失败的情况"""
        self.cap_false_counts[camera_idx] += 1
        if self.cap_false_counts[camera_idx] > 5:  # 增加容错次数
            print(f"摄像头 {camera_idx} 获取失败次数过多，尝试重新初始化")
            self.release_camera(camera_idx)
            time.sleep(2.0)  # 增加等待时间，避免频繁重连导致驱动崩溃
            self.init_camera(camera_idx)
            self.cap_false_counts[camera_idx] = 0

    @exception_handler
    def release_camera(self, camera_idx):
        """释放指定摄像头资源"""
        if self.caps[camera_idx]:
            try:
                self.caps[camera_idx].release()
                time.sleep(0.5)  # 增加等待时间，确保资源完全释放
            except Exception as e:
                print(f"释放摄像头 {camera_idx} 异常: {e}")
            finally:
                self.caps[camera_idx] = None

    @exception_handler
    def release_all_cameras(self):
        """释放所有摄像头资源"""
        for i in range(len(self.caps)):
            self.release_camera(i)

    @exception_handler
    def check_camera_health(self):
        """检查所有摄像头的健康状态"""
        current_time = time.time()

        if current_time - self.last_health_check < self.camera_health_check_interval:
            return

        self.last_health_check = current_time

        for camera_idx in range(len(self.caps)):
            if self.caps[camera_idx] is None:
                continue

            # 检查摄像头是否长时间未成功读取
            if current_time - self.last_successful_reads[camera_idx] > self.read_timeout:
                print(f"摄像头 {camera_idx} 已 {int(current_time - self.last_successful_reads[camera_idx])} 秒未成功读取，尝试重新初始化")
                self.release_camera(camera_idx)
                time.sleep(1)
                self.init_camera(camera_idx)

    @exception_handler
    def run(self):
        """改进的主运行函数"""
        print("多摄像头帧读取线程启动")

        # 初始化摄像头
        if not self.init_cameras():
            print("摄像头初始化失败，线程退出")
            return

        # 启动每路独立捕获线程，避免串行读取导致的卡顿
        self._start_capture_workers()

        while self.run_flag:
            try:
                self.paused.wait()

                # 监控系统资源
                self.resource_monitor.check_resources()

                # 健康检查（掉线重连）
                self.check_camera_health()

                # 主循环仅做健康监控，减轻阻塞
                time.sleep(0.1)

            except Exception as e:
                print(f"摄像头读取主循环异常: {e}")
                time.sleep(0.1)

        # 停止子线程并释放资源
        self._stop_capture_workers()
        self.release_all_cameras()
        print("多摄像头帧读取线程退出")

    def _start_capture_workers(self):
        """为每个摄像头启动独立捕获线程"""
        # 设置线程优先级（仅Windows有效，尝试提高USB读取优先级）
        try:
            import win32api, win32process, win32con
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
        except Exception:
            pass

        for idx in range(len(self.camera_configs)):
            t = threading.Thread(target=self._capture_loop, args=(idx,), name=f"cam-cap-{idx}", daemon=True)
            t.start()
            self.capture_workers.append(t)

    def _stop_capture_workers(self):
        """停止捕获线程"""
        # 线程会在 run_flag 关闭后退出，这里简单等待一下
        for t in self.capture_workers:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        self.capture_workers = []

    def _capture_loop(self, camera_idx):
        """单路摄像头捕获循环（独立线程）"""
        while self.run_flag:
            self.paused.wait()
            try:
                cap = self.caps[camera_idx] if camera_idx < len(self.caps) else None
                if cap is None or not cap.isOpened():
                    time.sleep(0.05)
                    continue

                # 文件源节流：按视频FPS或配置FPS播放，避免“高倍速快进”
                if self._is_file_source[camera_idx]:
                    try:
                        cam_cfg = self.camera_configs[camera_idx] if camera_idx < len(self.camera_configs) else {}
                    except Exception:
                        cam_cfg = {}
                    # 配置优先：video_playback_fps>0 则固定按该fps播放
                    try:
                        cfg_fps = float(cam_cfg.get("video_playback_fps", 0.0) or 0.0) if isinstance(cam_cfg, dict) else 0.0
                    except Exception:
                        cfg_fps = 0.0
                    src_fps = float(self._file_source_fps[camera_idx] or 0.0)
                    target_fps = cfg_fps if cfg_fps > 0 else (src_fps if src_fps > 0 else float(self.optimal_fps))
                    target_fps = max(1.0, float(target_fps))
                    interval = 1.0 / target_fps
                    last_t = float(self._last_file_read_time[camera_idx] or 0.0)
                    now = time.time()
                    if last_t > 0 and (now - last_t) < interval:
                        time.sleep(max(0.0, interval - (now - last_t)))

                # 设备源节流：按配置FPS采样，减少USB带宽压力
                if not self._is_file_source[camera_idx]:
                    try:
                        cam_cfg = self.camera_configs[camera_idx] if camera_idx < len(self.camera_configs) else {}
                    except Exception:
                        cam_cfg = {}
                    try:
                        cfg_fps = float(cam_cfg.get("fps", 0.0) or 0.0) if isinstance(cam_cfg, dict) else 0.0
                    except Exception:
                        cfg_fps = 0.0
                    target_fps = cfg_fps if cfg_fps > 0 else float(self.optimal_fps)
                    target_fps = max(1.0, float(target_fps))
                    interval = 1.0 / target_fps
                    last_t = float(self._last_device_read_time[camera_idx] or 0.0)
                    now = time.time()
                    if last_t > 0 and (now - last_t) < interval:
                        time.sleep(max(0.0, interval - (now - last_t)))

                ret, frame = cap.read()
                if not ret or frame is None:
                    # 文件源：通常是EOF，直接回到开头（默认循环播放），避免反复重连刷屏
                    if self._is_file_source[camera_idx]:
                        try:
                            cam_cfg = self.camera_configs[camera_idx] if camera_idx < len(self.camera_configs) else {}
                        except Exception:
                            cam_cfg = {}
                        loop_play = True
                        if isinstance(cam_cfg, dict) and "video_loop" in cam_cfg:
                            try:
                                loop_play = bool(cam_cfg.get("video_loop", True))
                            except Exception:
                                loop_play = True
                        if loop_play:
                            try:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            except Exception:
                                pass
                            time.sleep(0.02)
                            continue
                        # 不循环则轻微等待，避免忙等
                        time.sleep(0.1)
                        continue

                    self.handle_capture_failure(camera_idx)
                    time.sleep(0.01)
                    continue

                # “硬黑屏”检测：不是“很暗”，而是帧几乎全0/全黑（常见于带宽/驱动抖动/采集异常）
                # 处理策略：
                # 1) 先快速抓取若干帧尝试恢复（不重连）
                # 2) 若仍为硬黑，则按失败处理并触发重连逻辑
                try:
                    sample = frame[::32, ::32]
                    m = float(sample.mean())
                    s = float(sample.std())
                    hard_black = (m < 1.0 and s < 1.0)
                except Exception:
                    hard_black = False

                if hard_black:
                    recovered = False
                    try:
                        for _ in range(8):
                            cap.grab()
                        ret_r, frm_r = cap.read()
                        if ret_r and frm_r is not None:
                            try:
                                smp_r = frm_r[::32, ::32]
                                mr = float(smp_r.mean())
                                sr = float(smp_r.std())
                                if not (mr < 1.0 and sr < 1.0):
                                    frame = frm_r
                                    recovered = True
                            except Exception:
                                recovered = True
                                frame = frm_r
                    except Exception:
                        recovered = False

                    if not recovered:
                        try:
                            print(f"[CAM][WARN] 摄像头 {camera_idx} 出现硬黑屏帧，触发重连/降载建议：检查USB带宽/供电，必要时降低fps或分辨率")
                        except Exception:
                            pass
                        self.handle_capture_failure(camera_idx)
                        time.sleep(0.02)
                        continue

                # 针对下层摄像头（或配置了 preprocess_gamma 的摄像头）进行强制提亮
                # 这样 YOLO 检出率、UI 观感、解码成功率都会提升
                try:
                    cam_cfg = self.camera_configs[camera_idx] if camera_idx < len(self.camera_configs) else {}
                    gamma = cam_cfg.get('preprocess_gamma')
                    if gamma is not None:
                        frame = _apply_gamma_u8(frame, float(gamma))
                except Exception:
                    pass

                # 更新最后成功读取时间
                self.last_successful_reads[camera_idx] = time.time()
                self.cap_false_counts[camera_idx] = 0
                if self._is_file_source[camera_idx]:
                    self._last_file_read_time[camera_idx] = time.time()
                else:
                    self._last_device_read_time[camera_idx] = time.time()

                # 入队（非阻塞，满则丢旧保新）
                try:
                    if self.frame_queues[camera_idx].full():
                        try:
                            self.frame_queues[camera_idx].get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queues[camera_idx].put_nowait(frame)
                except Exception:
                    pass
            except Exception as e:
                print(f"摄像头 {camera_idx} 捕获异常: {e}")
                time.sleep(0.02)

    @exception_handler
    def get_frame(self, camera_idx):
        """从指定摄像头读取帧"""
        try:
            if not self.caps[camera_idx] or not self.caps[camera_idx].isOpened():
                return

            ret, frame = self.caps[camera_idx].read()
            if not ret or frame is None:
                self.handle_capture_failure(camera_idx)
                return

            # 更新最后成功读取时间
            self.last_successful_reads[camera_idx] = time.time()

            # 重置失败计数
            self.cap_false_counts[camera_idx] = 0

            # 尝试向队列中放入帧，非阻塞方式
            try:
                if not self.frame_queues[camera_idx].full():
                    self.frame_queues[camera_idx].put_nowait(frame)
                else:
                    # 队列已满，清理旧帧
                    try:
                        self.frame_queues[camera_idx].get_nowait()
                        self.frame_queues[camera_idx].put_nowait(frame)
                    except queue.Empty:
                        pass
            except Exception as e:
                pass  # 忽略队列异常

        except Exception as e:
            print(f"摄像头 {camera_idx} 帧获取异常: {e}")


class MultiCameraTrackThread(BaseThread):
    def __init__(self, yolo_model, frame_queues, track_queues, batch_size=1, early_emit=None, on_early_qr=None, global_cfg=None, model_type='yolo'):
        """
        跟踪线程，负责处理多个摄像头的目标检测和跟踪（改进版）
        """
        super().__init__()
        self.yolo_model = yolo_model
        self.model_type = model_type
        self.frame_queues = frame_queues
        self.track_queues = track_queues
        self.camera_count = len(frame_queues)
        self.global_cfg = global_cfg if isinstance(global_cfg, dict) else {}
        # 重要说明（直接关系到“移动中能否稳定检测/跟踪”）：
        # Ultralytics 的 track(persist=True) 会把“tracker 状态”绑定到 batch 的 index（0..bs-1）。
        # 如果同一个 YOLO 实例在不同调用里使用了不同的 batch 大小或不同的顺序，
        # 就会出现 tracker 串流/ID 丢失，表现为：移动时框断断续续、甚至看起来“只有停下才检得到”。
        # 因此这里强制每次推理都使用固定 batch=camera_count，并保持顺序恒为 camera_idx 升序。
        self.batch_size = self.camera_count
        self.early_emit = early_emit
        # 当早期解码出二维码文本时回调（用于写回 MatchingCounting.decode_id，保证结束汇总不为0）
        self.on_early_qr = on_early_qr
        # 兜底OpenCV二维码检测器（可选）
        try:
            self.cv_qr_detector = cv2.QRCodeDetector()
        except Exception:
            self.cv_qr_detector = None

        # 添加处理时间监控
        self.last_process_times = [0] * self.camera_count
        self.process_interval = 1.0 / 15  # 目标~15fps
        self.resource_monitor = ResourceMonitor()
        # 每路摄像头的上一帧缓存：当队列短暂为空时用于补齐固定 batch
        self._last_frames = [None] * self.camera_count

        # 早期二维码解码（用于UI即时高亮与早期注入）
        qr_decode_cfg = self.global_cfg.get('qr_decode', {}) if isinstance(self.global_cfg, dict) else {}
        self.early_decode_enabled = bool(qr_decode_cfg.get('early_decode_enabled', True))
        self.early_decode_interval_ms = int(qr_decode_cfg.get('early_decode_interval_ms', 600))
        self.early_decode_min_box_size = int(qr_decode_cfg.get('early_decode_min_box_size', 24))
        self.early_decode_max_per_frame = int(qr_decode_cfg.get('early_decode_max_per_frame', 1))
        self.early_decode_success_ttl_s = float(qr_decode_cfg.get('early_decode_success_ttl_s', 90.0))
        try:
            self.early_decode_filter_enabled = bool(qr_decode_cfg.get('qr_filter_enabled', True))
        except Exception:
            self.early_decode_filter_enabled = True
        try:
            self.early_decode_top_ratio = float(qr_decode_cfg.get('qr_top_ratio', 0.45))
        except Exception:
            self.early_decode_top_ratio = 0.45
        self.early_decode_top_ratio = max(0.05, min(0.95, self.early_decode_top_ratio))
        self._early_qr_last_ms = [0] * self.camera_count
        self._early_decoded_tids = [dict() for _ in range(self.camera_count)]  # cam -> {track_id: ts}

    @exception_handler
    def run(self):
        print("多摄像头跟踪线程启动（改进版）")

        while self.run_flag:
            self.paused.wait()

            current_time = time.time()

            # 监控资源
            self.resource_monitor.check_resources()

            # 构建固定 batch（按 camera_idx 升序，长度恒为 camera_count）
            frames = [None] * self.camera_count
            has_new = [False] * self.camera_count
            for camera_idx in range(self.camera_count):
                # 节流：不需要达到更高频率时，仍然允许读取队列（更新 last_frames），但只在满足间隔时下发结果
                got = False
                try:
                    frame = self.frame_queues[camera_idx].get_nowait()
                    got = True
                except queue.Empty:
                    frame = self._last_frames[camera_idx]

                if frame is None:
                    # 首帧兜底：用黑图填充，保持 batch 形态稳定
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self._last_frames[camera_idx] = frame
                frames[camera_idx] = frame
                has_new[camera_idx] = got

            # 若本轮没有任何相机产生新帧，则无需做一次完整推理（避免空转抢算力导致实际FPS下降）
            if not any(has_new):
                time.sleep(0.005)
                continue

            # 一次推理（YOLO: 固定 batch 避免 tracker 串流；RTMDet: 逐帧推理返回 dict）
            try:
                if self.model_type == 'rtmdet':
                    batch_results = [self.yolo_model.track(frames[i]) for i in range(self.camera_count)]
                else:
                    batch_results = self.yolo_model.batch_track(frames)
            except Exception as e:
                print(f"批量跟踪异常: {e}")
                batch_results = [None] * len(frames)

            if batch_results is None:
                batch_results = [None] * len(frames)
            # 有些异常情况下 batch_results 可能长度不匹配，做兜底对齐
            if not isinstance(batch_results, (list, tuple)) or len(batch_results) != self.camera_count:
                batch_results = list(batch_results) if isinstance(batch_results, (list, tuple)) else []
                if len(batch_results) < self.camera_count:
                    batch_results = batch_results + [None] * (self.camera_count - len(batch_results))
                else:
                    batch_results = batch_results[:self.camera_count]

            for camera_idx in range(self.camera_count):
                # 仅对“确实拿到新帧且达到处理间隔”的相机下发结果，避免重复帧导致 stable_frames 虚增
                if not has_new[camera_idx]:
                    continue
                if current_time - self.last_process_times[camera_idx] < self.process_interval:
                    continue

                frame = frames[camera_idx]
                track_results = batch_results[camera_idx]
                if track_results is None:
                    continue

                try:
                    # YOLO路径：保证下游拿到的是“单帧 List[Results]”形态
                    # RTMDet路径：track() 返回 dict，直接传递，不需要包装
                    if self.model_type != 'rtmdet' and not isinstance(track_results, (list, tuple)):
                        track_results = [track_results]

                    # 早期二维码直达UI（只对YOLO新帧触发）
                    try:
                        if self.model_type != 'rtmdet' and self.early_emit is not None and self.early_decode_enabled:
                            now_ms = int(time.time() * 1000)
                            if (now_ms - int(self._early_qr_last_ms[camera_idx] or 0)) >= int(self.early_decode_interval_ms):
                                self._early_qr_last_ms[camera_idx] = now_ms
                                self._emit_early_qr_if_any(frame, track_results, camera_idx)
                    except Exception:
                        pass

                    try:
                        self.track_queues[camera_idx].put((frame, track_results), block=False)
                    except queue.Full:
                        try:
                            self.track_queues[camera_idx].get(block=False)
                            self.track_queues[camera_idx].put((frame, track_results), block=False)
                        except (queue.Empty, queue.Full):
                            pass

                    self.last_process_times[camera_idx] = current_time
                except Exception as e:
                    print(f"摄像头 {camera_idx} 跟踪处理异常: {e}")

            # 适当休眠，避免线程消耗过多资源
            time.sleep(0.02)

        print("多摄像头跟踪线程退出")

    def _emit_early_qr_if_any(self, frame, track_results, camera_idx):
        try:
            names = track_results[0].names
            boxes = track_results[0].boxes.xyxy.cpu().numpy().astype(int)
            cls = track_results[0].boxes.cls.cpu().numpy().astype(int)
            # track_id（用于把早期解码写回匹配器，否则结束巡检汇总会是0）
            tids = None
            try:
                if getattr(track_results[0].boxes, "id", None) is not None:
                    tids = track_results[0].boxes.id.cpu().numpy().astype(int)
            except Exception:
                tids = None
            qr_indices = [i for i, c in enumerate(cls) if names[c] == 'qr']
            if not qr_indices:
                return
            h, w = frame.shape[:2]
            pattern = re.compile(r'^\d{4}-\d{5}$')
            
            # 简单的锐化核（对抗运动模糊）
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            
            # 根据面积从大到小，优先解码更大的二维码
            try:
                qr_indices = sorted(qr_indices, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]), reverse=True)
            except Exception:
                pass

            max_per_frame = max(1, int(self.early_decode_max_per_frame))
            processed = 0
            for idx in qr_indices:
                if processed >= max_per_frame:
                    break
                x1, y1, x2, y2 = boxes[idx]
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                if min(bw, bh) < int(self.early_decode_min_box_size):
                    continue
                if self.early_decode_filter_enabled:
                    try:
                        cy = (float(y1) + float(y2)) / 2.0
                        if cy > float(h) * float(self.early_decode_top_ratio):
                            continue
                    except Exception:
                        pass

                # 已成功解码的二维码track短时间内不再重复解码
                try:
                    if tids is not None and idx < len(tids):
                        tid = int(tids[idx])
                        last_ts = self._early_decoded_tids[camera_idx].get(tid)
                        if last_ts is not None and (time.time() - float(last_ts)) < float(self.early_decode_success_ttl_s):
                            continue
                except Exception:
                    pass

                x1 = max(0, x1 - 5); y1 = max(0, y1 - 5)
                x2 = min(w, x2 + 5); y2 = min(h, y2 + 5)
                roi = frame[y1:y2, x1:x2]
                
                # 1. 尝试锐化
                try:
                    roi_sharp = cv2.filter2D(roi, -1, kernel)
                except Exception:
                    roi_sharp = roi
                
                # 2. 尝试 CLAHE 增强（对抗暗光）
                roi_clahe = None
                try:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    roi_clahe = clahe.apply(gray)
                except Exception:
                    roi_clahe = None

                txt = None
                # 候选队列：锐化图 -> CLAHE图 -> 原图
                candidates = [roi_sharp]
                if roi_clahe is not None:
                    candidates.append(roi_clahe)
                candidates.append(roi)
                
                for img in candidates:
                    # 只尝试解 QRCode
                    for c in zbar_decode(img, symbols=[ZBarSymbol.QRCODE]):
                        try:
                            t = c.data.decode('utf-8', 'ignore').strip()
                            if pattern.match(t):
                                txt = t; break
                            if txt is None and len(t) > 0:
                                txt = t
                        except Exception:
                            pass
                    if txt: break
                
                if txt is None and hasattr(self, 'cv_qr_detector') and getattr(self, 'cv_qr_detector', None) is not None:
                    try:
                        # OpenCV检测器也试一下锐化图
                        res, points, _ = self.cv_qr_detector.detectAndDecode(roi_sharp)
                        if isinstance(res, str) and len(res.strip()) > 0:
                            res = res.strip()
                            m = pattern.search(res)
                            txt = m.group(0) if m else (res if pattern.match(res) else None)
                    except Exception:
                        pass
                if txt:
                    try:
                        print(f"[EARLY_EMIT_TRACK] cam={camera_idx} cage={txt} @ {int(time.time()*1000)}")
                    except Exception:
                        pass
                    # 写回匹配器：把“笼号”绑定到该二维码的 track_id 上（关键修复：结束巡检汇总才能统计到蛋数）
                    try:
                        if self.on_early_qr is not None and tids is not None and idx < len(tids):
                            self.on_early_qr(camera_idx, int(tids[idx]), txt)
                    except Exception:
                        pass
                    # 记录已解码，避免短时间重复解码
                    try:
                        if tids is not None and idx < len(tids):
                            self._early_decoded_tids[camera_idx][int(tids[idx])] = time.time()
                    except Exception:
                        pass
                    # 仍然保留UI早期提示（egg_num固定=0，仅用于UI高亮）
                    self.early_emit(camera_idx, [{ 'cage_id': txt, 'egg_num': 0, 'early': True, 'track_id': (int(tids[idx]) if tids is not None and idx < len(tids) else None) }])
                    break
                processed += 1
        except Exception:
            pass


class SingleCameraMatchThread(BaseThread):
    def __init__(self, camera_idx, matching_instance, track_queue, result_queue, early_emit=None, qr_emit=None, model_type='yolo', topology_matcher=None):
        """
        单摄像头匹配线程，独立处理一路摄像头的匹配逻辑，避免串行阻塞
        """
        super().__init__()
        self.camera_idx = camera_idx
        self.matching_instance = matching_instance
        self.track_queue = track_queue
        self.result_queue = result_queue
        self.early_emit = early_emit
        self.qr_emit = qr_emit
        self.model_type = model_type
        self.topology_matcher = topology_matcher
        self.error_counts = 0
        self.max_errors = 5

    @exception_handler
    def run(self):
        print(f"摄像头 {self.camera_idx} 匹配线程启动")
        while self.run_flag:
            self.paused.wait()
            self.process()
            # 极短休眠，让出CPU给其他线程
            time.sleep(0.001)
        print(f"摄像头 {self.camera_idx} 匹配线程退出")

    @exception_handler
    def process(self):
        try:
            frame, track_results = self.track_queue.get(timeout=0.05)
        except queue.Empty:
            return

        # RTMDet 路径：track_results 是 dict {'qr_detections': [...], 'egg_detections': [...]}
        if self.model_type == 'rtmdet':
            self._process_rtmdet(frame, track_results)
            return

        # YOLO 路径（原有逻辑）
        try:
            # 若没有检测结果或为空，直接推送空结果避免下游越界
            if track_results is None or len(track_results) == 0:
                self.result_queue.put((frame, []), block=False)
                return

            # 若无检测框，避免匹配阶段越界
            try:
                boxes = track_results[0].boxes
                if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
                    self.result_queue.put((frame, []), block=False)
                    return
            except Exception:
                self.result_queue.put((frame, []), block=False)
                return

            # 检查是否有跟踪ID
            if track_results[0].boxes.id is not None:
                try:
                    self.matching_instance.match(track_results, frame)
                except Exception as e:
                    print(f"摄像头 {self.camera_idx} 匹配过程异常: {e}")
                    self.error_counts += 1

                    # 如果错误次数过多，尝试重置匹配实例
                    if self.error_counts >= self.max_errors:
                        print(f"摄像头 {self.camera_idx} 错误次数过多，尝试重置匹配实例")
                        try:
                            if hasattr(self.matching_instance, 'reset'):
                                self.matching_instance.reset()
                            self.error_counts = 0
                        except Exception as reset_e:
                            print(f"摄像头 {self.camera_idx} 重置匹配实例失败: {reset_e}")

                    # 生成空结果，确保处理流程继续
                    match_results = []
                    self.result_queue.put((frame, match_results), block=False)
                    return

            try:
                # 先获取早期识别结果（仅用于UI即时高亮，不参与计数/上传）
                early_results = []
                if hasattr(self.matching_instance, 'drain_early_results'):
                    early_results = self.matching_instance.drain_early_results() or []

                # 确保数据类型转换
                match_results = self.matching_instance.update_and_delete_records()

                # 确保匹配结果是可迭代对象
                if not isinstance(match_results, (list, tuple)):
                    if match_results is None:
                        match_results = []
                    else:
                        match_results = [match_results]

                # 重置错误计数
                self.error_counts = 0
            except Exception as e:
                print(f"摄像头 {self.camera_idx} 更新记录异常: {e}")
                match_results = []
                self.error_counts += 1

            # 抓拍二维码图片结果（异步保存，不影响主流程）
            try:
                qr_images = []
                if hasattr(self.matching_instance, 'drain_qr_image_results'):
                    qr_images = self.matching_instance.drain_qr_image_results() or []
                if qr_images and self.qr_emit is not None:
                    for item in qr_images:
                        try:
                            self.qr_emit(item)
                        except Exception:
                            pass
            except Exception:
                pass

            # 非阻塞放入结果队列
            try:
                # 优先推送早期结果，促使UI尽快切换图标
                if early_results:
                    if self.early_emit is not None:
                        # 直接发送到UI，避免等待主接口线程轮询
                        try:
                            try:
                                # print(f"[EARLY_EMIT] cam={self.camera_idx} n={len(early_results)} t={int(time.time()*1000)}")
                                pass
                            except Exception:
                                pass
                            self.early_emit(self.camera_idx, early_results)
                        except Exception:
                            # 降级到队列
                            self.result_queue.put((frame, early_results), block=False)
                    else:
                        self.result_queue.put((frame, early_results), block=False)

                self.result_queue.put((frame, match_results), block=False)
            except queue.Full:
                # 清理旧结果
                try:
                    self.result_queue.get(block=False)
                    # 尝试再次推送早期结果（仅当未直接发射时）
                    if early_results and self.early_emit is None:
                        self.result_queue.put((frame, early_results), block=False)
                    self.result_queue.put((frame, match_results), block=False)
                except (queue.Empty, queue.Full):
                    pass

        except Exception as e:
            print(f"摄像头 {self.camera_idx} 匹配处理异常: {e}")
            self.error_counts += 1

    def _process_rtmdet(self, frame, track_results):
        """RTMDet 路径：QR解码 + TopologyMatcher 蛋-笼匹配，在帧上绘制检测框"""
        try:
            if not isinstance(track_results, dict):
                self.result_queue.put((frame, []), block=False)
                return

            qr_dets = track_results.get('qr_detections', [])
            egg_dets = track_results.get('egg_detections', [])
            match_results = []

            # 对 QR 检测框做解码，填充 cage_id
            self._decode_qr_boxes(frame, qr_dets)

            # 在帧上绘制 RTMDet 检测结果
            try:
                from model.inference.pipeline_logic import draw_rtmdet_detections
                annotated = frame.copy()
                draw_rtmdet_detections(annotated, qr_dets, egg_dets)
            except Exception:
                annotated = frame

            if self.topology_matcher is not None and (qr_dets or egg_dets):
                egg_centers = []
                egg_meta = []
                for det in egg_dets:
                    center = det.get('center')
                    if center is not None:
                        egg_centers.append((float(center[0]), float(center[1])))
                    else:
                        bbox = det.get('bbox', [0, 0, 0, 0])
                        egg_centers.append(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))
                    egg_meta.append({
                        'class_id': det.get('class_id', 0),
                        'is_invalid': det.get('class_id', 0) == 1,
                        'score': det.get('score', 0.0),
                    })

                tm_qr_dets = []
                for det in qr_dets:
                    hbb = det.get('hbb', [0, 0, 0, 0])
                    cx = (hbb[0] + hbb[2]) / 2.0
                    cy = (hbb[1] + hbb[3]) / 2.0
                    tm_qr_dets.append({
                        'center': (cx, cy),
                        'hbb': hbb,
                        'rotated_box': det.get('rotated_box'),
                        'validity_score': det.get('validity_score', det.get('score', 0.0)),
                        'score': det.get('score', 0.0),
                        'class_id': det.get('class_id', 0),
                        'cage_id': det.get('cage_id'),       # 解码得到的笼位 ID
                        'decode_id': det.get('cage_id'),     # TopologyMatcher 兼容字段
                    })

                try:
                    results = self.topology_matcher.match(
                        egg_centers, tm_qr_dets, annotated, egg_meta=egg_meta
                    )
                    if results:
                        match_results = results if isinstance(results, list) else [results]
                except Exception as e:
                    print(f"摄像头 {self.camera_idx} TopologyMatcher 异常: {e}")
                    self.error_counts += 1

            try:
                self.result_queue.put((annotated, match_results), block=False)
            except queue.Full:
                try:
                    self.result_queue.get(block=False)
                    self.result_queue.put((annotated, match_results), block=False)
                except (queue.Empty, queue.Full):
                    pass

        except Exception as e:
            print(f"摄像头 {self.camera_idx} RTMDet匹配异常: {e}")
            self.error_counts += 1

    def _decode_qr_boxes(self, frame, qr_dets):
        """对每个 QR 检测框裁剪区域并用 WeChatQRCode 解码，结果写入 det['cage_id']"""
        if not qr_dets:
            return
        # 懒加载解码器（只初始化一次）
        if not hasattr(self, '_wechat_detector'):
            self._wechat_detector = None
            try:
                import cv2 as _cv2
                if hasattr(_cv2, 'wechat_qrcode_WeChatQRCode'):
                    wechat_dir = os.path.join(os.getcwd(), 'resources', 'wechat')
                    det_proto = os.path.join(wechat_dir, 'detect.prototxt')
                    det_model = os.path.join(wechat_dir, 'detect.caffemodel')
                    sr_proto = os.path.join(wechat_dir, 'sr.prototxt')
                    sr_model = os.path.join(wechat_dir, 'sr.caffemodel')
                    if all(os.path.isfile(p) for p in [det_proto, det_model, sr_proto, sr_model]):
                        self._wechat_detector = _cv2.wechat_qrcode_WeChatQRCode(
                            det_proto, det_model, sr_proto, sr_model)
                    else:
                        self._wechat_detector = _cv2.wechat_qrcode_WeChatQRCode()
                    print(f"[RTMDet] 摄像头{self.camera_idx} WeChatQRCode 加载成功")
            except Exception as e:
                print(f"[RTMDet] 摄像头{self.camera_idx} WeChatQRCode 初始化失败: {e}")

        h, w = frame.shape[:2]
        for det in qr_dets:
            if det.get('cage_id'):
                continue  # 已有解码结果，跳过
            hbb = det.get('hbb')
            if hbb is None:
                continue
            x1, y1, x2, y2 = int(hbb[0]), int(hbb[1]), int(hbb[2]), int(hbb[3])
            # 加一点 padding
            pad = 10
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # 尝试解码
            text = None
            try:
                if self._wechat_detector is not None:
                    texts, _ = self._wechat_detector.detectAndDecode(crop)
                    if isinstance(texts, (list, tuple)):
                        texts = [str(t).strip() for t in texts if t]
                        if texts:
                            text = texts[0]
            except Exception:
                pass
            if not text:
                try:
                    import cv2 as _cv2
                    res, _, _ = _cv2.QRCodeDetector().detectAndDecode(crop)
                    if res and res.strip():
                        text = res.strip()
                except Exception:
                    pass
            if text:
                det['cage_id'] = text
                print(f"[RTMDet] 摄像头{self.camera_idx} 解码 QR: {text}")


class MultiCameraMatchThread(BaseThread):
    # 保留旧类名以防兼容性问题，但不建议使用了
    def __init__(self, matching_instances, track_queues, result_queues, early_emit=None):
        super().__init__()
        self.matching_instances = matching_instances
        self.track_queues = track_queues
        self.result_queues = result_queues
        self.camera_count = len(track_queues)
        self.early_emit = early_emit
        self.error_counts = [0] * self.camera_count
        self.max_errors = 5

    @exception_handler
    def run(self):
        print("警告：MultiCameraMatchThread 被调用（串行模式），建议使用 SingleCameraMatchThread")
        while self.run_flag:
            self.paused.wait()
            for i in range(self.camera_count):
                self.process_camera(i)
            time.sleep(0.003)

    @exception_handler
    def process_camera(self, camera_idx):
        # 旧的串行处理逻辑，暂时保留内容
        try:
            frame, track_results = self.track_queues[camera_idx].get_nowait()
            if track_results is None or len(track_results) == 0:
                self.result_queues[camera_idx].put((frame, []), block=False)
                return
            try:
                boxes = track_results[0].boxes
                if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
                    self.result_queues[camera_idx].put((frame, []), block=False)
                    return
            except Exception:
                self.result_queues[camera_idx].put((frame, []), block=False)
                return
            if track_results[0].boxes.id is not None:
                self.matching_instances[camera_idx].match(track_results, frame)
            early_results = []
            if hasattr(self.matching_instances[camera_idx], 'drain_early_results'):
                early_results = self.matching_instances[camera_idx].drain_early_results() or []
            match_results = self.matching_instances[camera_idx].update_and_delete_records()
            if not isinstance(match_results, (list, tuple)):
                match_results = [match_results] if match_results is not None else []
            if early_results and self.early_emit is not None:
                self.early_emit(camera_idx, early_results)
            self.result_queues[camera_idx].put((frame, match_results), block=False)
        except queue.Empty:
            return
        except Exception:
            pass


class MultiCameraHTTPThread(BaseThread):
    def __init__(self, no_picture_result_queues, global_cfg, qr_image_queue=None):
        """
        数据保存线程，负责处理多个摄像头的数据库保存任务（本地+远程）
        """
        super().__init__()
        self.no_picture_result_queues = no_picture_result_queues
        self.camera_count = len(no_picture_result_queues)
        self.qr_image_queue = qr_image_queue

        # 错误恢复机制
        self.error_counts = [0] * self.camera_count
        self.max_errors = 5
        self.camera_paused = [False] * self.camera_count
        self.pause_duration = 60
        self.pause_start_times = [0] * self.camera_count

        # 去重缓存: {cage_id: {'time': datetime, 'egg_num': int}}
        self.last_saved_records = {}
        # 同一笼位在该时间窗内重复扫描：不重复写库（默认30分钟）
        self.cage_dedup_seconds = 30 * 60
        try:
            if isinstance(global_cfg, dict):
                if global_cfg.get('cage_dedup_seconds') is not None:
                    self.cage_dedup_seconds = float(global_cfg.get('cage_dedup_seconds'))
                elif global_cfg.get('cage_dedup_minutes') is not None:
                    self.cage_dedup_seconds = float(global_cfg.get('cage_dedup_minutes')) * 60.0
        except Exception:
            self.cage_dedup_seconds = 30 * 60

        # 初始化数据库连接（按摄像头表分流）
        # 本地数据库
        local_cfg = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': '123456',
            'db': 'wenshi_eggs_record',
            'table': 'duckdata1',
            'include_img': True
        }
        # 远程数据库
        remote_cfg = {
            'host': '8.138.181.75',
            'port': 3306,
            'user': 'root',
            'password': 'WSNwsn640',
            'db': 'wenshi_eggs_record',
            'table': 'duckdata1',
            'include_img': False  # 远程不传图片
        }

        self.local_savers = []
        self.remote_savers = []
        for i in range(self.camera_count):
            cam_cfg = global_cfg.get(f'camera_{i}', {}) if isinstance(global_cfg, dict) else {}
            table_name = cam_cfg.get('table', local_cfg.get('table'))

            cfg_local = local_cfg.copy()
            cfg_local['table'] = table_name
            self.local_savers.append(SaveToMySQL(cfg_local))

            cfg_remote = remote_cfg.copy()
            cfg_remote['table'] = table_name
            self.remote_savers.append(SaveToMySQL(cfg_remote))

        # 二维码抓拍保存（仅本地）
        qr_cfg = global_cfg.get('qr_image', {}) if isinstance(global_cfg, dict) else {}
        try:
            min_h = float(qr_cfg.get('min_interval_h', 12.0))
        except Exception:
            min_h = 12.0
        self.qr_min_interval_s = max(0.0, float(min_h) * 3600.0)
        try:
            base_dir = qr_cfg.get('save_dir')
        except Exception:
            base_dir = None
        if not base_dir:
            try:
                base_dir = str(global_cfg.get('picture_recognition_path', 'output'))
            except Exception:
                base_dir = 'output'
            base_dir = os.path.join(base_dir, 'qr_scan')
        self.qr_image_save_dir = base_dir
        self.qr_last_saved = {}
        try:
            from model.communication.SaveToMySQL import SaveQrToMySQL
            qr_db_cfg = {
                'host': 'localhost',
                'port': 3306,
                'user': 'root',
                'password': '123456',
                'db': 'wenshi_eggs_record',
                'table': qr_cfg.get('table', 'qr_code_images'),
                'ensure_table': True,
            }
            self.qr_saver = SaveQrToMySQL(qr_db_cfg)
        except Exception:
            self.qr_saver = None

    @exception_handler
    def run(self):
        print("多摄像头数据保存线程启动")

        while self.run_flag:
            self.paused.wait()

            current_time = time.time()
            has_data = False

            for camera_idx in range(self.camera_count):
                # 检查暂停状态
                if self.camera_paused[camera_idx]:
                    if current_time - self.pause_start_times[camera_idx] > self.pause_duration:
                        print(f"恢复摄像头 {camera_idx} 的数据保存")
                        self.camera_paused[camera_idx] = False
                        self.error_counts[camera_idx] = 0
                    else:
                        continue

                try:
                    try:
                        no_picture_result = self.no_picture_result_queues[camera_idx].get(timeout=0.1)
                    except queue.Empty:
                        continue

                    no_picture_result['camera_idx'] = camera_idx

                    # 业务逻辑：去重处理
                    cage_id = str(no_picture_result.get('cage_id', ''))
                    egg_num = int(no_picture_result.get('egg_num', 0))
                    record_time_str = no_picture_result.get('record_time', '')
                    
                    try:
                        record_dt = datetime.datetime.strptime(record_time_str, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        record_dt = datetime.datetime.now()

                    should_save = True
                    if cage_id in self.last_saved_records:
                        last_record = self.last_saved_records[cage_id]
                        last_time = last_record['time']
                        
                        # 时间窗内重复扫描：一律不新增记录（避免重复计数/重复写库）
                        try:
                            if (record_dt - last_time).total_seconds() < float(self.cage_dedup_seconds):
                                should_save = False
                                print(f"笼位 {cage_id} 重复扫描（{int(self.cage_dedup_seconds // 60)}分钟内），跳过保存")
                        except Exception:
                            # 异常情况下宁可不保存，避免重复写库
                            should_save = False
                    
                    if should_save:
                        # 保存到本地/远程（按摄像头表分流）
                        local_saver = self.local_savers[camera_idx] if camera_idx < len(self.local_savers) else None
                        remote_saver = self.remote_savers[camera_idx] if camera_idx < len(self.remote_savers) else None
                        if local_saver:
                            local_saver.save(no_picture_result)
                        if remote_saver:
                            remote_saver.save(no_picture_result)
                        
                        # 更新缓存
                        self.last_saved_records[cage_id] = {
                            'time': record_dt,
                            'egg_num': egg_num
                        }
                        
                        # 重置错误计数
                        self.error_counts[camera_idx] = 0

                    has_data = True
                except Exception as e:
                    print(f"摄像头 {camera_idx} 数据保存异常: {e}")
                    self.error_counts[camera_idx] += 1
                    
                    if self.error_counts[camera_idx] >= self.max_errors:
                        print(f"摄像头 {camera_idx} 错误过多，暂停保存 {self.pause_duration} 秒")
                        self.camera_paused[camera_idx] = True
                        self.pause_start_times[camera_idx] = current_time

            # 处理二维码抓拍保存（非阻塞）
            try:
                self._process_qr_image_queue()
            except Exception:
                pass

            if not has_data:
                time.sleep(0.1)
            else:
                time.sleep(0.02)

        print("多摄像头数据保存线程退出")

    def _should_save_qr_image(self, id_code: str, now_ts: float) -> bool:
        try:
            if not id_code:
                return False
            if self.qr_min_interval_s <= 0:
                return True
            last_ts = self.qr_last_saved.get(id_code)
            if last_ts is not None:
                if (now_ts - float(last_ts)) < float(self.qr_min_interval_s):
                    return False
            # 若缓存无记录，尝试查库避免重启后重复保存
            if self.qr_saver is not None:
                try:
                    db_ts = self.qr_saver.get_last_scan_time(id_code)
                except Exception:
                    db_ts = None
                if db_ts is not None:
                    try:
                        if (now_ts - float(db_ts)) < float(self.qr_min_interval_s):
                            self.qr_last_saved[id_code] = float(db_ts)
                            return False
                    except Exception:
                        pass
            return True
        except Exception:
            return False

    def _save_qr_image_item(self, item: dict) -> None:
        if not isinstance(item, dict):
            return
        id_code = str(item.get('id_code', '') or '').strip()
        if not id_code:
            return
        now_ts = time.time()
        if not self._should_save_qr_image(id_code, now_ts):
            return
        img = item.get('image')
        if img is None or getattr(img, "size", 0) == 0:
            return
        record_time = item.get('record_time')
        if not isinstance(record_time, str) or not record_time:
            record_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))
        cam_idx = item.get('camera_idx', -1)
        # 生成保存路径
        try:
            date_dir = time.strftime('%Y%m%d', time.localtime(now_ts))
            out_dir = os.path.join(self.qr_image_save_dir, date_dir)
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            return
        safe_code = id_code.replace('\\', '_').replace('/', '_').replace(':', '_').replace(' ', '')
        t_str = time.strftime('%H%M%S', time.localtime(now_ts))
        file_name = f"qr_{safe_code}_{t_str}_cam{cam_idx}.jpg"
        out_path = os.path.join(out_dir, file_name)
        try:
            cv2.imwrite(out_path, img)
        except Exception:
            return
        # 写库（仅本地）
        try:
            if self.qr_saver is not None:
                self.qr_saver.save_qr_image(id_code, out_path, record_time)
        except Exception:
            pass
        self.qr_last_saved[id_code] = now_ts

    def _process_qr_image_queue(self) -> None:
        if self.qr_image_queue is None:
            return
        # 每轮最多处理少量，避免阻塞
        for _ in range(4):
            try:
                item = self.qr_image_queue.get_nowait()
            except Exception:
                break
            try:
                self._save_qr_image_item(item)
            except Exception:
                pass


class MultiCameraInterface(QThread):
    frames_generated = pyqtSignal(list)  # 发送所有摄像头的帧列表
    detection_results_generated = pyqtSignal(int, list) # 发送 (摄像头索引, 检测结果列表)
    egg_count_updated = pyqtSignal(int) # 发送总蛋数更新信号
    # 停止巡检后的汇总结果（由后台线程发出，避免UI线程在 stop() / get_detection_summary() 上卡死）
    stop_summary_ready = pyqtSignal(dict)

    _shared_model = None
    _shared_model_event = threading.Event()
    _shared_model_lock = threading.Lock()

    @classmethod
    def preload_model_async(cls, cfg):
        """后台预加载YOLO模型，减少点击开始时的阻塞"""
        # RTMDet 模式不需要预热 YOLO
        if isinstance(cfg, dict) and cfg.get('model_type') == 'rtmdet':
            return
        with cls._shared_model_lock:
            if cls._shared_model is not None or cls._shared_model_event.is_set():
                return
            cls._shared_model_event.clear()

            def _load():
                try:
                    cls._shared_model = YOLOTrack(cfg)
                finally:
                    cls._shared_model_event.set()

            t = threading.Thread(target=_load, name="yolo-preload", daemon=True)
            t.start()

    @classmethod
    def get_or_create_model(cls, cfg):
        """获取共享模型；若未预加载则同步创建"""
        if cls._shared_model is not None:
            return cls._shared_model
        # 若正在预加载则等待完成（最多5秒）
        if cls._shared_model_event.is_set():
            cls._shared_model_event.wait(timeout=5)
            return cls._shared_model
        with cls._shared_model_lock:
            if cls._shared_model is None:
                cls._shared_model = YOLOTrack(cfg)
                cls._shared_model_event.set()
        return cls._shared_model

    def __init__(self, cfg):
        """
        多摄像头接口（完整改进版）
        """
        super().__init__()
        self.global_cfg = cfg

        # 首先获取预配置摄像头（优先使用配置，避免启动时的搜索耗时）
        available_cameras = self._resolve_camera_sources(cfg)
        if not available_cameras:
            detector = CameraDetector()
            available_cameras = detector.detect_available_cameras()

        if not available_cameras:
            raise RuntimeError("未检测到任何可用摄像头！请检查摄像头连接。")

        # 根据实际可用摄像头调整配置
        requested_count = cfg.get('camera_count', len(available_cameras))
        actual_count = min(requested_count, len(available_cameras))

        if actual_count < requested_count:
            print(f"警告：请求 {requested_count} 个摄像头，但只检测到 {len(available_cameras)} 个")
            print(f"将使用 {actual_count} 个摄像头: {available_cameras[:actual_count]}")

        cfg['camera_count'] = actual_count

        # 创建摄像头配置
        self.camera_configs = self._create_camera_configs(cfg, available_cameras[:actual_count])
        self.camera_count = len(self.camera_configs)

        # 为每个摄像头创建队列，保持小容量以减少延迟（满则丢旧保新）
        self.frame_queues = [queue.Queue(maxsize=2) for _ in range(self.camera_count)]
        self.track_queues = [queue.Queue(maxsize=2) for _ in range(self.camera_count)]
        self.result_queues = [queue.Queue(maxsize=2) for _ in range(self.camera_count)]
        self.no_picture_result_queues = [queue.Queue() for _ in range(self.camera_count)]
        # 二维码抓拍队列（用于复核）
        self.qr_image_queue = queue.Queue(maxsize=64)

        # 创建/复用检测模型（支持 model_type='yolo' 或 'rtmdet'）
        self.model_type = cfg.get('model_type', 'yolo')
        if self.model_type == 'rtmdet':
            self.yolo_model = create_detector(cfg)
        else:
            self.yolo_model = self.get_or_create_model(cfg)

        # 为每个摄像头创建匹配实例
        self.matching_instances = [MatchingCounting(self._adjust_config_for_camera(cfg, i))
                                   for i in range(self.camera_count)]

        # RTMDet 路径：为每个摄像头创建 TopologyMatcher
        self.topology_matchers = [None] * self.camera_count
        if self.model_type == 'rtmdet':
            for i in range(self.camera_count):
                camera_cfg = self._adjust_config_for_camera(cfg, i)
                self.topology_matchers[i] = create_matcher(camera_cfg)
        # 将早期直达UI的回调安装到每个匹配实例（可选使用）
        try:
            for i in range(self.camera_count):
                if hasattr(self.matching_instances[i], 'set_early_emit'):
                    # 直接传入 (results) -> emit(cam_idx, results)
                    self.matching_instances[i].set_early_emit(
                        lambda results, cam_idx=i: self.detection_results_generated.emit(cam_idx, results)
                    )
        except Exception:
            pass

        # 为每个摄像头创建HTTP上传实例 (已弃用，改为在MultiCameraHTTPThread内部处理数据库保存)
        # self.send_http_instances = [SendHttp(self._adjust_config_for_camera(cfg, i))
        #                             for i in range(self.camera_count)]
        self.send_http_instances = []

        # 控制标志
        self.run_flag = True
        self.paused = threading.Event()
        self.paused.set()  # 开始时不暂停

        # 添加资源监控
        self.resource_monitor = ResourceMonitor()
        
        # 添加蛋数统计
        self.total_egg_count = 0
        self.detected_results = {}  # 存储检测结果，用于最终汇总
        # 同一笼位在该时间窗内重复扫描：不新增计数（默认30分钟）
        self.cage_dedup_seconds = 30 * 60
        self._cage_last_count_ts = {}  # cage_id -> last_count_time(time.time())
        try:
            if isinstance(cfg, dict):
                if cfg.get('cage_dedup_seconds') is not None:
                    self.cage_dedup_seconds = float(cfg.get('cage_dedup_seconds'))
                elif cfg.get('cage_dedup_minutes') is not None:
                    self.cage_dedup_seconds = float(cfg.get('cage_dedup_minutes')) * 60.0
        except Exception:
            self.cage_dedup_seconds = 30 * 60
        # 记录每路摄像头上一帧有效图像，避免UI缺帧时使用占位图导致闪烁
        self.last_frames = [None] * self.camera_count
        # 停止后汇总缓存（供UI兜底读取）
        self._last_summary = None

        # 创建线程 - 使用已存在的方法
        self._create_threads()  # 修改这里，使用正确的方法名

        print(f"多摄像头接口初始化完成，将管理 {self.camera_count} 个摄像头")


    def _create_camera_configs(self, global_cfg, available_camera_indices):
        """根据实际可用摄像头创建配置"""
        camera_configs = []

        for i, actual_idx in enumerate(available_camera_indices):
            camera_cfg = global_cfg.copy()
            # 避免把顶层 video_file 误“继承”到每一路摄像头配置里（否则会出现6路都读同一个文件→高倍速/EOF反复重连）
            camera_cfg.pop('video_file', None)
            camera_cfg['camera_idx'] = i
            camera_cfg['actual_video_idx'] = actual_idx

            # 如果有特定配置则使用
            camera_key = f'camera_{i}'
            if camera_key in global_cfg:
                camera_cfg.update(global_cfg[camera_key])

            # 使用实际检测到的摄像头索引/文件路径
            if isinstance(actual_idx, str) and len(actual_idx) > 0:
                camera_cfg['video_file'] = actual_idx
                # video 字段保留但置空，避免后续误当作设备索引使用
                camera_cfg['video'] = None
            else:
                camera_cfg['video'] = actual_idx

            camera_configs.append(camera_cfg)

        return camera_configs

    def _resolve_camera_sources(self, cfg):
        """优先从配置读取固定摄像头源，避免启动时的搜索和预热等待"""
        try:
            # 固定ID优先（如 fixed_camera_ids: [0,1,2,3,4,5]）
            if isinstance(cfg.get('fixed_camera_ids'), (list, tuple)) and len(cfg.get('fixed_camera_ids')) > 0:
                return list(cfg.get('fixed_camera_ids'))

            # 单视频模式：若顶层 video_file 配置存在且每路 camera_i 未显式配置 video_file，
            # 则认为用户要用“单个文件源”进行离线测试，自动返回单路源（避免6路都读同一文件导致高倍速/反复重连）。
            top_vfile = cfg.get('video_file')
            if isinstance(top_vfile, str) and len(top_vfile) > 0:
                has_per_camera_file = False
                camera_count = int(cfg.get('camera_count', 6))
                for i in range(camera_count):
                    cam_cfg = cfg.get(f'camera_{i}', {}) if isinstance(cfg, dict) else {}
                    vfile = cam_cfg.get('video_file')
                    if isinstance(vfile, str) and len(vfile) > 0:
                        has_per_camera_file = True
                        break
                if not has_per_camera_file:
                    return [top_vfile]

            camera_count = int(cfg.get('camera_count', 6))
            sources = []
            for i in range(camera_count):
                cam_cfg = cfg.get(f'camera_{i}', {}) if isinstance(cfg, dict) else {}
                if cam_cfg.get('video_file'):
                    sources.append(cam_cfg['video_file'])
                elif 'video' in cam_cfg:
                    sources.append(cam_cfg['video'])
            if sources:
                return sources

            # 兜底：返回连续索引
            return list(range(camera_count))
        except Exception:
            return []

    def _adjust_config_for_camera(self, cfg, camera_idx):
        """为特定摄像头调整配置"""
        camera_cfg = cfg.copy()

        # 修改特定于摄像头的设置
        camera_cfg['camera_idx'] = camera_idx

        # 修改文件保存路径，为每个摄像头创建子目录
        if 'picture_recognition_path' in camera_cfg:
            base_path = camera_cfg['picture_recognition_path']
            print(f"摄像头 {camera_idx} 原始配置路径: {base_path}")
            
            # 确保使用绝对路径并正确构建目录结构
            import os
            if not os.path.isabs(base_path):
                # 如果是相对路径，转换为绝对路径
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                base_path = os.path.join(project_root, base_path)
                print(f"摄像头 {camera_idx} 转换后的绝对路径: {base_path}")
            
            camera_path = os.path.join(base_path, "Recognition", f"camera_{camera_idx}")
            camera_cfg['picture_recognition_path'] = camera_path
            print(f"摄像头 {camera_idx} 最终保存路径: {camera_path}")
            
            # 确保目录存在
            if not os.path.exists(camera_path):
                os.makedirs(camera_path)
                print(f"创建摄像头 {camera_idx} 图片保存目录: {camera_path}")
            else:
                print(f"摄像头 {camera_idx} 图片保存目录已存在: {camera_path}")

        # 如果有特定的摄像头配置，则覆盖
        camera_key = f'camera_{camera_idx}'
        if camera_key in cfg:
            for key, value in cfg[camera_key].items():
                camera_cfg[key] = value

        return camera_cfg

    def _create_threads(self):
        """创建处理线程"""
        # 帧读取线程
        self.frame_thread = MultiCameraFrameThread(self.camera_configs, self.frame_queues)

        # 跟踪线程
        if 'batch_size' in self.global_cfg:
            batch_size = int(self.global_cfg.get('batch_size', 1) or 1)
        else:
            batch_size = 3 if torch.cuda.is_available() else 1
        batch_size = max(1, min(batch_size, self.camera_count))
        self.track_thread = MultiCameraTrackThread(
            self.yolo_model, self.frame_queues, self.track_queues, batch_size,
            early_emit=lambda cam_idx, results: self.detection_results_generated.emit(cam_idx, results),
            on_early_qr=self._on_track_early_qr,
            global_cfg=self.global_cfg,
            model_type=self.model_type
        )

        # 匹配线程（每路独立）
        self.match_threads = []
        for i in range(self.camera_count):
            t = SingleCameraMatchThread(
                i, self.matching_instances[i], self.track_queues[i], self.result_queues[i],
                early_emit=lambda cam_idx, results: self.detection_results_generated.emit(cam_idx, results),
                qr_emit=self._enqueue_qr_image,
                model_type=self.model_type,
                topology_matcher=self.topology_matchers[i] if hasattr(self, 'topology_matchers') else None
            )
            self.match_threads.append(t)

        self.match_thread = None

        # HTTP上传线程
        self.http_thread = MultiCameraHTTPThread(
            self.no_picture_result_queues, self.global_cfg, self.qr_image_queue
        )

    def _on_track_early_qr(self, cam_idx: int, qr_track_id: int, cage_id: str):
        """Track线程早期解码回调：把笼号写回该路摄像头的 MatchingCounting，保证 stop 汇总能计数。"""
        try:
            if not hasattr(self, "matching_instances"):
                return
            if cam_idx < 0 or cam_idx >= len(self.matching_instances):
                return
            matcher = self.matching_instances[cam_idx]
            if matcher is None:
                return
            if hasattr(matcher, "ingest_external_qr_decode"):
                matcher.ingest_external_qr_decode(int(qr_track_id), str(cage_id))
        except Exception:
            pass

    def _enqueue_qr_image(self, payload: dict) -> None:
        """将二维码抓拍任务放入后台保存队列（不阻塞检测）。"""
        if payload is None:
            return
        try:
            self.qr_image_queue.put_nowait(payload)
        except Exception:
            # 队列满则丢弃，避免检测线程阻塞
            pass

    def start_interface(self):
        """启动所有线程"""
        print("启动多摄像头接口所有线程")
        self.frame_thread.start()
        time.sleep(0.2)  # 减少等待时间
        self.track_thread.start()
        time.sleep(0.1)  # 减少等待时间
        # 启动并行匹配线程
        for t in self.match_threads:
            t.start()
        time.sleep(0.1)  # 减少等待时间
        self.http_thread.start()

    def stop_interface(self):
        """停止所有线程"""
        print("停止多摄像头接口所有线程")
        # 先发出停止指令
        self.frame_thread.stop()
        self.track_thread.stop()
        for t in self.match_threads:
            t.stop()
        self.http_thread.stop()

        # 主动释放摄像头，避免等待线程退出期间占用设备
        try:
            self.frame_thread.release_all_cameras()
        except Exception as e:
            print(f"主动释放摄像头时异常: {e}")

        # 等待线程完成
        self.frame_thread.wait(5000)  # 5秒超时
        self.track_thread.wait(5000)
        for t in self.match_threads:
            t.wait(5000)
        self.http_thread.wait(5000)

        # 再次尝试确保摄像头资源已经释放
        try:
            self.frame_thread.release_all_cameras()
        except Exception:
            pass

    def resume_interface(self):
        """恢复所有线程"""
        self.frame_thread.resume()
        self.track_thread.resume()
        for t in self.match_threads:
            t.resume()
        self.http_thread.resume()

    def pause_interface(self):
        """暂停所有线程"""
        self.frame_thread.pause()
        self.track_thread.pause()
        for t in self.match_threads:
            t.pause()
        self.http_thread.pause()

    def stop(self):
        """停止接口（非阻塞）：仅请求停止，避免在UI线程里 wait/join 导致卡死/崩溃。"""
        self.run_flag = False
        # 尽快通知子线程退出（不在这里 wait）
        try:
            if getattr(self, "frame_thread", None) is not None:
                self.frame_thread.stop()
        except Exception:
            pass
        try:
            if getattr(self, "track_thread", None) is not None:
                self.track_thread.stop()
        except Exception:
            pass
        try:
            if getattr(self, "match_threads", None) is not None:
                for t in self.match_threads:
                    t.stop()
        except Exception:
            pass
        try:
            if getattr(self, "http_thread", None) is not None:
                self.http_thread.stop()
        except Exception:
            pass
        # 确保线程不会被暂停事件阻塞
        self.resume()

    def __del__(self):
        # 析构时确保资源释放
        try:
            if getattr(self, 'run_flag', False):
                self.stop()
        except Exception:
            pass

    def pause(self):
        """暂停接口"""
        self.paused.clear()
        self.pause_interface()

    def resume(self):
        """恢复接口"""
        self.paused.set()
        self.resume_interface()

    def _should_accept_cage_count(self, cage_id, now_ts=None) -> bool:
        """
        笼位去重：同一 cage_id 在指定时间窗内重复扫描不新增计数。
        返回 True 表示“允许计数/允许写库”；False 表示“重复扫描，跳过新增”。
        """
        try:
            cid = str(cage_id).strip()
        except Exception:
            return False
        if not cid:
            return False
        if now_ts is None:
            now_ts = time.time()
        try:
            now_ts = float(now_ts)
        except Exception:
            now_ts = time.time()
        try:
            last_ts = self._cage_last_count_ts.get(cid)
            if last_ts is not None:
                try:
                    if (now_ts - float(last_ts)) < float(self.cage_dedup_seconds):
                        return False
                except Exception:
                    # 去重判断失败时，保守策略：视为重复，避免误增计数
                    return False
            self._cage_last_count_ts[cid] = now_ts
            return True
        except Exception:
            # 极端情况下兜底：允许一次并写入缓存，避免系统完全不计数
            try:
                self._cage_last_count_ts[cid] = now_ts
            except Exception:
                pass
            return True
    
    def get_detection_summary(self):
        """获取检测汇总信息"""
        try:
            # 在汇总前，强制从各摄像头的匹配器中拉取尚未触发超时的结果
            if hasattr(self, 'matching_instances'):
                for cam_idx, matcher in enumerate(self.matching_instances):
                    if matcher is None:
                        continue
                    try:
                        final_list = matcher.finalize_all_results(force=True)
                    except Exception:
                        final_list = []

                    if not final_list:
                        continue

                    for result in final_list:
                        egg_num = result.get('egg_num', 0)
                        cage_id = result.get('cage_id')
                        try:
                            cage_id = str(cage_id).strip() if cage_id is not None else ''
                        except Exception:
                            cage_id = ''
                        if egg_num > 0 and cage_id:
                            # 30分钟去重：避免停止巡检汇总时把短时间重复扫描再次叠加到总数
                            try:
                                if self._should_accept_cage_count(cage_id, time.time()):
                                    self.total_egg_count += egg_num
                            except Exception:
                                pass
                            # 汇总信息：保留蛋数更大的那次（不影响总数去重）
                            try:
                                prev = self.detected_results.get(cage_id, {})
                                prev_num = int(prev.get('egg_num', 0)) if isinstance(prev, dict) else 0
                            except Exception:
                                prev_num = 0
                            if egg_num >= prev_num:
                                self.detected_results[cage_id] = {
                                    'egg_num': egg_num,
                                    'frame_path': result.get('frame_path'),
                                    'camera_idx': cam_idx,
                                    'record_time': result.get('record_time')
                                }
        except Exception:
            pass

        return {
            'total_egg_count': self.total_egg_count,
            'detected_results': self.detected_results.copy()
        }

    @exception_handler
    def run(self):
        """主线程运行函数（改进版）"""
        print("初始化多摄像头检测接口（改进版）")
        self.start_interface()

        last_frame_time = time.time()
        ui_update_interval = 1.0 / 15  # UI更新频率10-15fps，减少负担

        try:
            while self.run_flag:
                self.paused.wait()

                current_time = time.time()

                # 监控系统资源（不阻塞早期事件）
                self.resource_monitor.check_resources()

                # 收集所有摄像头的当前帧（可能不会每次循环都发送到UI）
                frames = [None] * self.camera_count
                any_frame_received = False

                for camera_idx in range(self.camera_count):
                    try:
                        # 快速尽可能多地取结果，减少排队导致的延迟
                        while True:
                            try:
                                frame, match_results = self.result_queues[camera_idx].get_nowait()
                            except queue.Empty:
                                break

                            frames[camera_idx] = frame
                            any_frame_received = True

                            if not match_results:
                                continue

                            # 拆分早期结果与正式结果
                            early_list = [r for r in match_results if isinstance(r, dict) and r.get('early')]
                            formal_list = [r for r in match_results if not (isinstance(r, dict) and r.get('early'))]

                            # 先把早期结果发给UI（仅用于图标变更）
                            if early_list:
                                self.detection_results_generated.emit(camera_idx, early_list)

                            # 再处理正式结果：发信号 + 计数 + 存档
                            if formal_list:
                                self.detection_results_generated.emit(camera_idx, formal_list)

                                for result in formal_list:
                                    egg_num = result.get('egg_num', 0)
                                    cage_id = result.get('cage_id')
                                    try:
                                        cage_id = str(cage_id).strip() if cage_id is not None else ''
                                    except Exception:
                                        cage_id = ''
                                    if egg_num > 0 and cage_id:
                                        # 30分钟去重：同一笼位短时间重复扫描不叠加计数/不重复写库
                                        accepted = False
                                        try:
                                            accepted = bool(self._should_accept_cage_count(cage_id, time.time()))
                                        except Exception:
                                            accepted = True

                                        # 汇总信息：保留蛋数更大的那次（不影响总数去重）
                                        try:
                                            prev = self.detected_results.get(cage_id, {})
                                            prev_num = int(prev.get('egg_num', 0)) if isinstance(prev, dict) else 0
                                        except Exception:
                                            prev_num = 0

                                        if accepted:
                                            self.total_egg_count += egg_num
                                            self.detected_results[cage_id] = {
                                                'egg_num': egg_num,
                                                'frame_path': result.get('frame_path'),
                                                'camera_idx': camera_idx,
                                                'record_time': result.get('record_time')
                                            }
                                            print(f"摄像头 {camera_idx} 检测到 {egg_num} 枚蛋，总数: {self.total_egg_count}")
                                            self.egg_count_updated.emit(self.total_egg_count)
                                            # 上传/写库：仅对“新增计数”的结果入队
                                            try:
                                                self.no_picture_result_queues[camera_idx].put(result, block=False)
                                            except queue.Full:
                                                pass
                                        else:
                                            # 重复扫描：不新增计数，但可更新为“蛋数更大”的结果（用于最终汇总展示）
                                            if egg_num > prev_num:
                                                self.detected_results[cage_id] = {
                                                    'egg_num': egg_num,
                                                    'frame_path': result.get('frame_path'),
                                                    'camera_idx': camera_idx,
                                                    'record_time': result.get('record_time')
                                                }

                    except Exception as e:
                        print(f"摄像头 {camera_idx} 结果处理异常: {e}")

                # 如果收到了任何帧，并且到达帧更新间隔，则发送帧列表给UI
                if any_frame_received and (current_time - last_frame_time >= ui_update_interval):
                    qt_images = []

                    # 将新获取的帧写入last_frames，用于缺帧时回退
                    for i in range(self.camera_count):
                        if frames[i] is not None:
                            self.last_frames[i] = frames[i]

                    # 基于当前帧或最近一帧构建输出，避免使用灰色占位图导致闪烁
                    for i in range(self.camera_count):
                        selected = frames[i] if frames[i] is not None else self.last_frames[i]
                        if selected is not None:
                            try:
                                frame_rgb = cv2.cvtColor(selected, cv2.COLOR_BGR2RGB)
                                h, w, ch = frame_rgb.shape
                                bytes_per_line = ch * w
                                # 关键修复：使用 .copy() 确保 QImage 拥有数据的副本，避免底层数据被释放导致 0xC0000005 闪退
                                image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                                qt_images.append(image)
                            except Exception as e:
                                print(f"图像转换异常: {e}")
                                # 初次启动且尚无有效帧的兜底：仅在没有last_frames时生成一次占位
                                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                                image = QImage(empty_frame.data, empty_frame.shape[1], empty_frame.shape[0],
                                               empty_frame.strides[0], QImage.Format_RGB888).copy()
                                qt_images.append(image)
                        else:
                            # 初次启动尚未获得任何帧时的占位，不会在有过有效帧后反复插入
                            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            image = QImage(empty_frame.data, empty_frame.shape[1], empty_frame.shape[0],
                                           empty_frame.strides[0], QImage.Format_RGB888).copy()
                            qt_images.append(image)

                    self.frames_generated.emit(qt_images)
                    last_frame_time = current_time

                # 控制处理速度
                time.sleep(0.05)  # 增加休眠时间
        finally:
            # 关键：在后台线程做 stop_interface + 汇总，避免UI线程卡死
            try:
                self.stop_interface()
            except Exception as e:
                print(f"停止接口收尾异常: {e}")

            try:
                summary = self.get_detection_summary()
            except Exception:
                summary = {
                    'total_egg_count': getattr(self, 'total_egg_count', 0),
                    'detected_results': getattr(self, 'detected_results', {}).copy() if hasattr(self, 'detected_results') else {}
                }
            self._last_summary = summary
            try:
                self.stop_summary_ready.emit(summary)
            except Exception:
                pass

        print("多摄像头检测接口退出")
