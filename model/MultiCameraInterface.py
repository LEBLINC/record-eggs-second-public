# coding=utf-8
"""
    多摄像头接口，提供给QT访问（完整改进版）
    @project: EGGRECORDQT
    @Author：lzy
    @file： MultiCameraInterface.py
"""
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from model.track.yoloTrack import YOLOTrack
from model.match.matchingCounting import MatchingCounting
import cv2
import queue
from PyQt5.QtGui import QImage
import threading
import time
import numpy as np
from model.communication.SendHttp import SendHttp
from model.utils.exception import exception_handler
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import psutil

from model.utils.gpu_manager import GPUManager, GPUBatchProcessor


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
            # 直接返回固定的摄像头ID列表
            fixed_camera_ids = [0, 1, 2]  # 根据实际检测结果修改
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

        # 添加资源监控
        self.resource_monitor = ResourceMonitor()

    @exception_handler
    def init_cameras(self):
        """初始化所有摄像头（改进版）"""
        print("开始检测和初始化多摄像头...")

        # 首先检测可用摄像头
        self.available_cameras = [0, 1, 2]  # 固定ID

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

        actual_camera_idx = self.available_cameras[camera_idx]
        print(f"初始化摄像头 {camera_idx} (实际设备索引: {actual_camera_idx})")

        # 最多尝试3次
        for attempt in range(3):
            try:
                # 先释放现有连接
                if self.caps[camera_idx]:
                    self.caps[camera_idx].release()
                    time.sleep(0.1)  # 减少等待时间
                    self.caps[camera_idx] = None

                # 获取最佳backend
                backend = self.camera_detector.get_optimal_backend(actual_camera_idx)
                print(f"摄像头 {camera_idx} 尝试使用backend: {self._get_backend_name(backend)} (第{attempt + 1}次)")

                # 创建VideoCapture对象
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

            # 设置缓冲区大小为1（减少延迟）
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.optimal_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.optimal_height)

            # 设置帧率
            cap.set(cv2.CAP_PROP_FPS, self.optimal_fps)

            # 尝试设置MJPEG编码（更兼容）
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except:
                pass  # 如果不支持就跳过

            # 获取实际设置的参数
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"摄像头 {camera_idx} 实际参数: {actual_width}x{actual_height}@{actual_fps}fps")
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
            if ret and frame is not None:
                print(f"摄像头 {camera_idx} 验证成功，成功读取 1 帧")
                return True
            else:
                print(f"摄像头 {camera_idx} 验证失败，无法读取帧")
                return False

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
            time.sleep(0.3)  # 减少等待时间
            self.init_camera(camera_idx)
            self.cap_false_counts[camera_idx] = 0

    @exception_handler
    def release_camera(self, camera_idx):
        """释放指定摄像头资源"""
        if self.caps[camera_idx]:
            try:
                self.caps[camera_idx].release()
                time.sleep(0.2)  # 增加等待时间
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

        # 等待摄像头稳定（减少等待时间）
        time.sleep(0.5)

        while self.run_flag:
            try:
                self.paused.wait()

                # 监控系统资源
                self.resource_monitor.check_resources()

                # 检查摄像头健康状态
                self.check_camera_health()

                # 顺序读取每个摄像头（不使用多线程，减少竞争）
                for i in range(len(self.camera_configs)):
                    if not self.run_flag:
                        break
                    self.get_frame(i)

                # 控制帧率
                time.sleep(0.05)  # 20fps

            except Exception as e:
                print(f"摄像头读取主循环异常: {e}")
                time.sleep(0.1)

        self.release_all_cameras()
        print("多摄像头帧读取线程退出")

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
    def __init__(self, yolo_model, frame_queues, track_queues, batch_size=1):
        """
        跟踪线程，负责处理多个摄像头的目标检测和跟踪（改进版）
        """
        super().__init__()
        self.yolo_model = yolo_model
        self.frame_queues = frame_queues
        self.track_queues = track_queues
        self.batch_size = min(batch_size, len(frame_queues))
        self.camera_count = len(frame_queues)

        # 添加处理时间监控
        self.last_process_times = [0] * self.camera_count
        self.process_interval = 1.0 / 15  # 每个摄像头10FPS处理频率，降低GPU压力
        self.resource_monitor = ResourceMonitor()

    @exception_handler
    def run(self):
        print("多摄像头跟踪线程启动（改进版）")

        while self.run_flag:
            self.paused.wait()

            current_time = time.time()

            # 监控资源
            self.resource_monitor.check_resources()

            # 按批次处理摄像头
            for batch_start in range(0, self.camera_count, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.camera_count)
                batch_cameras = list(range(batch_start, batch_end))

                # 从每个摄像头的队列中获取帧
                frames = []
                camera_indices = []

                for camera_idx in batch_cameras:
                    try:
                        if current_time - self.last_process_times[camera_idx] < self.process_interval:
                            continue

                        frame = self.frame_queues[camera_idx].get(timeout=0.1)
                        frames.append(frame)
                        camera_indices.append(camera_idx)
                    except queue.Empty:
                        continue

                if not frames:
                    continue

                # 批量处理跟踪
                for i, (frame, camera_idx) in enumerate(zip(frames, camera_indices)):
                    try:
                        track_results = self.yolo_model.track(frame)

                        # 非阻塞放入结果队列
                        try:
                            self.track_queues[camera_idx].put((frame, track_results), block=False)
                        except queue.Full:
                            # 队列满时清理旧结果
                            try:
                                self.track_queues[camera_idx].get(block=False)
                                self.track_queues[camera_idx].put((frame, track_results), block=False)
                            except (queue.Empty, queue.Full):
                                pass

                        self.last_process_times[camera_idx] = current_time
                    except Exception as e:
                        print(f"摄像头 {camera_idx} 跟踪处理异常: {e}")

            # 适当休眠，避免线程消耗过多资源
            time.sleep(0.05)

        print("多摄像头跟踪线程退出")


class MultiCameraMatchThread(BaseThread):
    def __init__(self, matching_instances, track_queues, result_queues):
        """
        匹配线程，负责处理多个摄像头的匹配计数（改进版）
        """
        super().__init__()
        self.matching_instances = matching_instances
        self.track_queues = track_queues
        self.result_queues = result_queues
        self.camera_count = len(track_queues)

        # 添加错误恢复机制
        self.error_counts = [0] * self.camera_count
        self.max_errors = 5  # 最大错误次数，超过后重置匹配实例

    @exception_handler
    def run(self):
        print("多摄像头匹配线程启动（改进版）")

        while self.run_flag:
            self.paused.wait()

            # 创建线程池来并发处理每个摄像头的匹配
            threads = []
            for i in range(self.camera_count):
                thread = threading.Thread(target=self.process_camera, args=(i,))
                thread.daemon = True
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=0.1)

            # 控制处理速度
            time.sleep(0.02)

        print("多摄像头匹配线程退出")

    @exception_handler
    def process_camera(self, camera_idx):
        """处理单个摄像头的匹配计数"""
        try:
            frame, track_results = self.track_queues[camera_idx].get(timeout=0.1)

            # 检查是否有跟踪ID
            if track_results[0].boxes.id is not None:
                try:
                    self.matching_instances[camera_idx].match(track_results, frame)
                except Exception as e:
                    print(f"摄像头 {camera_idx} 匹配过程异常: {e}")
                    self.error_counts[camera_idx] += 1

                    # 如果错误次数过多，尝试重置匹配实例
                    if self.error_counts[camera_idx] >= self.max_errors:
                        print(f"摄像头 {camera_idx} 错误次数过多，尝试重置匹配实例")
                        try:
                            # 重置匹配实例（如果MatchingCounting类有reset方法）
                            if hasattr(self.matching_instances[camera_idx], 'reset'):
                                self.matching_instances[camera_idx].reset()
                            self.error_counts[camera_idx] = 0
                        except Exception as reset_e:
                            print(f"摄像头 {camera_idx} 重置匹配实例失败: {reset_e}")

                    # 生成空结果，确保处理流程继续
                    match_results = []
                    self.result_queues[camera_idx].put((frame, match_results), block=False)
                    return

            try:
                # 确保数据类型转换
                match_results = self.matching_instances[camera_idx].update_and_delete_records()

                # 确保匹配结果是可迭代对象
                if not isinstance(match_results, (list, tuple)):
                    if match_results is None:
                        match_results = []
                    else:
                        match_results = [match_results]

                # 重置错误计数
                self.error_counts[camera_idx] = 0
            except Exception as e:
                print(f"摄像头 {camera_idx} 更新记录异常: {e}")
                match_results = []
                self.error_counts[camera_idx] += 1

            # 非阻塞放入结果队列
            try:
                self.result_queues[camera_idx].put((frame, match_results), block=False)
            except queue.Full:
                # 清理旧结果
                try:
                    self.result_queues[camera_idx].get(block=False)
                    self.result_queues[camera_idx].put((frame, match_results), block=False)
                except (queue.Empty, queue.Full):
                    pass

        except queue.Empty:
            pass
        except Exception as e:
            print(f"摄像头 {camera_idx} 匹配处理异常: {e}")
            self.error_counts[camera_idx] += 1


class MultiCameraHTTPThread(BaseThread):
    def __init__(self, no_picture_result_queues, send_http_instances):
        """
        HTTP上传线程，负责处理多个摄像头的上传任务（改进版）
        """
        super().__init__()
        self.no_picture_result_queues = no_picture_result_queues
        self.send_http_instances = send_http_instances
        self.camera_count = len(no_picture_result_queues)

        # 添加错误恢复机制
        self.error_counts = [0] * self.camera_count
        self.max_errors = 5  # 最大错误次数，超过后暂停该摄像头的HTTP上传
        self.camera_paused = [False] * self.camera_count  # 记录摄像头HTTP上传暂停状态
        self.pause_duration = 60  # 暂停上传的秒数
        self.pause_start_times = [0] * self.camera_count  # 记录摄像头HTTP上传暂停开始时间

    @exception_handler
    def run(self):
        print("多摄像头HTTP上传线程启动（改进版）")

        while self.run_flag:
            self.paused.wait()

            current_time = time.time()
            has_data = False

            for camera_idx in range(self.camera_count):
                # 检查是否需要恢复暂停的摄像头
                if self.camera_paused[camera_idx]:
                    if current_time - self.pause_start_times[camera_idx] > self.pause_duration:
                        print(f"恢复摄像头 {camera_idx} 的HTTP上传")
                        self.camera_paused[camera_idx] = False
                        self.error_counts[camera_idx] = 0
                    else:
                        continue  # 跳过暂停中的摄像头

                try:
                    # 非阻塞方式获取数据
                    try:
                        no_picture_result = self.no_picture_result_queues[camera_idx].get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # 添加摄像头索引到结果中
                    no_picture_result['camera_idx'] = camera_idx

                    # 确保结果是字典类型
                    if not isinstance(no_picture_result, dict):
                        print(f"摄像头 {camera_idx} 结果类型错误: {type(no_picture_result)}")
                        continue

                    # 尝试HTTP上传
                    try:
                        response = self.send_http_instances[camera_idx].http_post(no_picture_result)

                        # 检查HTTP响应状态
                        if hasattr(response, 'status_code') and response.status_code != 200:
                            print(f"摄像头 {camera_idx} HTTP上传返回非200状态码: {response}")
                            self.error_counts[camera_idx] += 1
                        else:
                            # 成功上传，重置错误计数
                            self.error_counts[camera_idx] = 0
                    except Exception as e:
                        print(f"摄像头 {camera_idx} HTTP上传异常: {e}")
                        self.error_counts[camera_idx] += 1

                    # 检查错误次数，是否需要暂停该摄像头的上传
                    if self.error_counts[camera_idx] >= self.max_errors:
                        print(f"摄像头 {camera_idx} HTTP上传错误次数过多，暂停上传 {self.pause_duration} 秒")
                        self.camera_paused[camera_idx] = True
                        self.pause_start_times[camera_idx] = current_time

                    has_data = True
                except Exception as e:
                    print(f"摄像头 {camera_idx} HTTP处理异常: {e}")

            # 如果没有数据，增加休眠时间
            if not has_data:
                time.sleep(0.1)
            else:
                time.sleep(0.02)

        print("多摄像头HTTP上传线程退出")


class MultiCameraInterface(QThread):
    frames_generated = pyqtSignal(list)  # 发送所有摄像头的帧列表

    def __init__(self, cfg):
        """
        多摄像头接口（完整改进版）
        """
        super().__init__()
        self.global_cfg = cfg

        # 首先检测可用摄像头
        detector = CameraDetector()
        available_cameras = detector.detect_available_cameras()

        if not available_cameras:
            raise RuntimeError("未检测到任何可用摄像头！请检查摄像头连接。")

        # 根据实际可用摄像头调整配置
        requested_count = cfg.get('camera_count', 6)
        actual_count = min(requested_count, len(available_cameras))

        if actual_count < requested_count:
            print(f"警告：请求 {requested_count} 个摄像头，但只检测到 {len(available_cameras)} 个")
            print(f"将使用 {actual_count} 个摄像头: {available_cameras[:actual_count]}")

        cfg['camera_count'] = actual_count

        # 创建摄像头配置
        self.camera_configs = self._create_camera_configs(cfg, available_cameras[:actual_count])
        self.camera_count = len(self.camera_configs)

        # 为每个摄像头创建队列，减小队列大小以避免内存积压
        self.frame_queues = [queue.Queue(maxsize=2) for _ in range(self.camera_count)]
        self.track_queues = [queue.Queue(maxsize=3) for _ in range(self.camera_count)]
        self.result_queues = [queue.Queue() for _ in range(self.camera_count)]
        self.no_picture_result_queues = [queue.Queue() for _ in range(self.camera_count)]

        # 创建YOLO模型 (共享模型以节省GPU内存)
        self.yolo_model = YOLOTrack(cfg)

        # 为每个摄像头创建匹配实例
        self.matching_instances = [MatchingCounting(self._adjust_config_for_camera(cfg, i))
                                   for i in range(self.camera_count)]

        # 为每个摄像头创建HTTP上传实例
        self.send_http_instances = [SendHttp(self._adjust_config_for_camera(cfg, i))
                                    for i in range(self.camera_count)]

        # 控制标志
        self.run_flag = True
        self.paused = threading.Event()
        self.paused.set()  # 开始时不暂停

        # 添加资源监控
        self.resource_monitor = ResourceMonitor()

        # 创建线程 - 使用已存在的方法
        self._create_threads()  # 修改这里，使用正确的方法名

        print(f"多摄像头接口初始化完成，将管理 {self.camera_count} 个摄像头")


    def _create_camera_configs(self, global_cfg, available_camera_indices):
        """根据实际可用摄像头创建配置"""
        camera_configs = []

        for i, actual_idx in enumerate(available_camera_indices):
            camera_cfg = global_cfg.copy()
            camera_cfg['camera_idx'] = i
            camera_cfg['actual_video_idx'] = actual_idx

            # 如果有特定配置则使用
            camera_key = f'camera_{i}'
            if camera_key in global_cfg:
                camera_cfg.update(global_cfg[camera_key])

            # 使用实际检测到的摄像头索引
            camera_cfg['video'] = actual_idx

            camera_configs.append(camera_cfg)

        return camera_configs

    def _adjust_config_for_camera(self, cfg, camera_idx):
        """为特定摄像头调整配置"""
        camera_cfg = cfg.copy()

        # 修改特定于摄像头的设置
        camera_cfg['camera_idx'] = camera_idx

        # 修改文件保存路径，为每个摄像头创建子目录
        if 'picture_recognition_path' in camera_cfg:
            base_path = camera_cfg['picture_recognition_path']
            camera_cfg['picture_recognition_path'] = f"{base_path}/camera_{camera_idx}"

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

        # 跟踪线程 - 使用批处理提高GPU利用率
        batch_size = 1  # 保守设置，避免GPU过载
        self.track_thread = MultiCameraTrackThread(
            self.yolo_model, self.frame_queues, self.track_queues, batch_size
        )

        # 匹配线程
        self.match_thread = MultiCameraMatchThread(
            self.matching_instances, self.track_queues, self.result_queues
        )

        # HTTP上传线程
        self.http_thread = MultiCameraHTTPThread(
            self.no_picture_result_queues, self.send_http_instances
        )

    def start_interface(self):
        """启动所有线程"""
        print("启动多摄像头接口所有线程")
        self.frame_thread.start()
        time.sleep(0.2)  # 减少等待时间
        self.track_thread.start()
        time.sleep(0.1)  # 减少等待时间
        self.match_thread.start()
        time.sleep(0.1)  # 减少等待时间
        self.http_thread.start()

    def stop_interface(self):
        """停止所有线程"""
        print("停止多摄像头接口所有线程")
        self.frame_thread.stop()
        self.track_thread.stop()
        self.match_thread.stop()
        self.http_thread.stop()

        # 等待线程完成
        self.frame_thread.wait(5000)  # 5秒超时
        self.track_thread.wait(5000)
        self.match_thread.wait(5000)
        self.http_thread.wait(5000)

    def resume_interface(self):
        """恢复所有线程"""
        self.frame_thread.resume()
        self.track_thread.resume()
        self.match_thread.resume()
        self.http_thread.resume()

    def pause_interface(self):
        """暂停所有线程"""
        self.frame_thread.pause()
        self.track_thread.pause()
        self.match_thread.pause()
        self.http_thread.pause()

    def stop(self):
        """停止接口"""
        self.run_flag = False
        self.stop_interface()
        self.resume()  # 确保线程不会被暂停事件阻塞

    def pause(self):
        """暂停接口"""
        self.paused.clear()
        self.pause_interface()

    def resume(self):
        """恢复接口"""
        self.paused.set()
        self.resume_interface()

    @exception_handler
    def run(self):
        """主线程运行函数（改进版）"""
        print("初始化多摄像头检测接口（改进版）")
        self.start_interface()

        last_frame_time = time.time()
        ui_update_interval = 1.0 / 15  # UI更新频率10fps，减少负担

        while self.run_flag:
            self.paused.wait()

            current_time = time.time()

            # 控制UI更新频率
            if current_time - last_frame_time < ui_update_interval:
                time.sleep(0.02)
                continue

            # 监控系统资源
            self.resource_monitor.check_resources()

            # 收集所有摄像头的当前帧
            frames = [None] * self.camera_count
            any_frame_received = False

            for camera_idx in range(self.camera_count):
                try:
                    frame, match_results = self.result_queues[camera_idx].get(timeout=0.1)
                    frames[camera_idx] = frame
                    any_frame_received = True

                    # 将检测结果放入上传队列
                    for match_result in match_results:
                        try:
                            self.no_picture_result_queues[camera_idx].put(match_result, block=False)
                        except queue.Full:
                            pass  # 队列满，丢弃当前结果

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"摄像头 {camera_idx} 结果处理异常: {e}")

            # 如果收到了任何帧，就发送帧列表给UI
            if any_frame_received:
                # 将OpenCV帧转换为PyQt图像
                qt_images = []

                for frame in frames:
                    if frame is not None:
                        try:
                            # BGR转RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # 创建QImage
                            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                                           frame_rgb.strides[0], QImage.Format_RGB888)
                            qt_images.append(image)
                        except Exception as e:
                            print(f"图像转换异常: {e}")
                            # 创建空白图像
                            empty_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
                            image = QImage(empty_frame.data, empty_frame.shape[1], empty_frame.shape[0],
                                           empty_frame.strides[0], QImage.Format_RGB888)
                            qt_images.append(image)
                    else:
                        # 创建空白图像占位
                        empty_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # 灰色背景
                        image = QImage(empty_frame.data, empty_frame.shape[1], empty_frame.shape[0],
                                       empty_frame.strides[0], QImage.Format_RGB888)
                        qt_images.append(image)

                # 发送所有图像到UI
                self.frames_generated.emit(qt_images)
                last_frame_time = current_time

            # 控制处理速度
            time.sleep(0.05)  # 增加休眠时间

        print("多摄像头检测接口退出")