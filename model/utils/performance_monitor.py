# coding=utf-8
"""
性能监控模块
"""
import time
import psutil
import threading
from collections import deque
import numpy as np


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.gpu_times = deque(maxlen=window_size)
        self.cpu_times = deque(maxlen=window_size)

        self.start_time = time.time()
        self.frame_count = 0
        self.last_report_time = time.time()
        self.report_interval = 5.0  # 5秒报告一次

        self._lock = threading.Lock()

    def add_frame_time(self, duration):
        """添加帧处理时间"""
        with self._lock:
            self.frame_times.append(duration)
            self.frame_count += 1

    def add_gpu_time(self, duration):
        """添加GPU处理时间"""
        with self._lock:
            self.gpu_times.append(duration)

    def add_cpu_time(self, duration):
        """添加CPU处理时间"""
        with self._lock:
            self.cpu_times.append(duration)

    def get_stats(self):
        """获取统计信息"""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self.start_time

            stats = {
                'fps': self.frame_count / elapsed if elapsed > 0 else 0,
                'total_frames': self.frame_count,
                'elapsed_time': elapsed,
                'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
                'avg_gpu_time': np.mean(self.gpu_times) if self.gpu_times else 0,
                'avg_cpu_time': np.mean(self.cpu_times) if self.cpu_times else 0,
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent
            }

            return stats

    def report(self):
        """生成性能报告"""
        current_time = time.time()
        if current_time - self.last_report_time < self.report_interval:
            return

        self.last_report_time = current_time
        stats = self.get_stats()

        print("\n=== 性能报告 ===")
        print(f"总帧数: {stats['total_frames']}")
        print(f"平均FPS: {stats['fps']:.2f}")
        print(f"平均帧处理时间: {stats['avg_frame_time'] * 1000:.2f}ms")
        print(f"平均GPU时间: {stats['avg_gpu_time'] * 1000:.2f}ms")
        print(f"平均CPU时间: {stats['avg_cpu_time'] * 1000:.2f}ms")
        print(f"CPU使用率: {stats['cpu_percent']:.1f}%")
        print(f"内存使用率: {stats['memory_percent']:.1f}%")
        print("================\n")
