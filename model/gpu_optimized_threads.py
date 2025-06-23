# coding=utf-8
"""
GPU优化的多摄像头处理线程
"""
import torch
import threading
from queue import Queue, Empty
import numpy as np
import time
from model.utils.exception import exception_handler
from concurrent.futures import ThreadPoolExecutor
import asyncio


class GPUOptimizedTrackThread(threading.Thread):
    """GPU优化的跟踪线程"""

    def __init__(self, yolo_model, frame_queues, track_queues, gpu_manager):
        super().__init__()
        self.yolo_model = yolo_model
        self.frame_queues = frame_queues
        self.track_queues = track_queues
        self.gpu_manager = gpu_manager
        self.camera_count = len(frame_queues)
        self.run_flag = True

        # 批处理设置
        self.batch_size = 3  # RTX 4060Ti可以处理更大的批次
        self.batch_timeout = 0.05  # 50ms超时

        # 创建批处理缓冲区
        self.batch_buffer = []
        self.camera_buffer = []
        self.last_batch_time = time.time()

    @exception_handler
    def run(self):
        """主运行循环"""
        print("GPU优化跟踪线程启动")

        while self.run_flag:
            try:
                # 收集帧到批处理缓冲区
                self._collect_frames()

                # 检查是否需要处理批次
                if self._should_process_batch():
                    self._process_batch()

                # 避免CPU空转
                time.sleep(0.001)

            except Exception as e:
                print(f"GPU跟踪线程异常: {e}")

        print("GPU优化跟踪线程退出")

    def _collect_frames(self):
        """收集帧到批处理缓冲区"""
        for camera_idx in range(self.camera_count):
            try:
                # 非阻塞获取帧
                frame = self.frame_queues[camera_idx].get_nowait()
                self.batch_buffer.append(frame)
                self.camera_buffer.append(camera_idx)

                # 如果缓冲区满了，立即处理
                if len(self.batch_buffer) >= self.batch_size:
                    self._process_batch()

            except Empty:
                continue

    def _should_process_batch(self):
        """判断是否应该处理批次"""
        if not self.batch_buffer:
            return False

        # 缓冲区满或超时
        current_time = time.time()
        if (len(self.batch_buffer) >= self.batch_size or
                current_time - self.last_batch_time > self.batch_timeout):
            return True

        return False

    def _process_batch(self):
        """处理批次"""
        if not self.batch_buffer:
            return

        # 复制缓冲区内容
        frames = self.batch_buffer.copy()
        camera_indices = self.camera_buffer.copy()

        # 清空缓冲区
        self.batch_buffer.clear()
        self.camera_buffer.clear()
        self.last_batch_time = time.time()

        # GPU批量处理
        try:
            # 批量跟踪
            results = self.yolo_model.batch_track(frames, camera_indices)

            # 分发结果
            for (camera_idx, result), frame in zip(
                    zip(camera_indices, results), frames):
                try:
                    self.track_queues[camera_idx].put_nowait((frame, result))
                except:
                    # 队列满，丢弃旧数据
                    try:
                        self.track_queues[camera_idx].get_nowait()
                        self.track_queues[camera_idx].put_nowait((frame, result))
                    except:
                        pass

        except Exception as e:
            print(f"批处理异常: {e}")

    def stop(self):
        """停止线程"""
        self.run_flag = False


class GPUMemoryOptimizer(threading.Thread):
    """GPU内存优化器线程"""

    def __init__(self, gpu_manager, interval=10):
        super().__init__()
        self.gpu_manager = gpu_manager
        self.interval = interval
        self.run_flag = True
        self.daemon = True

    def run(self):
        """定期清理GPU内存"""
        while self.run_flag:
            try:
                # 获取内存信息
                mem_info = self.gpu_manager.get_memory_info()

                # 如果内存使用率超过85%，清理缓存
                if mem_info['percent'] > 85:
                    print(f"GPU内存使用率 {mem_info['percent']:.1f}%，清理缓存...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                time.sleep(self.interval)

            except Exception as e:
                print(f"GPU内存优化器异常: {e}")

    def stop(self):
        self.run_flag = False


class AsyncBatchProcessor:
    """异步批处理器"""

    def __init__(self, process_func, batch_size=3, timeout=0.1):
        self.process_func = process_func
        self.batch_size = batch_size
        self.timeout = timeout
        self.loop = None
        self.queue = asyncio.Queue(maxsize=100)

    async def add_item(self, item):
        """异步添加项目"""
        await self.queue.put(item)

    async def process_batches(self):
        """异步处理批次"""
        while True:
            batch = []

            # 收集批次
            try:
                # 等待第一个项目
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.timeout
                )
                batch.append(item)

                # 尝试填满批次
                while len(batch) < self.batch_size:
                    try:
                        item = self.queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

            except asyncio.TimeoutError:
                if not batch:
                    continue

            # 处理批次
            if batch:
                await self._process_batch_async(batch)

    async def _process_batch_async(self, batch):
        """异步处理单个批次"""
        loop = asyncio.get_event_loop()
        # 在线程池中运行CPU密集型任务
        result = await loop.run_in_executor(
            None,
            self.process_func,
            batch
        )
        return result
