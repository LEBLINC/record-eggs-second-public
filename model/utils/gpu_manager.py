# coding=utf-8
"""
GPU资源管理器 - 针对RTX 4060Ti优化
"""
import torch
import numpy as np
import cv2
from torch.nn.functional import interpolate
import warnings
try:
    # `pynvml`（deprecated）与 `nvidia-ml-py`（recommended）在导入名上都是 `pynvml`。
    # 一些环境会把包安装到用户级 site-packages（AppData/Roaming），并可能被上层代码禁用 usersite。
    # 因此这里做“可选依赖”处理：缺失时不影响主流程，仅禁用 NVML 监控功能。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml  # type: ignore
except Exception:
    pynvml = None
import threading
import time
from contextlib import contextmanager
from queue import Queue
import logging


class GPUManager:
    """GPU资源管理器"""

    def __init__(self):
        self.device = None
        self.gpu_info = {}
        self.streams = {}
        self.memory_pool = None
        self._lock = threading.Lock()

        # 初始化NVML（可选）
        self.nvml_available = False
        self.gpu_handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml_available = False
                self.gpu_handle = None

        self._initialize_gpu()

    def _initialize_gpu(self):
        """初始化GPU设置"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

            # 获取GPU信息
            self.gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'capability': torch.cuda.get_device_capability(0),
                'total_memory': torch.cuda.get_device_properties(0).total_memory,
                'multi_processor_count': torch.cuda.get_device_properties(0).multi_processor_count
            }

            print(f"检测到GPU: {self.gpu_info['name']}")
            print(f"显存大小: {self.gpu_info['total_memory'] / 1024 ** 3:.2f} GB")
            print(f"SM数量: {self.gpu_info['multi_processor_count']}")

            # RTX 4060Ti优化设置
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
            torch.backends.cudnn.allow_tf32 = True

            # 设置内存分配策略
            torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存
            torch.cuda.empty_cache()

            # 创建CUDA流池（为每个摄像头创建独立的流）
            self.create_stream_pool(6)  # 6个摄像头

        else:
            raise RuntimeError("未检测到可用的GPU")

    def create_stream_pool(self, num_streams):
        """创建CUDA流池"""
        self.streams = {
            'preprocess': [torch.cuda.Stream() for _ in range(num_streams)],
            'inference': [torch.cuda.Stream() for _ in range(num_streams)],
            'postprocess': [torch.cuda.Stream() for _ in range(num_streams)]
        }

    def get_stream(self, stream_type, camera_idx):
        """获取指定类型和摄像头的CUDA流"""
        return self.streams.get(stream_type, [None] * 6)[camera_idx]

    @contextmanager
    def cuda_stream_context(self, stream_type, camera_idx):
        """CUDA流上下文管理器"""
        stream = self.get_stream(stream_type, camera_idx)
        if stream:
            with torch.cuda.stream(stream):
                yield stream
        else:
            yield None

    def get_memory_info(self):
        """获取GPU内存信息"""
        if self.nvml_available and pynvml is not None and self.gpu_handle is not None:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return {
                'total': info.total / 1024 ** 3,
                'used': info.used / 1024 ** 3,
                'free': info.free / 1024 ** 3,
                'percent': (info.used / info.total) * 100
            }
        else:
            return {
                'total': self.gpu_info['total_memory'] / 1024 ** 3,
                'used': torch.cuda.memory_allocated() / 1024 ** 3,
                'free': (self.gpu_info['total_memory'] - torch.cuda.memory_allocated()) / 1024 ** 3,
                'percent': (torch.cuda.memory_allocated() / self.gpu_info['total_memory']) * 100
            }

    def get_gpu_utilization(self):
        """获取GPU利用率"""
        if self.nvml_available and pynvml is not None and self.gpu_handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return util.gpu
        return -1


class GPUImageProcessor:
    """GPU加速的图像处理器"""

    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.device

        # 预分配GPU张量缓存
        self.tensor_cache = {}

    def preprocess_batch_gpu(self, frames, target_size=(640, 640), camera_indices=None):
        """GPU批量预处理"""
        if not frames:
            return None

        batch_size = len(frames)
        camera_indices = camera_indices or list(range(batch_size))

        # 使用不同的流并行处理每个摄像头的图像
        processed_tensors = []

        for i, (frame, cam_idx) in enumerate(zip(frames, camera_indices)):
            with self.gpu_manager.cuda_stream_context('preprocess', cam_idx):
                # 转换为tensor并移到GPU
                tensor = torch.from_numpy(frame).to(self.device, non_blocking=True)
                tensor = tensor.permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0)

                # GPU上进行resize
                resized = interpolate(
                    tensor,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

                processed_tensors.append(resized)

        # 同步所有流
        torch.cuda.synchronize()

        # 合并为批次
        batch_tensor = torch.cat(processed_tensors, dim=0)
        return batch_tensor

    def postprocess_batch_gpu(self, results, original_sizes):
        """GPU批量后处理"""
        # 在GPU上进行坐标变换等后处理
        processed_results = []

        for i, (result, orig_size) in enumerate(zip(results, original_sizes)):
            with self.gpu_manager.cuda_stream_context('postprocess', i):
                # 这里可以添加GPU加速的后处理逻辑
                processed_results.append(result)

        return processed_results


class GPUBatchProcessor:
    """GPU批处理管理器"""

    def __init__(self, gpu_manager, batch_size=3, timeout=0.1):
        self.gpu_manager = gpu_manager
        self.batch_size = batch_size
        self.timeout = timeout
        self.batch_queue = Queue(maxsize=10)
        self.result_queues = {}

    def add_frame(self, camera_idx, frame):
        """添加帧到批处理队列"""
        try:
            self.batch_queue.put((camera_idx, frame), timeout=0.01)
            return True
        except:
            return False

    def process_batch(self, process_func):
        """处理一个批次"""
        batch = []
        camera_indices = []

        # 收集批次
        while len(batch) < self.batch_size:
            try:
                camera_idx, frame = self.batch_queue.get(timeout=self.timeout)
                batch.append(frame)
                camera_indices.append(camera_idx)
            except:
                break

        if not batch:
            return []

        # 批量处理
        with torch.cuda.amp.autocast():  # 使用混合精度
            results = process_func(batch, camera_indices)

        return list(zip(camera_indices, results))
