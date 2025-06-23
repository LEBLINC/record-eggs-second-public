# coding=utf-8
"""
多摄像头优化版YOLO跟踪模块
@project: EGGRECORDQT
@Author：lzy
@file： YOLOTrack.py
@date：2023/6/15 15:02
"""
from ultralytics import YOLO
from model.track.tracker_zoo import on_predict_start
from functools import partial
import torch
import cv2
import numpy as np
from model.utils.exception import exception_handler


class YOLOTrack:
    def __init__(self, cfg):
        """
        初始化YOLO跟踪模型
        Args:
            cfg: 配置字典
        """
        # 加载模型
        self.model = YOLO(cfg['modelPath'], task='detect')

        # 添加跟踪回调
        self.model.add_callback('on_predict_start',
                                partial(on_predict_start, persist=True, tracking_config=cfg['tracking_config']))

        # 从配置中获取参数
        self.imgsz = cfg.get('imgsz', 640)
        self.conf = cfg.get('conf', 0.5)
        self.iou = cfg.get('iou', 0.5)

        # 检查并设置GPU设备
        self.device = self._get_device()

        # 检查是否支持半精度
        self.half = False  # 先默认关闭半精度
        if self.device != 'cpu':
            # 检查GPU是否支持半精度
            if torch.cuda.get_device_capability()[0] >= 7:  # compute capability >= 7.0
                self.half = cfg.get('gpu_optimization', {}).get('use_fp16', False)
                if self.half:
                    print("启用半精度推理")
                    try:
                        self.model.model.half()  # 转换模型到半精度
                    except Exception as e:
                        print(f"半精度转换失败: {e}，使用全精度")
                        self.half = False

        # 预热模型
        self._warmup()

    @exception_handler
    def _get_device(self):
        """确定最佳设备"""
        try:
            if torch.cuda.is_available():
                # 如果有CUDA设备，使用第一个设备
                device = f"cuda:0"
                # 设置CUDA工作流
                torch.backends.cudnn.benchmark = True
                # 启用TF32（RTX 4060Ti支持）
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # 打印设备信息
                device_name = torch.cuda.get_device_name(0)
                print(f"使用GPU设备: {device_name}")
                # 获取GPU内存信息
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                mem_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
                mem_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                print(f"GPU内存: 总计 {mem_total:.2f}GB, 已分配 {mem_allocated:.2f}GB, 已保留 {mem_reserved:.2f}GB")
            else:
                # 否则使用CPU
                device = "cpu"
                print("CUDA不可用，使用CPU。如果你有NVIDIA显卡，请确保已安装CUDA和GPU版本的PyTorch。")
        except Exception as e:
            print(f"获取设备信息时出错: {e}")
            device = "cpu"
            print("出现异常，将使用CPU")

        return device

    @exception_handler
    def _warmup(self):
        """预热模型，以减少首次推理时间"""
        print("预热YOLO模型...")
        try:
            # 创建一个小尺寸的随机图像
            dummy_input = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

            # 直接使用track方法进行预热，避免类型不匹配
            _ = self.model.track(
                dummy_input,
                persist=True,
                conf=0.25,
                iou=0.45,
                verbose=False,
                imgsz=320,
                device=self.device,
                half=self.half  # 使用配置的半精度设置
            )

            # 强制清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("YOLO模型预热完成")
        except Exception as e:
            print(f"模型预热异常: {e}")
            # 如果半精度预热失败，尝试关闭半精度
            if self.half:
                print("尝试关闭半精度重新预热...")
                self.half = False
                try:
                    _ = self.model.track(
                        dummy_input,
                        persist=True,
                        conf=0.25,
                        iou=0.45,
                        verbose=False,
                        imgsz=320,
                        device=self.device,
                        half=False
                    )
                    print("全精度模式预热成功")
                except Exception as e2:
                    print(f"全精度预热也失败: {e2}")

    @exception_handler
    def track(self, frame):
        """
        使用YOLO模型进行目标跟踪
        Args:
            frame: 输入帧
        Returns:
            跟踪结果
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            show=False,
            save_txt=False,
            show_labels=False,
            verbose=False,
            exist_ok=False,
            imgsz=self.imgsz,
            vid_stride=1,
            line_width=None,
            device=self.device,
            half=self.half  # 使用配置的半精度设置
        )
        return results

    @exception_handler
    def batch_track(self, frames):
        """
        批量处理多个帧
        Args:
            frames: 包含多个帧的列表
        Returns:
            包含每个帧跟踪结果的列表
        """
        if not frames:
            return []

        # 批量推理
        results = self.model.track(
            frames,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            show=False,
            save_txt=False,
            show_labels=False,
            verbose=False,
            exist_ok=False,
            imgsz=self.imgsz,
            vid_stride=1,
            line_width=None,
            device=self.device,
            batch=len(frames),
            half=self.half  # 使用配置的半精度设置
        )

        return results

    def __del__(self):
        """清理资源"""
        self.model = None
        # 清除CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
