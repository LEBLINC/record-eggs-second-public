# coding=utf-8
"""
    配置示例
    @project: EGGRECORDQT
    @Author：lzy
    @file： config_sample.py
"""
import cv2


def get_default_config():
    """返回默认配置"""
    return {
        # 基础配置
        'mode': 0,  # 0:工作模式, 1:采集模式, 2:示范模式
        'upload': True,  # 是否自动上传
        'uploadUrl': 'http://8.138.181.75/leblinc_wenshi_eggCage/cageDuck/eggRecord/save',  # 上传地址
        'deviceId': 'hn_xy_device_0001',  # 设备ID
        'upload_interval': 300,  # 上传间隔（秒）

        # 视频配置
        'width': 1920,  # 视频宽度
        'height': 1080,  # 视频高度
        'video': 0,  # 默认视频源索引
        'picture_save_path': 'D:/DuckEggData',  # 图片保存路径
        'picture_recognition_path': 'D:/DuckEggData/Recognition',  # 识别图片保存路径

        # 目标检测配置
        'modelPath': 'resources/best.pt',  # 模型路径
        'tracking_config': 'configs/ocsort.yaml',  # 跟踪配置文件
        'imgsz': 640,  # 输入图像尺寸
        'conf': 0.5,  # 置信度阈值
        'iou': 0.5,  # IOU阈值

        # 匹配计数配置
        'match_center': 640,  # 匹配中心点
        'match_range': 300,  # 匹配范围

        # 示范模式配置
        'demo_video': 'D:/DuckEggData/demo.mp4',  # 示范视频路径

        # 多摄像头配置
        'camera_count': 6,  # 摄像头数量
        # 摄像头初始化优化参数
        'camera_detection': {
            'max_detection_time': 5,  # 最大检测时间5秒
            'quick_verify_frames': 1,  # 只验证1帧确认可用
            'backend_priority': [cv2.CAP_DSHOW],  # 只使用DirectShow
            'parallel_init': False,  # 暂时关闭并行初始化
        },

        # 线程配置优化
        'thread_management': {
            'frame_queue_size': 2,  # 减小队列大小
            'track_queue_size': 2,
            'result_queue_size': 3,
            'thread_sleep_time': 0.02,  # 线程休眠时间
            'ui_update_fps': 10,  # UI更新频率
        },

        # RTX 4060Ti优化参数
        'batch_size': 2,  # 批处理大小（每批处理的摄像头数量）
        'optimal_width': 640,  # 最佳处理宽度
        'optimal_height': 480,  # 最佳处理高度
        'optimal_fps': 15,  # 最佳帧率

        # GPU优化配置
        'gpu_optimization': {
            'use_fp16': False,  # 暂时关闭半精度，避免类型不匹配
            'use_tf32': True,  # 启用TF32加速（RTX 4060Ti支持）
        },

        # 摄像头配置（保持原有配置）
        'camera_0': {
            'video': 0,
            'side': 'left',
            'layer': 0,
            'use_global_params': True,
        },
        'camera_1': {
            'video': 1,
            'side': 'left',
            'layer': 1,
            'use_global_params': True,
        },
        'camera_2': {
            'video': 2,
            'side': 'left',
            'layer': 2,
            'use_global_params': True,
        },
        'camera_3': {
            'video': 3,
            'side': 'right',
            'layer': 0,
            'use_global_params': True,
        },
        'camera_4': {
            'video': 4,
            'side': 'right',
            'layer': 1,
            'use_global_params': True,
        },
        'camera_5': {
            'video': 5,
            'side': 'right',
            'layer': 2,
            'use_global_params': True,
        },
    }
