# coding=utf-8
"""
    配置包装器 - 用于多摄像头配置的管理
    @project: EGGRECORDQT
    @Author：lzy
    @file： configWrapper.py
"""


def create_multi_camera_config(cfg):
    """
    将单摄像头配置扩展为多摄像头配置

    Args:
        cfg: 原始配置字典

    Returns:
        扩展后的多摄像头配置字典
    """
    # 克隆配置以避免修改原始配置
    multi_cfg = cfg.copy()

    # 设置摄像头数量（默认6个）
    if 'camera_count' not in multi_cfg:
        multi_cfg['camera_count'] = 6

    # 为RTX 3070优化的显示和处理参数
    # 为每个摄像头指定优化的分辨率和FPS
    multi_cfg['optimal_width'] = 640  # 平衡分辨率和性能的宽度
    multi_cfg['optimal_height'] = 480  # 平衡分辨率和性能的高度
    multi_cfg['optimal_fps'] = 15  # 目标FPS

    # YOLO模型优化参数
    multi_cfg['batch_size'] = 2  # 每批处理的摄像头数量

    # 针对多摄像头的配置
    camera_configs = {}
    for i in range(multi_cfg['camera_count']):
        # 默认视频源索引为摄像头索引
        camera_config = {
            'video': i,  # 摄像头索引
            'match_range': multi_cfg['match_range'],  # 保持原始匹配范围
            'match_center': multi_cfg['match_center'],  # 保持原始匹配中心
        }

        # 针对左右两侧摄像头分组配置
        if i < 3:  # 左侧3个摄像头
            camera_config['side'] = 'left'
            # 如果需要特定设置，可以在这里添加
        else:  # 右侧3个摄像头
            camera_config['side'] = 'right'
            # 如果需要特定设置，可以在这里添加

        # 按照层级分组（每侧3个摄像头分3层）
        layer = i % 3
        camera_config['layer'] = layer

        # 保存到配置中
        camera_key = f'camera_{i}'
        camera_configs[camera_key] = camera_config

    # 将摄像头特定配置添加到主配置中
    multi_cfg.update(camera_configs)

    # 为每个摄像头创建单独的保存路径
    if 'picture_recognition_path' in multi_cfg:
        base_path = multi_cfg['picture_recognition_path']
        multi_cfg['base_picture_recognition_path'] = base_path

    return multi_cfg


def get_camera_config(multi_cfg, camera_idx):
    """
    获取特定摄像头的配置

    Args:
        multi_cfg: 多摄像头配置字典
        camera_idx: 摄像头索引

    Returns:
        该摄像头的专用配置
    """
    # 创建基础配置
    camera_cfg = multi_cfg.copy()

    # 添加摄像头特定配置
    camera_key = f'camera_{camera_idx}'
    if camera_key in multi_cfg:
        camera_cfg.update(multi_cfg[camera_key])

    # 添加摄像头索引标识
    camera_cfg['camera_idx'] = camera_idx

    # 修改图像保存路径
    if 'base_picture_recognition_path' in multi_cfg:
        base_path = multi_cfg['base_picture_recognition_path']
        camera_cfg['picture_recognition_path'] = f"{base_path}/camera_{camera_idx}"

    return camera_cfg
