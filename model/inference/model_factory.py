# coding=utf-8
"""
模型工厂模块
根据配置选择并创建检测器和匹配器实例，保持向后兼容性
@project: EGGRECORDQT
@Author：lzy
@file： model_factory.py
"""
import logging

logger = logging.getLogger(__name__)

# 默认 RTMDet 配置（当 cfg['rtmdet'] 中缺少对应键时使用）
_RTMDET_DEFAULTS = {
    'imgsz': 640,
    'conf': 0.5,
    'iou': 0.5,
    'validity_threshold': 0.7,
    'use_fp16': False,
}

# 默认 topology_matcher 配置
_TOPOLOGY_MATCHER_DEFAULTS = {
    'max_match_distance': 200,
    'cost_weights': {
        'position': 0.6,
        'validity': 0.3,
        'topology': 0.1,
    },
    'validity_threshold': 0.7,
    'min_appear_frames': 5,
}


def _apply_defaults(cfg_section: dict, defaults: dict, section_name: str) -> dict:
    """
    将 defaults 中缺失的键填充到 cfg_section，并对每个缺失键记录警告日志。

    Args:
        cfg_section:  原始配置子字典（可为 None，此时视为空字典）
        defaults:     默认值字典
        section_name: 配置节名称（用于日志）

    Returns:
        填充了默认值的配置字典（不修改原始对象）
    """
    result = dict(cfg_section) if cfg_section else {}

    for key, default_val in defaults.items():
        if key not in result:
            logger.warning(
                "配置节 '%s' 缺少键 '%s'，使用默认值: %s",
                section_name, key, default_val,
            )
            result[key] = default_val

    return result


def create_detector(cfg: dict):
    """
    根据 cfg['model_type'] 创建并返回对应的检测器实例。

    - model_type == 'yolo'   → 返回 YOLOTrack(cfg)（现有行为不变）
    - model_type == 'rtmdet' → 提取 cfg['rtmdet'] 子字典，
                               填充缺失默认值后返回 RTMDetInferenceEngine(rtmdet_cfg)
    - 其他值                 → 记录警告并回退到 'yolo'

    Args:
        cfg: 顶层配置字典，至少包含 'model_type' 键（缺失时默认 'yolo'）

    Returns:
        YOLOTrack 或 RTMDetInferenceEngine 实例

    Raises:
        KeyError:          当 model_type='rtmdet' 但 cfg 中缺少 'rtmdet' 节时
        FileNotFoundError: 当模型文件路径不存在时（由 RTMDetInferenceEngine 抛出）
    """
    model_type = cfg.get('model_type', 'yolo')

    if model_type not in ('yolo', 'rtmdet'):
        logger.warning(
            "未知的 model_type '%s'，回退到 'yolo'。"
            "支持的值为: 'yolo', 'rtmdet'",
            model_type,
        )
        model_type = 'yolo'

    if model_type == 'rtmdet':
        from model.inference.rtmdet_engine import RTMDetInferenceEngine

        # 采集模式（mode=1）不需要加载推理模型，避免 ONNX 文件不存在时报错
        mode = cfg.get('mode', 0)
        if mode == 1:
            logger.info("mode=1（采集模式），跳过 RTMDet 模型加载")
            return None

        rtmdet_cfg_raw = cfg.get('rtmdet')
        if rtmdet_cfg_raw is None:
            raise KeyError(
                "model_type 为 'rtmdet' 但配置中缺少 'rtmdet' 节。"
                "请在 configs/config.yaml 中添加 rtmdet 配置块。"
            )

        rtmdet_cfg = _apply_defaults(rtmdet_cfg_raw, _RTMDET_DEFAULTS, 'rtmdet')
        logger.info("创建 RTMDetInferenceEngine（model_type='rtmdet'）")
        return RTMDetInferenceEngine(rtmdet_cfg)

    else:  # model_type == 'yolo'
        from model.track.yoloTrack import YOLOTrack

        logger.info("创建 YOLOTrack（model_type='yolo'）")
        return YOLOTrack(cfg)


def create_matcher(cfg: dict):
    """
    根据 cfg['model_type'] 创建并返回对应的匹配器实例。

    - model_type == 'rtmdet' → 提取 cfg['topology_matcher'] 子字典，
                               填充缺失默认值后返回 TopologyMatcher(matcher_cfg)
    - model_type == 'yolo'   → 返回 None（使用现有的 MatchingCounting）
    - 其他值                 → 记录警告并回退到 None

    Args:
        cfg: 顶层配置字典

    Returns:
        TopologyMatcher 实例，或 None（yolo 模式下使用现有 MatchingCounting）
    """
    model_type = cfg.get('model_type', 'yolo')

    if model_type not in ('yolo', 'rtmdet'):
        logger.warning(
            "未知的 model_type '%s'，create_matcher 回退到 None（使用 MatchingCounting）。",
            model_type,
        )
        return None

    if model_type == 'rtmdet':
        from model.match.topology_matcher import TopologyMatcher

        matcher_cfg_raw = cfg.get('topology_matcher', {})
        matcher_cfg = _apply_defaults(
            matcher_cfg_raw, _TOPOLOGY_MATCHER_DEFAULTS, 'topology_matcher'
        )

        # 将顶层 picture_recognition_path 传递给 matcher（若 matcher_cfg 中未指定）
        if 'picture_recognition_path' not in matcher_cfg and 'picture_recognition_path' in cfg:
            matcher_cfg['picture_recognition_path'] = cfg['picture_recognition_path']

        logger.info("创建 TopologyMatcher（model_type='rtmdet'）")
        return TopologyMatcher(matcher_cfg)

    else:  # model_type == 'yolo'
        logger.info("model_type='yolo'，create_matcher 返回 None（使用 MatchingCounting）")
        return None
