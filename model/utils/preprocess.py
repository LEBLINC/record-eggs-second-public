import cv2
import numpy as np
from typing import Any, Dict, Optional


def _apply_gamma_u8(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma 校正（uint8），gamma>1 通常提亮暗场；gamma<1 通常压暗。"""
    if gamma is None:
        return bgr
    try:
        gamma = float(gamma)
    except Exception:
        return bgr
    if gamma <= 0 or abs(gamma - 1.0) < 1e-6:
        return bgr

    inv = 1.0 / gamma
    table = (np.arange(256, dtype=np.float32) / 255.0) ** inv * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, table)


def _apply_clahe_l(bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """在 LAB 的 L 通道做 CLAHE（轻量、对光照变化更稳）。"""
    tile = int(tile) if tile else 8
    tile = max(2, tile)
    clip_limit = float(clip_limit) if clip_limit else 2.0
    clip_limit = max(0.1, clip_limit)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _gray_world_awb(bgr: np.ndarray) -> np.ndarray:
    """简单 Gray-World 白平衡（可选，默认关闭）。"""
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = float(b.mean()), float(g.mean()), float(r.mean())
    m = (mb + mg + mr) / 3.0 + 1e-6
    b *= (m / (mb + 1e-6))
    g *= (m / (mg + 1e-6))
    r *= (m / (mr + 1e-6))
    out = cv2.merge((b, g, r))
    return np.clip(out, 0, 255).astype(np.uint8)


def _unsharp_mask(bgr: np.ndarray, sigma: float = 1.0, amount: float = 0.6) -> np.ndarray:
    """轻锐化（可选）。"""
    sigma = max(0.1, float(sigma))
    amount = float(amount)
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def preprocess_bgr(frame: np.ndarray, cfg: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    可选的轻量预处理（保持尺寸不变，便于直接复用坐标）。

    cfg 示例（configs/config.yaml）：
      preprocess:
        enabled: false
        gamma: 1.0
        clahe:
          enabled: false
          clipLimit: 2.0
          tileGridSize: 8
        awb: false
        sharpen: false
    """
    if frame is None:
        return frame

    # 兼容红外/灰度输入（部分红外摄像头/历史素材可能是单通道）：
    # - 模型与大多数 OpenCV 颜色变换期望 BGR 3通道
    # - 这里仅做形态兼容，不改变几何尺寸；对 RGB 摄像头输入无影响
    try:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            # 兼容 BGRA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    except Exception:
        # 转换失败则继续使用原始输入（后续也有异常兜底）
        pass

    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return frame

    try:
        out = frame

        # CLAHE（建议优先尝试）
        clahe_cfg = cfg.get("clahe", {}) if isinstance(cfg.get("clahe", {}), dict) else {}
        if bool(clahe_cfg.get("enabled", False)):
            out = _apply_clahe_l(
                out,
                clip_limit=float(clahe_cfg.get("clipLimit", 2.0)),
                tile=int(clahe_cfg.get("tileGridSize", 8)),
            )

        # Gamma
        out = _apply_gamma_u8(out, cfg.get("gamma", 1.0))

        # AWB
        if bool(cfg.get("awb", False)):
            out = _gray_world_awb(out)

        # Sharpen
        if bool(cfg.get("sharpen", False)):
            out = _unsharp_mask(out, sigma=float(cfg.get("sharpen_sigma", 1.0)), amount=float(cfg.get("sharpen_amount", 0.6)))

        return out
    except Exception:
        # 预处理失败时，不影响主流程
        return frame


