# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from types import SimpleNamespace
import yaml


def on_predict_start(predictor, persist=False, tracking_config="ocsort.yaml"):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """
    if hasattr(predictor, 'trackers') and persist:
        return
    tracking_config = tracking_config
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            tracking_config,
            True
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


def create_tracker(tracker_config, per_class):
    # Windows 默认编码可能为 GBK，而配置文件通常为 UTF-8；这里做自适应解码，避免报错
    with open(tracker_config, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gbk")
        except UnicodeDecodeError:
            # 最后兜底：替换不可解码字符（一般仅出现在注释里，不影响YAML键值解析）
            text = raw.decode("utf-8", errors="replace")

    cfg = yaml.load(text, Loader=yaml.FullLoader)
    cfg = SimpleNamespace(**cfg)  # easier dict acces by dot, instead of ['']

    from ..track.ocsort import OCSort
    ocsort = OCSort(
        per_class,
        det_thresh=cfg.det_thresh,
        max_age=cfg.max_age,
        min_hits=cfg.min_hits,
        iou_threshold=cfg.iou_thresh,
        delta_t=cfg.delta_t,
        asso_func=cfg.asso_func,
        inertia=cfg.inertia,
        use_byte=cfg.use_byte,
        low_thresh=getattr(cfg, "low_thresh", 0.1),
        edge_filter_ratio=getattr(cfg, "edge_filter_ratio", 0.05),
    )
    return ocsort
