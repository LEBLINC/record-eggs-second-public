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
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
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
    )
    return ocsort
