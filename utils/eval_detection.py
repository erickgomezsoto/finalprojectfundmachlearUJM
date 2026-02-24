"""
utils/eval_detection.py

Provided to students. Do not modify.

Computes mAP@0.5 and per-class AP given a predictions dataframe and a
ground truth dataframe, using torchmetrics and the NMS utility in nms.py.

Usage:
    from utils.eval_detection import evaluate_detection

    results = evaluate_detection(predictions_df, ground_truth_df)
    print(results)
"""

import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.nms import build_detection_outputs

# Label encoding — must be consistent with how you trained your classifier
LABEL_NAMES = {
    0: "background",
    1: "person",
    2: "car",
    3: "truck",
}
FOREGROUND_LABELS = [1, 2, 3]


def evaluate_detection(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    iou_threshold: float = 0.5,
    background_label: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Compute mAP@0.5 and per-class AP.

    Args:
        predictions_df  : DataFrame with columns
                          [image_id, x1, y1, x2, y2, predicted_label, confidence]
                          One row per proposed region (before NMS).
        ground_truth_df : DataFrame with columns
                          [image_id, x1, y1, x2, y2, class_label]
                          One row per ground truth box (background excluded).
        iou_threshold   : IoU threshold for NMS and AP computation (default 0.5)
        background_label: integer label for background class (default 0)
        verbose         : if True, print a formatted summary

    Returns:
        dict with keys:
            "map"           : overall mAP@0.5 (float)
            "map_per_class" : per-class AP as a list
            "raw"           : full torchmetrics output dict
    """
    # Apply NMS and build torchmetrics input format
    preds, targets = build_detection_outputs(
        predictions_df, ground_truth_df,
        iou_threshold=iou_threshold,
        background_label=background_label,
    )

    # Compute mAP
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
    metric.update(preds, targets)
    result = metric.compute()

    map_score = float(result["map"])

    # torchmetrics can return either a tensor or a scalar float for map_per_class
    # depending on the version and whether per-class AP could be computed.
    # We convert everything to a plain Python list to be safe.
    raw = result["map_per_class"]
    if hasattr(raw, "tolist"):
        converted = raw.tolist()
    else:
        converted = float(raw)
    map_per_class = converted if isinstance(converted, list) else [converted]

    if verbose:
        print(f"\n{'='*40}")
        print(f"  Detection Evaluation @ IoU={iou_threshold}")
        print(f"{'='*40}")
        print(f"  mAP@{iou_threshold:<4}         : {map_score:.4f}")
        print(f"  Per-class AP:")
        for i, label in enumerate(FOREGROUND_LABELS):
            name = LABEL_NAMES.get(label, str(label))
            ap   = map_per_class[i] if i < len(map_per_class) else float("nan")
            print(f"    {name:<12} : {ap:.4f}")
        print(f"{'='*40}\n")

    return {
        "map":           map_score,
        "map_per_class": map_per_class,
        "raw":           result,
    }

#corrected