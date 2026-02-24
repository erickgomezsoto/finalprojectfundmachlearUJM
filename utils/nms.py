"""
utils/nms.py

Provided to students. Do not modify.

Contains:
    - nms_single: NMS for one class in one image
    - build_detection_outputs: takes the predictions dataframe and ground truth,
      applies NMS per class per image, and returns (preds, targets) ready to
      pass directly to torchmetrics MeanAveragePrecision.
"""

import torch
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------- #
# IoU between one box and an array of boxes                                   #
# --------------------------------------------------------------------------- #
def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box and an array of boxes.

    Args:
        box   : shape (4,)  — [x1, y1, x2, y2]
        boxes : shape (N,4) — [x1, y1, x2, y2]

    Returns:
        iou : shape (N,)
    """
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union      = area_box + area_boxes - inter

    return np.where(union > 0, inter / union, 0.0)


# --------------------------------------------------------------------------- #
# NMS for a single class within a single image                                #
# --------------------------------------------------------------------------- #
def nms_single(boxes: np.ndarray, scores: np.ndarray,
               iou_threshold: float = 0.5) -> np.ndarray:
    """
    Non-Maximum Suppression for one class in one image.

    Args:
        boxes         : shape (N, 4) — [x1, y1, x2, y2]
        scores        : shape (N,)   — confidence score for each box
        iou_threshold : boxes with IoU > this value are suppressed

    Returns:
        keep : indices of boxes to keep, sorted by descending score
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    order = np.argsort(scores)[::-1]   # sort by descending score
    keep  = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        ious      = _iou(boxes[i], boxes[order[1:]])
        remaining = np.where(ious <= iou_threshold)[0]
        order     = order[remaining + 1]   # +1 because we sliced from order[1:]

    return np.array(keep, dtype=int)


# --------------------------------------------------------------------------- #
# Build preds / targets for torchmetrics                                      #
# --------------------------------------------------------------------------- #
def build_detection_outputs(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    iou_threshold: float = 0.5,
    background_label: int = 0,
):
    """
    Apply NMS per class per image and return (preds, targets) ready for:

        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_thresholds=[0.5])
        metric.update(preds, targets)

    Args:
        predictions_df  : DataFrame with columns
                          [image_id, x1, y1, x2, y2, predicted_label, confidence]
        ground_truth_df : DataFrame with columns
                          [image_id, x1, y1, x2, y2, class_label]
                          (class_label must use the same integer encoding as
                           predicted_label, background excluded)
        iou_threshold   : IoU threshold for NMS (default 0.5)
        background_label: integer label for background (default 0, excluded
                          from both preds and targets)

    Returns:
        preds   : list of dicts, one per image
        targets : list of dicts, one per image
    """
    all_image_ids = sorted(
        set(predictions_df["image_id"].unique()) |
        set(ground_truth_df["image_id"].unique())
    )

    preds   = []
    targets = []

    for image_id in all_image_ids:

        # ---- ground truth for this image ---------------------------------- #
        gt = ground_truth_df[
            (ground_truth_df["image_id"] == image_id) &
            (ground_truth_df["class_label"] != background_label)
        ]
        targets.append({
            "boxes":  torch.tensor(
                          gt[["x1", "y1", "x2", "y2"]].values,
                          dtype=torch.float32),
            "labels": torch.tensor(
                          gt["class_label"].values,
                          dtype=torch.long),
        })

        # ---- predictions for this image ----------------------------------- #
        img_preds = predictions_df[
            (predictions_df["image_id"] == image_id) &
            (predictions_df["predicted_label"] != background_label)
        ]

        if img_preds.empty:
            preds.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,),   dtype=torch.long),
                "scores": torch.zeros((0,),   dtype=torch.float32),
            })
            continue

        # Apply NMS per foreground class
        kept_indices = []
        for cls_label in img_preds["predicted_label"].unique():
            cls_mask = img_preds["predicted_label"] == cls_label
            cls_rows = img_preds[cls_mask]

            boxes  = cls_rows[["x1", "y1", "x2", "y2"]].values
            scores = cls_rows["confidence"].values
            keep   = nms_single(boxes, scores, iou_threshold)

            kept_indices.extend(cls_rows.iloc[keep].index.tolist())

        kept = img_preds.loc[kept_indices]
        preds.append({
            "boxes":  torch.tensor(
                          kept[["x1", "y1", "x2", "y2"]].values,
                          dtype=torch.float32),
            "labels": torch.tensor(
                          kept["predicted_label"].values,
                          dtype=torch.long),
            "scores": torch.tensor(
                          kept["confidence"].values,
                          dtype=torch.float32),
        })

    return preds, targets

