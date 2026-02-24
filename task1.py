"""
task1.py

Task 1 — Baseline Classification (SVM and Decision Tree)

Important:
    1. Read the project description carefully before starting.
    2. Complete the sections marked with T O D O.
    3. Do not modify the loading or evaluation code.

Usage:
    python task1.py --data_dir ./data

Do not forget to write a different .py file for each task
and the comparison.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from utils.eval_detection import evaluate_detection

LABEL_NAMES  = {0: "background", 1: "person", 2: "car", 3: "truck"}
CLASS_NAMES  = ["background", "person", "car", "truck"]
FOREGROUND   = [1, 2, 3]   # indices used for foreground-only macro F1


def load_features(path: Path):
    """Load features, labels, boxes, and image_ids from a .npz file."""
    data = np.load(path)
    return (
        data["features"],   # (N, 2048)  float32
        data["labels"],     # (N,)       int64
        data["boxes"],      # (N, 4)     float32  [x1, y1, x2, y2]
        data["image_ids"],  # (N,)       int64
    )


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def print_classification_report(y_true, y_pred, model_name: str):
    """Print per-class precision, recall, F1, support."""
    print(f"\n{'='*55}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*55}")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    ))
    # Foreground-only macro F1
    fg_mask = np.isin(y_true, FOREGROUND)
    fg_report = classification_report(
        y_true[fg_mask], y_pred[fg_mask],
        labels=FOREGROUND,
        target_names=["person", "car", "truck"],
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    fg_macro_f1 = fg_report["macro avg"]["f1-score"]
    print(f"  Foreground-only macro F1 : {fg_macro_f1:.3f}")
    print(f"{'='*55}\n")
    return fg_macro_f1


def build_predictions_df(image_ids, boxes, y_pred, confidence, y_true):
    """
    Build the predictions dataframe used by eval_detection.
    Columns: image_id, x1, y1, x2, y2, predicted_label, confidence, true_label
    """
    return pd.DataFrame({
        "image_id":        image_ids,
        "x1":              boxes[:, 0],
        "y1":              boxes[:, 1],
        "x2":              boxes[:, 2],
        "y2":              boxes[:, 3],
        "predicted_label": y_pred,
        "confidence":      confidence,
        "true_label":      y_true,
    })


def build_ground_truth_df(regions_csv: Path):
    """Load ground truth boxes from regions.csv (foreground only)."""
    label_map = {"person": 1, "car": 2, "truck": 3}
    df = pd.read_csv(regions_csv)
    df["class_label"] = df["class_label"].map(label_map)
    df = df.dropna(subset=["class_label"])
    df["class_label"] = df["class_label"].astype(int)
    return df[["image_id", "x1", "y1", "x2", "y2", "class_label"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path("outputs/task1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load features                                                  
    print("Loading features...")
    X_train, y_train, boxes_train, ids_train = load_features(
        data_dir / "features_train.npz"
    )
    X_val, y_val, boxes_val, ids_val = load_features(
        data_dir / "features_val.npz"
    )
    ground_truth_df = build_ground_truth_df(data_dir / "regions.csv")

    print(f"  Train: {X_train.shape[0]} regions | Val: {X_val.shape[0]} regions")

    # Print label distribution
    for split_name, y in [("train", y_train), ("val", y_val)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n  {split_name} label distribution:")
        for u, c in zip(unique, counts):
            print(f"    {LABEL_NAMES[u]:<12}: {c:>8} ({100*c/len(y):.1f}%)")

    # 2. T O D O: Train your classifiers here                              
    #
    # You need to train two models:
    #   (a) A multiclass SVM using a linear kernel
    #   (b) A Decision Tree with a fixed maximum depth
    #
    # For the SVM use:
    #   from sklearn.svm import LinearSVC
    #   svm = LinearSVC(max_iter=2000)
    #   svm.fit(X_train, y_train)
    #
    # For the Decision Tree use:
    #   from sklearn.tree import DecisionTreeClassifier
    #   tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    #   tree.fit(X_train, y_train)
    #
    # After training, compute predictions and confidence scores:
    #
    # SVM predictions:
    #   y_pred_svm   = svm.predict(X_val)
    #   scores_svm   = svm.decision_function(X_val)          # shape (N, n_classes)
    #   conf_svm     = scores_svm[range(len(X_val)), y_pred_svm]
    #
    # Decision Tree predictions:
    #   y_pred_tree  = tree.predict(X_val)
    #   probas_tree  = tree.predict_proba(X_val)              # shape (N, n_classes)
    #   conf_tree    = probas_tree[range(len(X_val)), y_pred_tree]
    #
    # Store your trained models and predictions in the variables below
    # so the evaluation code can use them.

    svm       = None   # replace with your trained SVM
    tree      = None   # replace with your trained Decision Tree
    y_pred_svm  = None # replace with svm.predict(X_val)
    conf_svm    = None # replace with confidence scores for SVM
    y_pred_tree = None # replace with tree.predict(X_val)
    conf_tree   = None # replace with confidence scores for Decision Tree

    # 3. Evaluation — do not modify below this line                    #
    assert svm  is not None, "Train your SVM before running evaluation."
    assert tree is not None, "Train your Decision Tree before running evaluation."

    for model_name, y_pred, confidence in [
        ("SVM",           y_pred_svm,  conf_svm),
        ("Decision Tree", y_pred_tree, conf_tree),
    ]:
        print(f"\n{'#'*55}")
        print(f"  Evaluating: {model_name}")
        print(f"{'#'*55}")

        # ---- Region-level F1 ---------------------------------------- #
        fg_macro_f1 = print_classification_report(y_val, y_pred, model_name)

        # ---- Confusion matrix --------------------------------------- #
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])
        plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix — {model_name}",
            output_path=output_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"
        )

        # ---- Build predictions dataframe and save ------------------- #
        preds_df = build_predictions_df(
            ids_val, boxes_val, y_pred, confidence, y_val
        )
        csv_name = f"predictions_{model_name.lower().replace(' ', '_')}.csv"
        preds_df.to_csv(output_dir / csv_name, index=False)
        print(f"  Predictions saved to outputs/task1/{csv_name}")

        # ---- Detection-level mAP ------------------------------------ #
        print(f"\n  mAP evaluation for {model_name}:")
        evaluate_detection(preds_df, ground_truth_df, verbose=True)

    print("\nTask 1 complete. Check outputs/task1/ for results.")


if __name__ == "__main__":
    main()
