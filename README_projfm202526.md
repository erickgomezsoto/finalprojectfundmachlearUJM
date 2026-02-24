# Final Project — Fundamentals of Machine Learning 2025/2026
## Object Detection with Classical Classifiers

This repository contains the starter code for the final project of the
Fundamentals of Machine Learning course at UJM. You will implement the
classification and evaluation components of a Region-based Convolutional
Neural Network (R-CNN) pipeline using Support Vector Machines and Decision
Trees.

Read the project description PDF carefully before starting.

---

## Getting the data

The large data files are not stored in this repository. Download them from
**Moodle → Final Project → Data** and place them as described below.

You need to download two things:

**1. images.zip**
Unzip it and place all the images inside `coco_filtered/images/`:
```
coco_filtered/images/000000000009.jpg
coco_filtered/images/000000000025.jpg
...
```

**2. features_train.npz and features_val.npz**
Place both files directly inside `coco_filtered/`:
```
coco_filtered/features_train.npz
coco_filtered/features_val.npz
```

You do not need to run `extract_features.py` unless you are doing Task 2a
(region-level augmentation), which requires rerunning the feature extraction
with augmented crops.

---

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If `pycocotools` fails to install due to a compilation error, use this
alternative instead:

```bash
pip install faster-coco-eval
```

---

## Project structure

```
project/
├── coco_filtered/
│   ├── images/              ← download from Moodle and unzip here
│   ├── regions.csv          ← ground truth annotations
│   ├── selected_images.csv  ← list of selected images with train/val split
│   ├── features_train.npz   ← download from Moodle
│   └── features_val.npz     ← download from Moodle
├── utils/
│   ├── __init__.py
│   ├── nms.py               ← provided, do not modify
│   └── eval_detection.py    ← provided, do not modify
├── outputs/
│   ├── task1/               ← results saved here by task1.py
│   ├── task2/               ← results saved here by task2.py
│   ├── task3/               ← results saved here by task3.py
│   └── task4/               ← results saved here by task4.py
├── extract_features.py      ← provided, run once only if doing Task 2a
├── task1.py                 ← Task 1: baseline SVM and Decision Tree
├── task2.py                 ← Task 2: class imbalance handling
├── task3.py                 ← Task 3: SVM kernel comparison
├── task4.py                 ← Task 4: Decision Tree depth analysis
├── compare.py               ← compare results across all tasks
├── requirements.txt
└── README.md
```

---

## Running the tasks

Run each task script from the root of the project folder:

```bash
python task1.py --data_dir ./coco_filtered
python task2.py --data_dir ./coco_filtered
python task3.py --data_dir ./coco_filtered
python task4.py --data_dir ./coco_filtered
python compare.py
```

Each script saves its results automatically to the corresponding subfolder
in `outputs/`. Run `compare.py` after all tasks are complete to generate
the final comparison across models.

---

## Submitting your work

Submit a `.zip` archive containing your code and a PDF report of maximum
5 pages (excluding appendices). See the project description for the full
list of deliverables and grading criteria.
