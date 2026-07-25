# Face Mask Detection using YOLOv8

## Overview
This project implements an end-to-end Face Mask Detection system using the YOLOv8 object detection framework. The model is trained on a publicly available Roboflow dataset to detect and classify people wearing face masks correctly, not wearing masks, or wearing masks incorrectly.

The project covers the complete object detection pipeline, including dataset preparation, model training, evaluation, and inference.

---

## Features

- Object detection using YOLOv8
- Three-class face mask detection
- Training, validation, and testing pipeline
- Performance evaluation using Precision, Recall, and mAP
- Prediction on unseen images
- Saved model weights for inference

---

## Dataset

**Source:** Roboflow Public Dataset

### Classes
- With Mask
- Without Mask
- Mask Worn Incorrectly

### Dataset Structure

```
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Matplotlib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Face-Mask-Detection-YOLOv8.git
cd Face-Mask-Detection-YOLOv8
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run the following command:

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=20 imgsz=640 batch=16
```

---

## Model Configuration

| Parameter | Value |
|----------|-------|
| Model | YOLOv8n |
| Epochs | 20 |
| Batch Size | 16 |
| Image Size | 640 × 640 |
| Device | CPU |

---

## Evaluation Metrics

| Metric | Value |
|---------|-------|
| Precision | 74.6% |
| Recall | 68.4% |
| mAP@0.5 | 71.3% |
| mAP@0.5:0.95 | 49.0% |

---

## Inference

Run prediction using:

```bash
yolo detect predict model=best.pt source=test/images save=True
```

Predicted images will be saved in:

```
runs/detect/predict/
```

---

## Project Structure

```
Face-Mask-Detection-YOLOv8
│
├── dataset/
├── runs/
├── weights/
├── data.yaml
├── train.py
├── detect.py
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Results

The YOLOv8 model successfully detects and localizes face masks in images with satisfactory accuracy.

### Overall Performance

- Precision: **74.6%**
- Recall: **68.4%**
- mAP@0.5: **71.3%**
- mAP@0.5:0.95: **49.0%**

Sample prediction outputs are available in the `runs/detect/predict` directory.

---

## Challenges Faced

- CPU-only training resulted in longer training time.
- Limited system RAM affected training speed.
- Dataset imbalance impacted performance on some classes.

---

## Future Improvements

- Train on GPU for faster convergence.
- Increase dataset size.
- Apply advanced data augmentation.
- Experiment with larger YOLOv8 models.
- Deploy the model as a real-time webcam application.

---

## Author

*Tanshi* 

B.Tech Artificial Intelligence & Data Science
