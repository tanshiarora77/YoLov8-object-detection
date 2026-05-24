from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os

# ── Configuration ──────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"        # nano model (downloads automatically)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

def load_model(model_path=MODEL_PATH):
    """Load YOLOv8 model"""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    return model

def detect_image(model, image_path, save=True):
    """Run object detection on a single image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    results = model.predict(
        source=img,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    annotated = results[0].plot()

    # Print detection results
    boxes = results[0].boxes
    print(f"\nDetected {len(boxes)} object(s) in {os.path.basename(image_path)}:")
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"  - {cls_name}: {conf:.2f} confidence")

    if save:
        output_path = "output_" + os.path.basename(image_path)
        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated image to: {output_path}")

    cv2.imshow("YOLOv8 Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return results

def detect_video(model, video_path=None):
    """Run object detection on video or webcam"""
    source = video_path if video_path else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Running detection. Press 'q' to quit.")
    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        annotated = results[0].plot()
        num_detections = len(results[0].boxes)
        total_detections += num_detections
        frame_count += 1

        # Display stats
        cv2.putText(annotated, f"Objects: {num_detections}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("YOLOv8 Real-Time Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames | Total detections: {total_detections}")

def evaluate_model(model, dataset_yaml):
    """Evaluate model performance — prints Precision, Recall, mAP"""
    print("Evaluating model on dataset...")
    metrics = model.val(data=dataset_yaml, verbose=True)
    print(f"\n── Evaluation Metrics ──")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@0.5   : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:95: {metrics.box.map:.4f}")
    return metrics

def train_custom_model(data_yaml, epochs=50, img_size=640):
    """Train YOLOv8 on a custom dataset"""
    model = YOLO("yolov8n.pt")
    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        name="custom_yolov8",
        patience=10,
        save=True,
        plots=True
    )
    print("Training complete. Best weights saved to runs/detect/custom_yolov8/weights/best.pt")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Pipeline")
    parser.add_argument("--mode", choices=["image", "video", "webcam", "eval", "train"],
                        default="webcam", help="Detection mode")
    parser.add_argument("--source", type=str, help="Path to image or video file")
    parser.add_argument("--data", type=str, help="Path to dataset YAML (for eval/train)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    model = load_model()

    if args.mode == "image":
        detect_image(model, args.source)
    elif args.mode == "video":
        detect_video(model, args.source)
    elif args.mode == "webcam":
        detect_video(model)
    elif args.mode == "eval":
        evaluate_model(model, args.data)
    elif args.mode == "train":
        train_custom_model(args.data, epochs=args.epochs)
