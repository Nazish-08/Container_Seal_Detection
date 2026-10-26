# Container Seal Detection

YOLO-based object detection project for locating container seals in images.

## Project Overview

- Task: Object detection (bounding boxes around container seals)
- Framework: Ultralytics YOLO
- Base model: yolov8n.pt
- Trained weights: best.pt
- Input image size: 640 × 640
- Dataset: Combined Dataset 1 and Dataset 2

## Project Structure

```text
container seal detection.v3i.coco/
├── Combined_Dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
├── best.pt
├── train_model.py
├── test_model.py
├── check_labels.py
├── venv/
├── runs/
│   └── detect/
│       └── predict/
└── README.md
```

## Dataset

- Training images: 1,470
- Validation images: 124
- Test images: 70
- Ground-truth test instances: 102

Dataset configuration: Combined_Dataset/data.yaml

Verify that class names and class IDs match the YOLO label files before retraining.

## Model Evaluation Results

These results came from the Google Colab test-split evaluation of best.pt.

| Metric | Result |
|---|---:|
| Precision | 82.8% |
| Recall | 76.5% |
| mAP@50 | 79.7% |
| mAP@50–95 | 28.9% |
| Test images | 70 |
| Ground-truth instances | 102 |

### What the Metrics Mean

- Precision (82.8%): Proportion of predicted detections that were correct under the evaluation matching rules.
- Recall (76.5%): Proportion of ground-truth objects detected by the model.
- mAP@50 (79.7%): Mean Average Precision at an IoU threshold of 0.50.
- mAP@50–95 (28.9%): Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

These are object-detection metrics, not one general accuracy percentage. The lower mAP@50–95 suggests bounding-box localization is less consistent at stricter overlap thresholds.

The evaluation output listed the seal class with 70 images and 102 instances. Verify class mapping if more than one class is intended.

## Requirements

Install Ultralytics in your Python virtual environment:

```bash
pip install ultralytics
```

## Run Detection on Test Images

Ensure best.pt and Combined_Dataset/ are in the project root. Activate your virtual environment and run:

```bash
python test_model.py
```

The script runs predictions on Combined_Dataset/images/test and should save annotated images under:

```text
runs/detect/predict/
```

If the output goes elsewhere, check the project and name arguments in test_model.py.

## Train the Model

The local training log shows training was interrupted with KeyboardInterrupt during epoch 2 of 50. The evaluation metrics above came from the Colab test evaluation, not that interrupted local run.

After checking train_model.py and Combined_Dataset/data.yaml, run:

```bash
python train_model.py
```

Training can take a long time on CPU. The local log showed GPU_mem 0G, indicating that run was using CPU rather than a CUDA GPU.

## Important Files

- best.pt — trained model weights used for inference.
- Combined_Dataset/data.yaml — dataset paths and class names.
- train_model.py — training script.
- test_model.py — prediction/testing script.
- check_labels.py — label-checking utility.
- runs/detect/predict/ — annotated prediction images.

## Limitations and Recommendations

1. Inspect annotated test images to confirm boxes are positioned correctly.
2. Review images where the model reports no detections.
3. Verify class IDs in label files match data.yaml.
4. Keep a backup of best.pt and Combined_Dataset/ before deleting original datasets.
5. Evaluate on additional real-world images before using the model in production.

---
README based on the project logs and evaluation values shared during the conversation.