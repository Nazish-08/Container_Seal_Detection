from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "Combined_Dataset" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=8,
    project=str(BASE_DIR / "runs"),
    name="container_seal"
)

print("Training complete!")
print("Best model:")
print(BASE_DIR / "runs" / "container_seal" / "weights" / "best.pt")
