
from pathlib import Path
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Test images folder
test_folder = Path("Combined_Dataset/images/test")

# Check folder
if not test_folder.exists():
    raise FileNotFoundError(f"Test folder not found: {test_folder.resolve()}")

images = [
    p for p in test_folder.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
]

print("Total test images:", len(images))

if not images:
    raise RuntimeError("Test folder mein koi image nahi mili.")

# Run detection

results = model.predict(
    source=str(test_folder),
    imgsz=640,
    conf=0.25,
    save=True,
    project=str(Path(__file__).resolve().parent / "runs" / "detect"),
    name="predict",
    exist_ok=True
)


print("\nDetection complete!")
print("Annotated images saved in the runs/detect/ folder.")

for result in results[:5]:
    print(
        result.path,
        "Detections:", len(result.boxes)
    )
