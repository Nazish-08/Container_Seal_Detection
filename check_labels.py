from pathlib import Path
from collections import Counter

base = Path("Combined_Dataset/labels")
counts = Counter()

for split in ["train", "val", "test"]:
    folder = base / split

    for file in folder.glob("*.txt"):
        for line in file.read_text().splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1

print("Class ID counts:", counts)