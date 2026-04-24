import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import random
from pathlib import Path
from PIL import Image
import numpy

# using the "set_seed" function from our reproducibility utility
from src.utils.reproducibility import set_seed

# configuration
RAW_DATA_DIR = Path(r"data/raw/utkface")  # <-- UPDATE THIS PATH (I used D:\utkface)
OUTPUT_DIR = Path("data/processed")
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1

set_seed(42)

# parse filename
def parse_filename(filename):
    try:
        parts = filename.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        return age, gender, race
    except Exception:
        return None

# validation of image files
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

# main function to preprocess data
def main():
    print("RAW_DATA_DIR exists:", RAW_DATA_DIR.exists())
    print("RAW_DATA_DIR path:", RAW_DATA_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_records = []
    skipped = []

    # walk through raw data
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        dirs.sort()
        print(f"{root} -> {len(files)} files")

        for file in sorted(files):
            # skip anything that doesn't resemble UTKFace naming
            if "_" not in file:
                continue

            path = os.path.join(root, file)

            # basic structure check
            parts = file.split("_")
            if len(parts) < 4:
                skipped.append((file, "invalid_filename"))
                continue

            # basic filename parsing
            parsed = parse_filename(file)
            if parsed is None:
                skipped.append((file, "invalid_filename"))
                continue

            age, gender, race = parsed

            # basic image validation
            if not is_valid_image(path):
                skipped.append((file, "corrupt_image"))
                continue

            # Step 4: add valid record
            valid_records.append({
                "path": str(Path(path).resolve()),
                "age": age,
                "gender": gender,
                "race": race,
            })

    # shuffle and split
    random.shuffle(valid_records)

    orig_records = valid_records

    total = len(valid_records)
    train_end = int(total * TRAIN_SPLIT)
    valid_end = train_end + int(total * VALID_SPLIT)

    train_records = orig_records[:train_end]
    valid_records = orig_records[train_end:valid_end]
    test_records = orig_records[valid_end:]

    # save to CSV
    def save_csv(records, filename):
        with open(OUTPUT_DIR / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "age", "gender", "race"])
            writer.writeheader()
            writer.writerows(records)

    save_csv(train_records, "train.csv")
    save_csv(valid_records, "val.csv")
    save_csv(test_records, "test.csv")

    # save skipped files log
    with open(OUTPUT_DIR / "skipped_files.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "reason"])
        writer.writerows(skipped)

    # print summary (Definition of Done)
    print(f"Skipped samples: {len(skipped)}")
    print(f"Train: {len(train_records)} | Valid: {len(valid_records)} | Test: {len(test_records)}")

    # Optional debug
    if skipped:
        print("Sample skipped:", skipped[:5])

    # age statistics by dataset split
    def age_stats(records, name):
        ages = numpy.array([r["age"] for r in records])
        mean = numpy.mean(ages)
        median = numpy.median(ages)
        std = numpy.std(ages)
        print(f"\n[{name}] age statistics (n={len(ages)})")
        print(f"  min={ages.min()}  max={ages.max()}  mean={mean:.2f}  median={median}  std={std:.2f}")

    age_stats(train_records, "train")
    age_stats(valid_records, "val")
    age_stats(test_records, "test")


if __name__ == "__main__":
    main()