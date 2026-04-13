from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw" / "utkface"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"

def main() -> None:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")
    
    print(f"USing raw data directory: {RAW_DATA_DIR}")
    print(f"Metadata will be written to: {OUTPUT_CSV}")

    image_paths = sorted(RAW_DATA_DIR.rglob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found in raw data directory: {RAW_DATA_DIR}")
    
    print(f"Found {len(image_paths)} image files.")
    print("Sample files: ")
    for path in image_paths[:5]:
        print(path)
    
    sample_rows = []
    skipped_count = 0

    for path in image_paths:
        age = parse_age_from_filename(path)

        if age is None:
            skipped_count += 1
            continue
        
        relative_path = path.relative_to(REPO_ROOT)

        sample_rows.append({
            "path": str(relative_path),
            "age": age
        })
    
    print(f"Valid rows found: {len(sample_rows)}")
    print(f"Skipped rows: {skipped_count}")
    print("Sample metadata rows: ")
    for row in sample_rows[:5]:
        print(row)

if __name__ == "__main__":
    main()

def parse_age_from_filename(path: Path) -> int | None:
    parts = path.stem.split("_")

    if len(parts) < 4:
        return None
    
    try:
        return int(parts[0])
    except ValueError:
        return None