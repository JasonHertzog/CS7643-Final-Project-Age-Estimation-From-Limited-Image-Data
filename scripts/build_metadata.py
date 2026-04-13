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

if __name__ == "__main__":
    main()