import os
import shutil
import urllib.request
import tarfile
from pathlib import Path
from config import DATA_DIR, GENRES_DIR

GTZAN_URL = "http://marsyasweb.appspot.com/download/genres.tar.gz"
DOWNLOAD_PATH = DATA_DIR / "genres.tar.gz"


def download_gtzan_dataset():
    """Download GTZAN dataset from the official source."""
    print("Downloading GTZAN dataset...")
    print(f"URL: {GTZAN_URL}")

    try:
        urllib.request.urlretrieve(GTZAN_URL, DOWNLOAD_PATH)
        print(f"✓ Downloaded to {DOWNLOAD_PATH}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_gtzan_dataset():
    """Extract GTZAN dataset and organize by genre."""
    print(f"\nExtracting GTZAN dataset...")

    try:
        with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
            tar.extractall(DATA_DIR)
        print("✓ Extraction complete")

        # Reorganize if needed
        extract_dir = DATA_DIR / "genres"
        if extract_dir.exists():
            print("✓ Dataset already in correct structure")
            return True

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def verify_dataset_structure():
    """Verify that genre folders exist and contain files."""
    print(f"\nVerifying dataset structure...")

    total_files = 0
    for genre_dir in GENRES_DIR.iterdir():
        if genre_dir.is_dir():
            files = list(genre_dir.glob("*.wav"))
            total_files += len(files)
            status = "✓" if len(files) > 0 else "✗"
            print(f"  {status} {genre_dir.name}: {len(files)} files")

    print(f"\nTotal audio files: {total_files}")
    return total_files > 0


def main():
    print("="*70)
    print("GTZAN DATASET SETUP")
    print("="*70)

    if not download_gtzan_dataset():
        print("\nManual Setup:")
        print("1. Download from: http://marsyasweb.appspot.com/download/genres.tar.gz")
        print(f"2. Place genres.tar.gz in: {DOWNLOAD_PATH}")
        print("3. Run this script again")
        return

    if not extract_gtzan_dataset():
        print("Extraction failed. Please check the downloaded file.")
        return

    if verify_dataset_structure():
        print("\n✓ GTZAN dataset setup complete!")
    else:
        print("\n✗ Dataset verification failed")


if __name__ == "__main__":
    main()
