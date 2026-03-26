import logging
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd

from config import GENRES_DIR, GENRES, AUDIO_EXTENSIONS, DEFAULT_SR, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "dataset_loading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GtganDatasetLoader:
    def __init__(self, genres_dir: Path = GENRES_DIR, sr: int = DEFAULT_SR):
        self.genres_dir = Path(genres_dir)
        self.sr = sr
        self.dataset: List[Dict] = []
        self.errors: List[Dict] = []

    def _get_audio_files(self) -> Dict[str, List[Path]]:
        """Get all audio files grouped by genre."""
        genre_files = {genre: [] for genre in GENRES}

        for genre in GENRES:
            genre_path = self.genres_dir / genre
            if not genre_path.exists():
                logger.warning(f"Genre directory not found: {genre_path}")
                continue

            for file_path in genre_path.iterdir():
                if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    genre_files[genre].append(file_path)

        return genre_files

    def _load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa."""
        signal, sr = librosa.load(file_path, sr=self.sr, mono=True)
        return signal, sr

    def load_dataset(self) -> pd.DataFrame:
        """Load all audio files and create dataset."""
        logger.info("Starting dataset loading...")

        genre_files = self._get_audio_files()

        for genre, files in genre_files.items():
            logger.info(f"Processing {genre}: {len(files)} files")

            for file_path in files:
                try:
                    signal, sr = self._load_audio(file_path)
                    duration = librosa.get_duration(y=signal, sr=sr)

                    self.dataset.append({
                        "file_path": str(file_path),
                        "genre": genre,
                        "sample_rate": sr,
                        "duration": duration,
                        "num_samples": len(signal)
                    })

                except Exception as e:
                    error_info = {
                        "file_path": str(file_path),
                        "genre": genre,
                        "error": str(e)
                    }
                    self.errors.append(error_info)
                    logger.error(f"Failed to load {file_path}: {e}")

        df = pd.DataFrame(self.dataset)
        logger.info(f"Dataset loaded with {len(df)} valid files")
        logger.info(f"Encountered {len(self.errors)} errors during loading")

        return df

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """Print dataset statistics and validation info."""
        print("\n" + "="*70)
        print("DATASET VALIDATION REPORT")
        print("="*70)

        print(f"\nTotal Audio Files: {len(df)}")
        print(f"Number of Genres: {df['genre'].nunique()}")
        print(f"Failed Loads: {len(self.errors)}")

        print("\n" + "-"*70)
        print("FILES PER GENRE")
        print("-"*70)
        genre_counts = df["genre"].value_counts().sort_index()
        for genre, count in genre_counts.items():
            print(f"  {genre:12s}: {count:3d} files")

        print("\n" + "-"*70)
        print("DURATION STATISTICS (seconds)")
        print("-"*70)
        print(f"  Mean:   {df['duration'].mean():.2f}s")
        print(f"  Median: {df['duration'].median():.2f}s")
        print(f"  Min:    {df['duration'].min():.2f}s")
        print(f"  Max:    {df['duration'].max():.2f}s")
        print(f"  Std:    {df['duration'].std():.2f}s")

        print("\n" + "-"*70)
        print("SAMPLE RATE STATISTICS")
        print("-"*70)
        sr_counts = df["sample_rate"].value_counts()
        for sr, count in sr_counts.items():
            print(f"  {sr} Hz: {count} files")

        print("\n" + "-"*70)
        print("SAMPLE DATASET ROWS")
        print("-"*70)
        print(df.head(10).to_string(index=False))

        if self.errors:
            print("\n" + "-"*70)
            print("ERROR DETAILS")
            print("-"*70)
            for error in self.errors[:5]:
                print(f"  File: {error['file_path']}")
                print(f"  Error: {error['error']}")
                print()

        print("="*70 + "\n")


def main():
    loader = GtganDatasetLoader(genres_dir=GENRES_DIR)
    df = loader.load_dataset()
    loader.validate_dataset(df)

    # Save dataset metadata
    output_path = GENRES_DIR.parent / "dataset_metadata.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset metadata saved to {output_path}")

    return df


if __name__ == "__main__":
    dataset_df = main()
