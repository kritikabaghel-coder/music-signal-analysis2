"""
Example usage of the dataset loader and signal utilities.
"""

from pathlib import Path
import pandas as pd

from dataset_loader import GtganDatasetLoader
from signal_utils import SignalValidator, DatasetStats
from config import GENRES_DIR


def example_basic_loading():
    """Basic example: Load and validate dataset."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Dataset Loading")
    print("="*70)

    loader = GtganDatasetLoader(genres_dir=GENRES_DIR)
    df = loader.load_dataset()
    loader.validate_dataset(df)

    return df


def example_signal_validation(df: pd.DataFrame):
    """Example: Validate individual signals."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Signal Validation")
    print("="*70)

    validator = SignalValidator()

    for idx, row in df.head(5).iterrows():
        file_path = Path(row['file_path'])
        print(f"\nFile: {file_path.name}")

        try:
            signal, sr = validator.load_signal(file_path)
            is_valid, msg = validator.validate_signal_quality(signal, sr)
            print(f"  Valid: {is_valid} ({msg})")
            print(f"  Duration: {row['duration']:.2f}s")
            print(f"  Sample Rate: {sr} Hz")

        except Exception as e:
            print(f"  Error: {e}")


def example_genre_statistics(df: pd.DataFrame):
    """Example: Analyze statistics by genre."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Genre Statistics")
    print("="*70)

    for genre in df['genre'].unique():
        genre_df = df[df['genre'] == genre]
        durations = genre_df['duration'].values

        stats = DatasetStats.compute_duration_stats(durations)

        print(f"\n{genre.upper()}")
        print(f"  Files: {len(genre_df)}")
        print(f"  Duration Mean: {stats['mean']:.2f}s")
        print(f"  Duration Range: {stats['min']:.2f}s - {stats['max']:.2f}s")
        print(f"  Total Duration: {stats['total']:.2f}s ({stats['total']/60:.2f}min)")


def example_sample_rate_analysis(df: pd.DataFrame):
    """Example: Analyze sample rate distribution."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Sample Rate Distribution")
    print("="*70)

    sr_counts = df['sample_rate'].value_counts()
    for sr, count in sr_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sr} Hz: {count} files ({percentage:.1f}%)")


def example_export_metadata(df: pd.DataFrame):
    """Example: Export dataset metadata."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Export Metadata")
    print("="*70)

    output_path = GENRES_DIR.parent / "dataset_metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    # Show first few rows
    print("\nDataset Preview:")
    print(df.head().to_string(index=False))


def main():
    print("\nMUSIC GENRE CLASSIFICATION - DATASET LOADING EXAMPLES")

    # Load dataset
    df = example_basic_loading()

    if len(df) > 0:
        # Run additional examples
        example_signal_validation(df)
        example_genre_statistics(df)
        example_sample_rate_analysis(df)
        example_export_metadata(df)
        print("\n✓ All examples completed successfully!\n")
    else:
        print("\n✗ Dataset is empty. Please download and extract the GTZAN dataset.")
        print("  Run: python setup_dataset.py\n")


if __name__ == "__main__":
    main()
