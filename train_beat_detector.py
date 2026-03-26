"""
Step 4: Train Beat Detector - Main Execution Script

Processes GTZAN dataset for beat detection and BPM estimation.
"""

import pandas as pd
from pathlib import Path

from beat_pipeline import BeatExtractionPipeline
from config import GENRES_DIR


def main():
    """Execute beat detection pipeline."""
    print("\n" + "="*80)
    print("STEP 4: BEAT DETECTION AND BPM ESTIMATION")
    print("="*80)

    # Load dataset metadata
    metadata_path = GENRES_DIR.parent / "dataset_metadata.csv"

    if not metadata_path.exists():
        print(f"\n✗ Metadata file not found: {metadata_path}")
        print("  Please run Step 1 (setup_dataset.py) first.")
        return

    print(f"\nLoading dataset metadata from {metadata_path}...")
    df_metadata = pd.read_csv(metadata_path)
    print(f"✓ Loaded {len(df_metadata)} audio files")

    # Initialize pipeline
    pipeline = BeatExtractionPipeline(sr=22050)

    # Process all files
    print("\nProcessing audio files for beat detection...")
    print("(This may take 2-5 minutes depending on hardware)")
    print()

    output_path = GENRES_DIR.parent / "beats_detected.csv"

    df_results = pipeline.process_dataset(
        df_metadata,
        output_path=output_path,
        verbose=True
    )

    # Compute and print statistics
    pipeline.compute_statistics()
    pipeline.print_results(n_samples=10)

    # Genre comparison
    pipeline.compare_genres_by_tempo()

    # Highlight some interesting findings
    print("\n" + "="*80)
    print("INTERESTING FINDINGS")
    print("="*80)

    print("\nTop 5 Fastest Tracks (Highest BPM):")
    print("-"*80)
    fastest = pipeline.get_high_tempo_tracks(threshold=0, top_n=5)
    for idx, row in fastest.iterrows():
        print(f"  {row['genre']:12s} | {row['tempo_bpm']:6.1f} BPM | "
              f"{Path(row['file_path']).name}")

    print("\nTop 5 Slowest Tracks (Lowest BPM):")
    print("-"*80)
    slowest = pipeline.get_low_tempo_tracks(threshold=0, top_n=5)
    for idx, row in slowest.iterrows():
        print(f"  {row['genre']:12s} | {row['tempo_bpm']:6.1f} BPM | "
              f"{Path(row['file_path']).name}")

    print("\n" + "="*80)
    print(f"✓ Beat detection complete!")
    print(f"✓ Results saved to {output_path}")
    print("="*80 + "\n")

    return df_results


if __name__ == "__main__":
    results = main()
