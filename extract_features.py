"""
Main feature extraction pipeline script.
Orchestrates dataset loading and feature extraction.
"""

import pandas as pd
from pathlib import Path

from dataset_loader import GtganDatasetLoader
from feature_pipeline import FeatureExtractionPipeline
from config import GENRES_DIR


def main():
    """
    Execute complete feature extraction pipeline.
    """
    print("\n" + "="*80)
    print("STEP 2: FEATURE EXTRACTION PIPELINE")
    print("="*80)

    # Step 1: Load dataset
    print("\n[1/3] Loading dataset...")
    loader = GtganDatasetLoader(genres_dir=GENRES_DIR)
    df_dataset = loader.load_dataset()

    if len(df_dataset) == 0:
        print("✗ Dataset is empty. Please run setup_dataset.py first.")
        return

    print(f"✓ Loaded {len(df_dataset)} audio files\n")

    # Step 2: Extract features
    print("[2/3] Extracting features...")
    pipeline = FeatureExtractionPipeline(
        dataset_df=df_dataset,
        n_mfcc=13,
        n_fft=2048,
        normalize=True
    )

    df_features = pipeline.process_dataset()

    if len(df_features) == 0:
        print("✗ Feature extraction failed for all files.")
        return

    # Step 3: Normalize features
    print("[3/3] Normalizing features...\n")
    df_features_normalized = pipeline.normalize_features(df_features)

    # Print summary
    pipeline.print_feature_summary(df_features_normalized)

    # Save features
    output_path = GENRES_DIR.parent / "features_extracted.csv"
    pipeline.save_features(df_features_normalized, output_path)

    # Return for further processing
    return df_features_normalized


if __name__ == "__main__":
    features_df = main()
