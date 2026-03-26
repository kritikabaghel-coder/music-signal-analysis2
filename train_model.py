"""
Step 3: Train Genre Classification Model

Trains Random Forest classifier on GTZAN feature dataset.
"""

import pandas as pd
from pathlib import Path

from model_pipeline import ClassificationPipeline
from config import GENRES_DIR


def main():
    """Train genre classification model."""
    print("\n" + "="*80)
    print("STEP 3: GENRE CLASSIFICATION MODEL TRAINING")
    print("="*80)

    # Load features
    features_path = GENRES_DIR.parent / "features_extracted.csv"

    if not features_path.exists():
        print(f"\n✗ Feature file not found: {features_path}")
        print("  Please run Step 2 (feature extraction) first.")
        print("  Execute: python extract_features.py")
        return

    print(f"\nLoading features from {features_path}...")
    df_features = pd.read_csv(features_path)
    print(f"✓ Loaded {len(df_features)} audio files with {df_features.shape[1]} features")

    # Run pipeline
    pipeline = ClassificationPipeline(random_state=42)

    results = pipeline.run_full_pipeline(
        df_features,
        test_size=0.2,
        n_estimators=100,
        max_depth=None,
        save_model=True,
        model_path=GENRES_DIR.parent / "trained_model.pkl"
    )

    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy:     {results['test_accuracy']:.4f}")
    print("="*80 + "\n")

    return results


if __name__ == "__main__":
    results = main()
