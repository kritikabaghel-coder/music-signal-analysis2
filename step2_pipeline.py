"""
STEP 2: FEATURE EXTRACTION - COMPREHENSIVE GUIDE

This module orchestrates the complete feature extraction pipeline with 
advanced analysis and reporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from dataset_loader import GtganDatasetLoader
from feature_pipeline import FeatureExtractionPipeline
from feature_analysis import FeatureAnalyzer, FeatureComparison
from config import GENRES_DIR


def run_complete_pipeline(
    limit_files: int = None,
    normalize: bool = True,
    analyze: bool = True
):
    """
    Run complete feature extraction and analysis pipeline.

    Parameters:
    -----------
    limit_files : int, optional
        Limit number of files to process (for testing)
    normalize : bool
        Normalize features
    analyze : bool
        Run feature analysis

    Returns:
    --------
    tuple : (df_features, df_analysis)
    """
    print("\n" + "="*80)
    print("STEP 2: FEATURE EXTRACTION PIPELINE")
    print("="*80)

    # STEP 1: Load Dataset
    print("\n[1/4] Loading dataset...")
    loader = GtganDatasetLoader(genres_dir=GENRES_DIR)
    df_dataset = loader.load_dataset()

    if len(df_dataset) == 0:
        print("✗ Dataset is empty. Run setup_dataset.py first.")
        return None, None

    if limit_files:
        df_dataset = df_dataset.head(limit_files)

    print(f"✓ Loaded {len(df_dataset)} audio files")

    # STEP 2: Extract Features
    print("\n[2/4] Extracting features...")
    pipeline = FeatureExtractionPipeline(
        dataset_df=df_dataset,
        n_mfcc=13,
        n_fft=2048,
        normalize=False  # Normalize in next step
    )

    df_features = pipeline.process_dataset()

    if len(df_features) == 0:
        print("✗ Feature extraction failed for all files.")
        return None, None

    print(f"✓ Extracted features from {len(df_features)} files")

    # STEP 3: Normalize Features
    print("\n[3/4] Normalizing features...")
    if normalize:
        df_features = pipeline.normalize_features(df_features)
        print("✓ Features normalized")

    # STEP 4: Analyze Features
    df_analysis = None
    if analyze:
        print("\n[4/4] Analyzing features...")
        FeatureAnalyzer.print_feature_analysis(df_features)
        df_analysis = FeatureAnalyzer.get_feature_statistics(df_features)
        print("✓ Analysis complete")

    # Print Summary
    pipeline.print_feature_summary(df_features)

    # Save Features
    output_path = GENRES_DIR.parent / "features_extracted.csv"
    pipeline.save_features(df_features, output_path)

    return df_features, df_analysis


def analyze_discriminative_features(df_features: pd.DataFrame):
    """
    Analyze which features best discriminate between genres.

    Parameters:
    -----------
    df_features : pd.DataFrame
        Feature DataFrame
    """
    print("\n" + "="*80)
    print("DISCRIMINATIVE FEATURES ANALYSIS")
    print("="*80)

    # Find discriminative features
    discriminative = FeatureComparison.find_discriminative_features(
        df_features,
        method="between_group_variance"
    )

    print("\nTop 20 Discriminative Features (by F-ratio):")
    print("-"*80)
    for i, (feature, f_score) in enumerate(discriminative[:20], 1):
        print(f"{i:2d}. {feature:40s} F-score: {f_score:12.4f}")

    # Detailed comparison for top feature
    if len(discriminative) > 0:
        top_feature = discriminative[0][0]
        print(f"\n" + "-"*80)
        print(f"DETAILED ANALYSIS: {top_feature}")
        print("-"*80)

        comparison = FeatureComparison.compare_genres_per_feature(
            df_features,
            top_feature
        )

        for genre, stats in sorted(comparison.items()):
            print(f"\n{genre.upper()}:")
            print(f"  Mean: {stats['mean']:8.4f}")
            print(f"  Std:  {stats['std']:8.4f}")
            print(f"  Range: [{stats['min']:8.4f}, {stats['max']:8.4f}]")
            print(f"  Samples: {stats['count']}")


def generate_feature_report(df_features: pd.DataFrame, output_dir: Path = None):
    """
    Generate comprehensive feature report.

    Parameters:
    -----------
    df_features : pd.DataFrame
        Feature DataFrame
    output_dir : Path, optional
        Output directory for report
    """
    if output_dir is None:
        output_dir = GENRES_DIR.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING FEATURE REPORT")
    print("="*80)

    # Feature statistics
    stats = FeatureAnalyzer.get_feature_statistics(df_features)
    stats_path = output_dir / "feature_statistics.csv"
    stats.to_csv(stats_path)
    print(f"✓ Feature statistics: {stats_path}")

    # Stats by genre
    stats_by_genre = FeatureAnalyzer.get_features_by_genre_stats(df_features)

    for genre, genre_stats in stats_by_genre.items():
        pd.DataFrame({
            'mean': genre_stats['mean'],
            'std': genre_stats['std'],
            'min': genre_stats['min'],
            'max': genre_stats['max']
        }).to_csv(output_dir / f"features_{genre}.csv")

    print(f"✓ Per-genre statistics: {len(stats_by_genre)} files")

    # Outlier analysis
    outliers = FeatureAnalyzer.detect_outliers(df_features, method='iqr')
    outliers_df = pd.DataFrame([
        {'feature': k, **v} for k, v in outliers.items()
    ])
    if len(outliers_df) > 0:
        outliers_df.to_csv(output_dir / "outlier_analysis.csv", index=False)
        print(f"✓ Outlier analysis: {outliers_df}")

    print("="*80 + "\n")


def main():
    """Main execution."""
    # Run pipeline
    df_features, df_analysis = run_complete_pipeline(
        limit_files=None,  # Set to number to test with subset
        normalize=True,
        analyze=True
    )

    if df_features is not None:
        # Additional analysis
        analyze_discriminative_features(df_features)

        # Generate report
        generate_feature_report(df_features)

        return df_features, df_analysis

    return None, None


if __name__ == "__main__":
    features_df, analysis_df = main()
