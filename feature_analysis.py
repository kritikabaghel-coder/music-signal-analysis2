"""
Feature Analysis Tools

Utilities for analyzing and exploring extracted features.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Tools for analyzing extracted feature matrices."""

    @staticmethod
    def get_feature_statistics(df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed statistics for each feature.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame

        Returns:
        --------
        pd.DataFrame : Statistics for each feature
        """
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]

        stats = df_features[feature_cols].describe().T
        stats['skew'] = df_features[feature_cols].skew()
        stats['kurtosis'] = df_features[feature_cols].kurtosis()

        return stats

    @staticmethod
    def identify_constant_features(df_features: pd.DataFrame, threshold: float = 1e-10) -> list:
        """
        Identify features with near-zero variance.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        threshold : float
            Variance threshold

        Returns:
        --------
        list : Features with low variance
        """
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]

        constant_features = []
        for col in feature_cols:
            if df_features[col].var() < threshold:
                constant_features.append(col)

        return constant_features

    @staticmethod
    def identify_correlated_features(
        df_features: pd.DataFrame,
        correlation_threshold: float = 0.95
    ) -> list:
        """
        Identify highly correlated features.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        correlation_threshold : float
            Correlation threshold

        Returns:
        --------
        list : Pairs of highly correlated features
        """
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]

        corr_matrix = df_features[feature_cols].corr().abs()

        # Find highly correlated pairs
        correlated_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    correlated_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": float(corr_matrix.iloc[i, j])
                    })

        return correlated_pairs

    @staticmethod
    def get_features_by_genre_stats(df_features: pd.DataFrame) -> dict:
        """
        Get feature statistics grouped by genre.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame

        Returns:
        --------
        dict : Statistics per genre
        """
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]

        stats_by_genre = {}
        for genre in df_features['genre'].unique():
            genre_data = df_features[df_features['genre'] == genre][feature_cols]
            stats_by_genre[genre] = {
                "num_samples": len(genre_data),
                "mean": genre_data.mean(),
                "std": genre_data.std(),
                "min": genre_data.min(),
                "max": genre_data.max()
            }

        return stats_by_genre

    @staticmethod
    def detect_outliers(
        df_features: pd.DataFrame,
        method: str = "iqr",
        iqr_multiplier: float = 1.5
    ) -> dict:
        """
        Detect outliers in features.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        method : str
            'iqr' (Interquartile Range) or 'zscore'
        iqr_multiplier : float
            IQR multiplier for outlier detection

        Returns:
        --------
        dict : Outlier information
        """
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]

        outliers = {}

        if method == "iqr":
            for col in feature_cols:
                Q1 = df_features[col].quantile(0.25)
                Q3 = df_features[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR

                outlier_mask = (df_features[col] < lower_bound) | (df_features[col] > upper_bound)
                if outlier_mask.sum() > 0:
                    outliers[col] = {
                        "num_outliers": int(outlier_mask.sum()),
                        "percentage": float(outlier_mask.sum() / len(df_features) * 100),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }

        elif method == "zscore":
            from scipy import stats
            for col in feature_cols:
                z_scores = np.abs(stats.zscore(df_features[col]))
                outlier_mask = z_scores > 3

                if outlier_mask.sum() > 0:
                    outliers[col] = {
                        "num_outliers": int(outlier_mask.sum()),
                        "percentage": float(outlier_mask.sum() / len(df_features) * 100)
                    }

        return outliers

    @staticmethod
    def print_feature_analysis(df_features: pd.DataFrame) -> None:
        """
        Print comprehensive feature analysis report.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        """
        print("\n" + "="*80)
        print("FEATURE ANALYSIS REPORT")
        print("="*80)

        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]

        print(f"\nDataset Shape: {df_features.shape}")
        print(f"Number of Features: {len(feature_cols)}")
        print(f"Number of Samples: {len(df_features)}")
        print(f"Number of Genres: {df_features['genre'].nunique()}")

        # Constant features
        constant_features = FeatureAnalyzer.identify_constant_features(df_features)
        print(f"\nConstant Features (low variance): {len(constant_features)}")
        if len(constant_features) > 0:
            for feat in constant_features[:5]:
                print(f"  - {feat}")

        # Correlated features
        correlated = FeatureAnalyzer.identify_correlated_features(df_features, threshold=0.95)
        print(f"\nHighly Correlated Features (r > 0.95): {len(correlated)}")
        if len(correlated) > 0:
            for pair in correlated[:5]:
                print(f"  - {pair['feature_1']} <-> {pair['feature_2']}: r={pair['correlation']:.4f}")

        # Outliers
        outliers = FeatureAnalyzer.detect_outliers(df_features, method="iqr")
        print(f"\nFeatures with Outliers (IQR method): {len(outliers)}")
        if len(outliers) > 0:
            for feat, info in list(outliers.items())[:5]:
                print(f"  - {feat}: {info['num_outliers']} outliers ({info['percentage']:.1f}%)")

        # Missing values
        missing_count = df_features[feature_cols].isnull().sum().sum()
        print(f"\nMissing Values: {missing_count}")

        # Feature value ranges
        print("\n" + "-"*80)
        print("FEATURE VALUE RANGES")
        print("-"*80)
        stats = FeatureAnalyzer.get_feature_statistics(df_features)
        print(stats[['min', 'max', 'mean', 'std']].to_string())

        print("\n" + "="*80 + "\n")


class FeatureComparison:
    """Tools for comparing features across genres."""

    @staticmethod
    def compare_genres_per_feature(
        df_features: pd.DataFrame,
        feature_name: str
    ) -> dict:
        """
        Compare a feature across genres.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        feature_name : str
            Feature column name

        Returns:
        --------
        dict : Comparison statistics per genre
        """
        comparison = {}
        for genre in df_features['genre'].unique():
            genre_values = df_features[df_features['genre'] == genre][feature_name]
            comparison[genre] = {
                "mean": float(genre_values.mean()),
                "std": float(genre_values.std()),
                "min": float(genre_values.min()),
                "max": float(genre_values.max()),
                "count": len(genre_values)
            }

        return comparison

    @staticmethod
    def find_discriminative_features(
        df_features: pd.DataFrame,
        method: str = "variance"
    ) -> list:
        """
        Find features that best discriminate between genres.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        method : str
            'variance', 'between_group_variance', or 'f_ratio'

        Returns:
        --------
        list : Ranked features
        """
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]

        scores = []

        if method == "variance":
            # Features with high between-genre variance
            genre_means = df_features.groupby('genre')[feature_cols].mean()
            between_var = genre_means.var(axis=0)
            scores = sorted(zip(feature_cols, between_var), key=lambda x: x[1], reverse=True)

        elif method == "between_group_variance":
            # More sophisticated: between-group vs within-group variance
            for col in feature_cols:
                overall_mean = df_features[col].mean()
                bg_var = sum(len(df_features[df_features['genre'] == g]) * 
                           (df_features[df_features['genre'] == g][col].mean() - overall_mean)**2
                           for g in df_features['genre'].unique()) / (df_features['genre'].nunique() - 1)

                wg_var = sum((df_features[df_features['genre'] == g][col] - 
                            df_features[df_features['genre'] == g][col].mean()).pow(2).sum()
                           for g in df_features['genre'].unique()) / (len(df_features) - df_features['genre'].nunique())

                f_ratio = bg_var / (wg_var + 1e-10) if wg_var > 0 else 0
                scores.append((col, f_ratio))

            scores = sorted(scores, key=lambda x: x[1], reverse=True)

        return scores
