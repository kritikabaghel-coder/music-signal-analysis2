import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

from feature_extractor import SpectralFeatureExtractor
from config import GENRES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureExtractionPipeline:
    """Pipeline for extracting and processing features from audio dataset."""

    def __init__(
        self,
        dataset_df: pd.DataFrame,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        normalize: bool = True
    ):
        """
        Parameters:
        -----------
        dataset_df : pd.DataFrame
            Dataset DataFrame from dataset_loader
        n_mfcc : int
            Number of MFCCs
        n_fft : int
            FFT size
        normalize : bool
            Whether to normalize features
        """
        self.dataset_df = dataset_df
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.extractor = SpectralFeatureExtractor(
            n_mfcc=n_mfcc,
            n_fft=n_fft
        )
        self.features_list: List[Dict] = []
        self.errors: List[Dict] = []

    def _load_audio(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file safely."""
        try:
            signal, sr = librosa.load(file_path, sr=None, mono=True)
            return signal, sr
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None, None

    def extract_features_from_file(self, row: pd.Series) -> Optional[Dict]:
        """
        Extract all features from a single audio file.

        Parameters:
        -----------
        row : pd.Series
            Row from dataset DataFrame

        Returns:
        --------
        dict : Feature dictionary or None if extraction fails
        """
        file_path = row['file_path']
        genre = row['genre']

        signal, sr = self._load_audio(file_path)

        if signal is None or sr is None:
            error_info = {
                "file_path": file_path,
                "genre": genre,
                "error": "Failed to load audio"
            }
            self.errors.append(error_info)
            return None

        try:
            features = self.extractor.extract_features(signal, sr)
            features["file_path"] = file_path
            features["genre"] = genre
            return features

        except Exception as e:
            error_info = {
                "file_path": file_path,
                "genre": genre,
                "error": str(e)
            }
            self.errors.append(error_info)
            logger.error(f"Feature extraction failed for {file_path}: {e}")
            return None

    def process_dataset(self) -> pd.DataFrame:
        """
        Process entire dataset and extract features.

        Returns:
        --------
        pd.DataFrame : Feature matrix with genre labels
        """
        logger.info(f"Processing {len(self.dataset_df)} audio files...")

        for idx, row in self.dataset_df.iterrows():
            features = self.extract_features_from_file(row)
            if features is not None:
                self.features_list.append(features)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(self.dataset_df)} files")

        df_features = pd.DataFrame(self.features_list)

        logger.info(f"Successfully extracted features from {len(df_features)} files")
        logger.info(f"Failed to process {len(self.errors)} files")

        return df_features

    def normalize_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize/scale features using StandardScaler.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame

        Returns:
        --------
        pd.DataFrame : Normalized feature DataFrame
        """
        if not self.normalize or self.scaler is None:
            return df_features

        # Separate features and metadata
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]
        
        metadata = df_features[["file_path", "genre"]].copy()
        features_only = df_features[feature_cols].copy()

        # Fit and transform
        features_normalized = self.scaler.fit_transform(features_only)
        df_normalized = pd.DataFrame(features_normalized, columns=feature_cols)

        # Add metadata back
        df_normalized["file_path"] = metadata["file_path"].values
        df_normalized["genre"] = metadata["genre"].values

        logger.info("Features normalized using StandardScaler")

        return df_normalized

    def get_feature_info(self, df_features: pd.DataFrame) -> Dict:
        """
        Get information about the feature matrix.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame

        Returns:
        --------
        dict : Feature matrix information
        """
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]

        return {
            "num_files": len(df_features),
            "num_features": len(feature_cols),
            "shape": df_features[feature_cols].shape,
            "feature_columns": feature_cols,
            "num_genres": df_features["genre"].nunique(),
            "genres": sorted(df_features["genre"].unique().tolist()),
            "missing_values": df_features[feature_cols].isnull().sum().sum()
        }

    def print_feature_summary(self, df_features: pd.DataFrame) -> None:
        """
        Print comprehensive feature summary.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        """
        info = self.get_feature_info(df_features)

        print("\n" + "="*80)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*80)

        print(f"\nFeature Matrix Shape: {info['shape']}")
        print(f"Total Audio Files: {info['num_files']}")
        print(f"Number of Features: {info['num_features']}")
        print(f"Number of Genres: {info['num_genres']}")
        print(f"Genres: {', '.join(info['genres'])}")
        print(f"Missing Values: {info['missing_values']}")

        if self.normalize:
            print("\n✓ Features normalized using StandardScaler")

        if len(self.errors) > 0:
            print(f"\n⚠ Failed to process {len(self.errors)} files")

        print("\n" + "-"*80)
        print("FEATURE COLUMNS")
        print("-"*80)
        for i, col in enumerate(info['feature_columns'], 1):
            print(f"{i:3d}. {col}")

        print("\n" + "-"*80)
        print("SAMPLE FEATURE VECTOR")
        print("-"*80)
        sample_row = df_features.iloc[0]
        print(f"File: {sample_row['file_path']}")
        print(f"Genre: {sample_row['genre']}\n")

        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]
        for col in feature_cols[:10]:
            print(f"  {col:30s}: {sample_row[col]:10.4f}")
        print(f"  ... ({len(feature_cols) - 10} more features)")

        print("\n" + "-"*80)
        print("FEATURE STATISTICS")
        print("-"*80)
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]
        stats_df = df_features[feature_cols].describe()
        print(stats_df.to_string())

        print("\n" + "-"*80)
        print("FILES PER GENRE")
        print("-"*80)
        genre_counts = df_features["genre"].value_counts().sort_index()
        for genre, count in genre_counts.items():
            print(f"  {genre:12s}: {count:3d} files")

        print("\n" + "="*80 + "\n")

    def save_features(self, df_features: pd.DataFrame, output_path: Path) -> None:
        """
        Save feature matrix to CSV.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        output_path : Path
            Output file path
        """
        df_features.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        print(f"✓ Feature matrix saved to {output_path}")

    def get_features_by_genre(
        self,
        df_features: pd.DataFrame,
        genre: str
    ) -> np.ndarray:
        """
        Get feature matrix for a specific genre.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        genre : str
            Genre name

        Returns:
        --------
        np.ndarray : Features for the genre (n_samples, n_features)
        """
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]
        
        genre_df = df_features[df_features["genre"] == genre]
        return genre_df[feature_cols].values
