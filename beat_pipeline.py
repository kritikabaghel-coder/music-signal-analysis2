"""
Beat Detection Pipeline - Orchestration

Orchestrates beat detection across dataset and manages results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

from beat_detector import BeatDetector
from config import GENRES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeatExtractionPipeline:
    """Orchestrate beat detection pipeline."""

    def __init__(self, sr: int = 22050):
        """
        Initialize pipeline.

        Parameters:
        -----------
        sr : int
            Sample rate for audio loading
        """
        self.sr = sr
        self.detector = BeatDetector(sr=sr)
        self.results_df = None
        self.statistics = None

    def process_dataset(
        self,
        df_metadata: pd.DataFrame,
        output_path: Path = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Process all audio files in dataset for beat detection.

        Parameters:
        -----------
        df_metadata : pd.DataFrame
            Metadata DataFrame with file_path and genre
        output_path : Path, optional
            Path to save results CSV
        verbose : bool
            Print progress

        Returns:
        --------
        pd.DataFrame
            Beat detection results
        """
        logger.info("Starting beat detection pipeline...")

        # Extract beats from all files
        file_paths = df_metadata["file_path"].tolist()
        genres = df_metadata["genre"].tolist()

        beat_results = self.detector.extract_beats_batch(
            file_paths,
            genres=genres,
            verbose=verbose
        )

        # Convert to DataFrame
        self.results_df = pd.DataFrame(beat_results)

        # Remove beat_times and beat_frames from final dataframe
        # (keep only numeric data for easier analysis)
        df_results = self.results_df[[
            "file_path", "genre", "tempo_bpm", "n_beats",
            "duration_seconds", "status"
        ]].copy()

        # Save if path provided
        if output_path:
            df_results.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        return df_results

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics from beat detection results.

        Returns:
        --------
        dict : Statistics
        """
        if self.results_df is None:
            raise ValueError("No results. Run process_dataset() first.")

        self.statistics = self.detector.get_statistics(
            self.results_df.to_dict('records')
        )

        return self.statistics

    def print_results(self, n_samples: int = 10) -> None:
        """
        Print sample results and statistics.

        Parameters:
        -----------
        n_samples : int
            Number of sample results to print
        """
        if self.results_df is None:
            print("No results available.")
            return

        print("\n" + "="*80)
        print("BEAT DETECTION RESULTS - SAMPLE")
        print("="*80)
        print(f"Showing first {min(n_samples, len(self.results_df))} results:")
        print()

        display_df = self.results_df[[
            "file_path", "genre", "tempo_bpm", "n_beats", "duration_seconds"
        ]].head(n_samples).copy()

        for col in ["tempo_bpm", "duration_seconds"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

        print(display_df.to_string(index=False))
        print()

        # Statistics
        if self.statistics is None:
            self.compute_statistics()

        self.detector.print_statistics(self.statistics)

    def get_tempo_by_genre(self) -> pd.DataFrame:
        """
        Get average tempo per genre.

        Returns:
        --------
        pd.DataFrame
            Genre and average BPM
        """
        if self.results_df is None:
            raise ValueError("No results. Run process_dataset() first.")

        successful = self.results_df[
            self.results_df["status"] == "success"
        ].copy()

        if len(successful) == 0:
            return pd.DataFrame()

        genre_stats = successful.groupby("genre").agg({
            "tempo_bpm": ["mean", "std", "min", "max", "count"],
            "n_beats": "mean"
        }).round(2)

        genre_stats.columns = ["BPM_mean", "BPM_std", "BPM_min", "BPM_max",
                               "count", "avg_beats"]

        return genre_stats.reset_index()

    def get_high_tempo_tracks(
        self,
        threshold: float = 130.0,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get tracks with highest tempo.

        Parameters:
        -----------
        threshold : float
            Only include tracks above this BPM
        top_n : int
            Number of tracks to return

        Returns:
        --------
        pd.DataFrame
            High-tempo tracks
        """
        if self.results_df is None:
            raise ValueError("No results. Run process_dataset() first.")

        high_tempo = self.results_df[
            self.results_df["tempo_bpm"] > threshold
        ].nlargest(top_n, "tempo_bpm")

        return high_tempo[[
            "file_path", "genre", "tempo_bpm", "n_beats"
        ]]

    def get_low_tempo_tracks(
        self,
        threshold: float = 80.0,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get tracks with lowest tempo.

        Parameters:
        -----------
        threshold : float
            Only include tracks below this BPM
        top_n : int
            Number of tracks to return

        Returns:
        --------
        pd.DataFrame
            Low-tempo tracks
        """
        if self.results_df is None:
            raise ValueError("No results. Run process_dataset() first.")

        low_tempo = self.results_df[
            self.results_df["tempo_bpm"] < threshold
        ].nsmallest(top_n, "tempo_bpm")

        return low_tempo[[
            "file_path", "genre", "tempo_bpm", "n_beats"
        ]]

    def compare_genres_by_tempo(self) -> None:
        """Print detailed genre comparison."""
        genre_stats = self.get_tempo_by_genre()

        print("\n" + "="*80)
        print("TEMPO COMPARISON BY GENRE")
        print("="*80)
        print(genre_stats.to_string(index=False))
        print("="*80 + "\n")

    def plot_tempo_distribution(self):
        """
        Plot tempo distribution (requires matplotlib).

        Returns:
        --------
        fig, ax : matplotlib figure and axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Skipping plot.")
            return None

        if self.results_df is None:
            raise ValueError("No results. Run process_dataset() first.")

        successful = self.results_df[
            self.results_df["status"] == "success"
        ]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Histogram of all tempos
        axes[0].hist(successful["tempo_bpm"], bins=30, color="steelblue",
                     alpha=0.7, edgecolor="black")
        axes[0].set_xlabel("Tempo (BPM)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution of Tempos Across Dataset")
        axes[0].grid(alpha=0.3)

        # Box plot by genre
        genres = sorted(successful["genre"].unique())
        tempo_by_genre = [successful[successful["genre"] == g]["tempo_bpm"]
                         for g in genres]

        axes[1].boxplot(tempo_by_genre, labels=genres)
        axes[1].set_ylabel("Tempo (BPM)")
        axes[1].set_title("Tempo Distribution by Genre")
        axes[1].grid(alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()

        return fig, axes

    def plot_beat_waveform(self, file_idx: int = 0):
        """
        Plot waveform with detected beats overlaid.

        Parameters:
        -----------
        file_idx : int
            Index of file in results

        Returns:
        --------
        fig, ax : matplotlib figure and axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Skipping plot.")
            return None

        if self.results_df is None:
            raise ValueError("No results. Run process_dataset() first.")

        # Get result and load audio
        result = self.results_df.iloc[file_idx]
        file_path = result["file_path"]

        try:
            import librosa

            y, sr = librosa.load(file_path, sr=self.sr)
            beat_frames = self.detector.detect_beats(y, sr)[0]
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            # Create waveform plot
            fig, ax = plt.subplots(figsize=(14, 5))

            # Plot waveform
            time_axis = np.arange(len(y)) / sr
            ax.plot(time_axis, y, alpha=0.7, color="steelblue", linewidth=0.5)

            # Plot beat positions
            for beat_time in beat_times:
                ax.axvline(beat_time, color="red", alpha=0.5, linewidth=1)

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Waveform with Detected Beats\n"
                        f"{result['genre']} - {result['tempo_bpm']:.1f} BPM "
                        f"({len(beat_frames)} beats)")
            ax.grid(alpha=0.3)

            return fig, ax

        except Exception as e:
            logger.error(f"Failed to plot: {e}")
            return None
