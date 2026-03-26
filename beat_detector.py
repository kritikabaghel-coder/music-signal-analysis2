"""
Beat Detection and BPM Estimation using Signal Processing

Implements beat tracking and tempo estimation using librosa.
Focus: Onset detection, peak picking, beat frame conversion.
"""

import numpy as np
import librosa
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeatDetector:
    """Detect beats and estimate tempo (BPM) from audio signals."""

    def __init__(self, sr: int = 22050):
        """
        Initialize beat detector.

        Parameters:
        -----------
        sr : int
            Target sample rate for audio loading
        """
        self.sr = sr
        self.results = {}

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Parameters:
        -----------
        file_path : str
            Path to audio file

        Returns:
        --------
        y : np.ndarray
            Audio time series
        sr : int
            Sample rate
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sr)
            logger.debug(f"Loaded audio: {len(y)} samples, {sr} Hz")
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def compute_onset_strength(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute onset strength envelope.

        Onset strength measures the energy in high-frequency bands,
        which increases at the start of percussive sounds (beats).

        Parameters:
        -----------
        y : np.ndarray
            Audio time series
        sr : int
            Sample rate

        Returns:
        --------
        onset_strength : np.ndarray
            Onset strength envelope (one value per frame, ~44 Hz)
        """
        try:
            # Compute onset strength using energy in frequency bands
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            
            logger.debug(f"Onset strength shape: {onset_strength.shape}")
            return onset_strength
        except Exception as e:
            logger.error(f"Failed to compute onset strength: {e}")
            raise

    def detect_beats(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
        """
        Detect beats and estimate tempo using librosa.beat.beat_track().

        This function:
        1. Computes onset strength
        2. Finds tempo using autocorrelation
        3. Aligns beats to grid
        4. Returns beat frames and tempo

        Parameters:
        -----------
        y : np.ndarray
            Audio time series
        sr : int
            Sample rate

        Returns:
        --------
        beat_frames : np.ndarray
            Frame indices of detected beats
        tempo : float
            Estimated tempo in BPM
        """
        try:
            # Compute onset strength dynamically in beat_track
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr
            )
            
            logger.debug(f"Detected tempo: {tempo:.1f} BPM, "
                        f"{len(beat_frames)} beats")
            return beat_frames, tempo
        except Exception as e:
            logger.error(f"Failed to detect beats: {e}")
            raise

    def convert_frames_to_time(
        self,
        frames: np.ndarray,
        sr: int,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Convert frame indices to time in seconds.

        Frame index → Time conversion:
        time = frame_idx * hop_length / sr

        Parameters:
        -----------
        frames : np.ndarray
            Frame indices
        sr : int
            Sample rate
        hop_length : int
            Hop length used in STFT (default 512)

        Returns:
        --------
        times : np.ndarray
            Times in seconds
        """
        try:
            times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
            logger.debug(f"Converted {len(frames)} frames to times")
            return times
        except Exception as e:
            logger.error(f"Failed to convert frames: {e}")
            raise

    def extract_beat_info(
        self,
        file_path: str,
        genre: str = None
    ) -> Dict[str, Any]:
        """
        Extract complete beat information from audio file.

        Parameters:
        -----------
        file_path : str
            Path to audio file
        genre : str, optional
            Genre label

        Returns:
        --------
        dict : Beat information
            ├─ file_path
            ├─ genre
            ├─ tempo (BPM)
            ├─ n_beats (number of beats detected)
            ├─ beat_times (seconds)
            ├─ beat_frames
            ├─ duration (total audio length in samples)
            └─ status (success/error)
        """
        try:
            # Load audio
            y, sr = self.load_audio(file_path)

            # Detect beats and tempo
            beat_frames, tempo = self.detect_beats(y, sr)

            # Convert frames to time
            beat_times = self.convert_frames_to_time(beat_frames, sr)

            # Compute total duration
            duration = len(y) / sr

            result = {
                "file_path": file_path,
                "genre": genre,
                "tempo_bpm": float(tempo),
                "n_beats": int(len(beat_frames)),
                "beat_times": beat_times,
                "beat_frames": beat_frames,
                "duration_seconds": float(duration),
                "status": "success"
            }

            logger.info(f"✓ Extracted beats from {file_path}: "
                       f"{tempo:.1f} BPM, {len(beat_frames)} beats")

            return result

        except Exception as e:
            logger.error(f"✗ Failed to extract beats from {file_path}: {e}")
            return {
                "file_path": file_path,
                "genre": genre,
                "tempo_bpm": np.nan,
                "n_beats": 0,
                "beat_times": [],
                "beat_frames": [],
                "duration_seconds": np.nan,
                "status": f"error: {str(e)}"
            }

    def extract_beats_batch(
        self,
        file_list: list,
        genres: list = None,
        verbose: bool = True
    ) -> list:
        """
        Extract beats from multiple audio files.

        Parameters:
        -----------
        file_list : list
            List of file paths
        genres : list, optional
            Corresponding genre labels
        verbose : bool
            Print progress

        Returns:
        --------
        list : List of beat information dictionaries
        """
        results = []

        for idx, file_path in enumerate(file_list):
            if verbose:
                print(f"[{idx+1}/{len(file_list)}] Processing {file_path}...")

            genre = genres[idx] if genres and idx < len(genres) else None
            result = self.extract_beat_info(file_path, genre)
            results.append(result)

        return results

    def compute_beat_density(
        self,
        beat_times: np.ndarray,
        duration: float
    ) -> float:
        """
        Compute beat density (beats per second).

        Parameters:
        -----------
        beat_times : np.ndarray
            Beat times in seconds
        duration : float
            Total duration in seconds

        Returns:
        --------
        density : float
            Beats per second
        """
        if duration == 0:
            return 0.0
        return len(beat_times) / duration

    def estimate_beat_regularity(self, beat_times: np.ndarray) -> float:
        """
        Estimate regularity of beats (lower = more regular).

        Computes coefficient of variation (std / mean) of 
        inter-beat intervals.

        Parameters:
        -----------
        beat_times : np.ndarray
            Beat times in seconds

        Returns:
        --------
        regularity : float
            Coefficient of variation (0.0 = perfectly regular, >0.5 = irregular)
        """
        if len(beat_times) < 2:
            return np.nan

        # Compute inter-beat intervals
        intervals = np.diff(beat_times)

        if len(intervals) == 0 or np.mean(intervals) == 0:
            return np.nan

        # Coefficient of variation
        cv = np.std(intervals) / np.mean(intervals)
        return float(cv)

    def get_statistics(self, beat_results: list) -> Dict[str, Any]:
        """
        Compute statistics from beat detection results.

        Parameters:
        -----------
        beat_results : list
            List of beat information dictionaries

        Returns:
        --------
        dict : Statistics
            ├─ total_files
            ├─ successful
            ├─ failed
            ├─ mean_tempo_bpm
            ├─ std_tempo_bpm
            ├─ min_tempo_bpm
            ├─ max_tempo_bpm
            ├─ mean_n_beats
            └─ per_genre_stats
        """
        successful = [r for r in beat_results if r["status"] == "success"]

        if not successful:
            return {
                "total_files": len(beat_results),
                "successful": 0,
                "failed": len(beat_results),
                "message": "No successful detections"
            }

        tempos = [r["tempo_bpm"] for r in successful
                 if not np.isnan(r["tempo_bpm"])]
        n_beats_list = [r["n_beats"] for r in successful]

        stats = {
            "total_files": len(beat_results),
            "successful": len(successful),
            "failed": len(beat_results) - len(successful),
            "mean_tempo_bpm": float(np.mean(tempos)),
            "std_tempo_bpm": float(np.std(tempos)),
            "min_tempo_bpm": float(np.min(tempos)),
            "max_tempo_bpm": float(np.max(tempos)),
            "median_tempo_bpm": float(np.median(tempos)),
            "mean_n_beats": float(np.mean(n_beats_list)),
            "std_n_beats": float(np.std(n_beats_list))
        }

        # Per-genre statistics
        genres = {r["genre"] for r in successful if r.get("genre")}
        per_genre = {}

        for genre in sorted(genres):
            genre_results = [r for r in successful if r["genre"] == genre]
            genre_tempos = [r["tempo_bpm"] for r in genre_results
                           if not np.isnan(r["tempo_bpm"])]

            if genre_tempos:
                per_genre[genre] = {
                    "count": len(genre_results),
                    "mean_bpm": float(np.mean(genre_tempos)),
                    "std_bpm": float(np.std(genre_tempos)),
                    "min_bpm": float(np.min(genre_tempos)),
                    "max_bpm": float(np.max(genre_tempos))
                }

        stats["per_genre_stats"] = per_genre

        return stats

    def print_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Print statistics in formatted output.

        Parameters:
        -----------
        stats : dict
            Statistics dictionary from get_statistics()
        """
        print("\n" + "="*80)
        print("BEAT DETECTION STATISTICS")
        print("="*80)

        print(f"Total files processed: {stats['total_files']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")

        if stats['successful'] == 0:
            print("No successful detections.")
            return

        print("\nOVERALL TEMPO STATISTICS (BPM):")
        print("-"*80)
        print(f"  Mean:   {stats['mean_tempo_bpm']:.2f}")
        print(f"  Std:    {stats['std_tempo_bpm']:.2f}")
        print(f"  Min:    {stats['min_tempo_bpm']:.2f}")
        print(f"  Max:    {stats['max_tempo_bpm']:.2f}")
        print(f"  Median: {stats['median_tempo_bpm']:.2f}")

        print(f"\nBEAT COUNT STATISTICS:")
        print("-"*80)
        print(f"  Mean:   {stats['mean_n_beats']:.1f} beats")
        print(f"  Std:    {stats['std_n_beats']:.1f}")

        if stats.get('per_genre_stats'):
            print(f"\nPER-GENRE TEMPO STATISTICS:")
            print("-"*80)
            for genre, genre_stats in stats['per_genre_stats'].items():
                print(f"  {genre:12s} | "
                      f"Count: {genre_stats['count']:3d} | "
                      f"Mean: {genre_stats['mean_bpm']:6.1f} BPM | "
                      f"Std: {genre_stats['std_bpm']:5.1f} | "
                      f"Range: {genre_stats['min_bpm']:6.1f}-{genre_stats['max_bpm']:6.1f}")

        print("="*80 + "\n")
