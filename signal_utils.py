import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional


class SignalValidator:
    """Utilities for signal validation and metadata extraction."""

    @staticmethod
    def is_valid_audio_file(file_path: Path) -> bool:
        """Check if file is a valid audio file."""
        valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        return file_path.suffix.lower() in valid_extensions

    @staticmethod
    def load_signal(
        file_path: Path,
        sr: Optional[int] = None,
        mono: bool = True,
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio signal with error handling.

        Parameters:
        -----------
        file_path : Path
            Path to audio file
        sr : int, optional
            Target sample rate (None preserves original)
        mono : bool
            Convert to mono if True
        offset : float
            Start reading after this time (seconds)
        duration : float, optional
            Only load this much audio (seconds)

        Returns:
        --------
        signal : np.ndarray
            Audio samples
        sr : int
            Sample rate
        """
        signal, sr = librosa.load(
            file_path,
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration
        )
        return signal, sr

    @staticmethod
    def get_signal_duration(
        file_path: Path,
        sr: Optional[int] = None
    ) -> float:
        """Get audio duration in seconds."""
        try:
            duration = librosa.get_duration(filename=str(file_path), sr=sr)
            return duration
        except Exception as e:
            raise ValueError(f"Cannot get duration: {e}")

    @staticmethod
    def get_signal_info(file_path: Path) -> dict:
        """Get comprehensive signal information."""
        try:
            signal, sr = librosa.load(file_path, sr=None, mono=True)
            duration = librosa.get_duration(y=signal, sr=sr)

            return {
                "sample_rate": sr,
                "duration": duration,
                "num_samples": len(signal),
                "rms_energy": float(np.sqrt(np.mean(signal ** 2))),
                "peak_amplitude": float(np.max(np.abs(signal))),
                "mean_amplitude": float(np.mean(np.abs(signal)))
            }
        except Exception as e:
            raise ValueError(f"Cannot extract signal info: {e}")

    @staticmethod
    def validate_signal_quality(
        signal: np.ndarray,
        sr: int,
        min_duration: float = 0.5,
        min_rms: float = 0.001
    ) -> Tuple[bool, str]:
        """
        Validate signal quality.

        Parameters:
        -----------
        signal : np.ndarray
            Audio samples
        sr : int
            Sample rate
        min_duration : float
            Minimum duration in seconds
        min_rms : float
            Minimum RMS energy threshold

        Returns:
        --------
        is_valid : bool
        message : str
        """
        duration = len(signal) / sr

        if duration < min_duration:
            return False, f"Duration too short: {duration:.2f}s"

        rms = np.sqrt(np.mean(signal ** 2))
        if rms < min_rms:
            return False, f"Energy too low: RMS={rms:.6f}"

        if np.any(np.isnan(signal)):
            return False, "Signal contains NaN values"

        if np.any(np.isinf(signal)):
            return False, "Signal contains infinite values"

        return True, "Valid"


class DatasetStats:
    """Statistics computation for audio datasets."""

    @staticmethod
    def compute_duration_stats(durations: np.ndarray) -> dict:
        """Compute duration statistics."""
        return {
            "mean": float(np.mean(durations)),
            "median": float(np.median(durations)),
            "min": float(np.min(durations)),
            "max": float(np.max(durations)),
            "std": float(np.std(durations)),
            "total": float(np.sum(durations))
        }

    @staticmethod
    def compute_energy_stats(signals: list, sr: int) -> dict:
        """Compute signal energy statistics."""
        rms_values = [np.sqrt(np.mean(s ** 2)) for s in signals]
        rms_values = np.array(rms_values)

        return {
            "mean_rms": float(np.mean(rms_values)),
            "median_rms": float(np.median(rms_values)),
            "min_rms": float(np.min(rms_values)),
            "max_rms": float(np.max(rms_values)),
            "std_rms": float(np.std(rms_values))
        }
