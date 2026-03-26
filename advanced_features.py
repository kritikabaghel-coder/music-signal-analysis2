import numpy as np
import librosa
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatures:
    """Advanced signal processing features beyond basic spectral features."""

    @staticmethod
    def compute_spectral_contrast(signal: np.ndarray, sr: int, n_fft: int = 2048) -> Dict:
        """
        Compute spectral contrast features.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate
        n_fft : int
            FFT size

        Returns:
        --------
        dict : Spectral contrast features
        """
        spectral_contrast = librosa.feature.spectral_contrast(
            y=signal,
            sr=sr,
            n_fft=n_fft
        )

        features = {}
        for i, contrast in enumerate(spectral_contrast):
            features[f"spectral_contrast_{i}_mean"] = float(np.mean(contrast))
            features[f"spectral_contrast_{i}_std"] = float(np.std(contrast))

        return features

    @staticmethod
    def compute_tempogram(signal: np.ndarray, sr: int) -> Dict:
        """
        Compute tempogram features related to rhythm.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Tempogram features
        """
        try:
            onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
            tempogram = librosa.feature.tempogram(onset_env=onset_env, sr=sr)

            return {
                "tempogram_mean": float(np.mean(tempogram)),
                "tempogram_std": float(np.std(tempogram)),
                "tempogram_max": float(np.max(tempogram)),
            }
        except Exception as e:
            logger.warning(f"Tempogram computation failed: {e}")
            return {}

    @staticmethod
    def compute_rmse(signal: np.ndarray, n_fft: int = 2048) -> Dict:
        """
        Compute RMSE (Root Mean Square Energy).

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        n_fft : int
            FFT size

        Returns:
        --------
        dict : RMSE statistics
        """
        rmse = librosa.feature.rms(y=signal, frame_length=n_fft)

        return {
            "rmse_mean": float(np.mean(rmse)),
            "rmse_std": float(np.std(rmse)),
            "rmse_max": float(np.max(rmse)),
            "rmse_min": float(np.min(rmse)),
        }

    @staticmethod
    def compute_cqt_features(signal: np.ndarray, sr: int) -> Dict:
        """
        Compute Constant-Q Transform features.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : CQT features
        """
        try:
            cqt_mag = np.abs(librosa.cqt(y=signal, sr=sr))

            return {
                "cqt_mean": float(np.mean(cqt_mag)),
                "cqt_std": float(np.std(cqt_mag)),
                "cqt_max": float(np.max(cqt_mag)),
            }
        except Exception as e:
            logger.warning(f"CQT computation failed: {e}")
            return {}

    @staticmethod
    def compute_onset_strength(signal: np.ndarray, sr: int) -> Dict:
        """
        Compute onset strength features (beat-related).

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Onset strength features
        """
        try:
            onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

            return {
                "onset_strength_mean": float(np.mean(onset_env)),
                "onset_strength_std": float(np.std(onset_env)),
                "onset_strength_max": float(np.max(onset_env)),
                "onset_strength_sum": float(np.sum(onset_env)),
            }
        except Exception as e:
            logger.warning(f"Onset strength computation failed: {e}")
            return {}

    @staticmethod
    def compute_energy_statistics(signal: np.ndarray) -> Dict:
        """
        Compute detailed energy statistics.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal

        Returns:
        --------
        dict : Energy statistics
        """
        energy = signal ** 2

        return {
            "energy_mean": float(np.mean(energy)),
            "energy_std": float(np.std(energy)),
            "energy_max": float(np.max(energy)),
            "energy_min": float(np.min(energy)),
            "energy_sum": float(np.sum(energy)),
        }

    @staticmethod
    def compute_signal_statistics(signal: np.ndarray) -> Dict:
        """
        Compute basic signal statistics.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal

        Returns:
        --------
        dict : Signal statistics
        """
        return {
            "signal_mean": float(np.mean(signal)),
            "signal_std": float(np.std(signal)),
            "signal_max": float(np.max(signal)),
            "signal_min": float(np.min(signal)),
            "signal_skew": float(np.mean(signal**3) / (np.std(signal)**3 + 1e-10)),
        }


class TimeFrequencyAnalysis:
    """Time-frequency analysis utilities."""

    @staticmethod
    def compute_spectrogram_statistics(
        signal: np.ndarray,
        sr: int,
        n_fft: int = 2048
    ) -> Dict:
        """
        Compute comprehensive spectrogram statistics.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate
        n_fft : int
            FFT size

        Returns:
        --------
        dict : Spectrogram statistics
        """
        stft_matrix = librosa.stft(signal, n_fft=n_fft)
        magnitude_spec = np.abs(stft_matrix)

        # Statistics across time and frequency
        mean_mag = np.mean(magnitude_spec, axis=1)
        std_mag = np.std(magnitude_spec, axis=1)

        return {
            "spectrogram_mean_of_means": float(np.mean(mean_mag)),
            "spectrogram_std_of_means": float(np.std(mean_mag)),
            "spectrogram_mean_of_stds": float(np.mean(std_mag)),
        }

    @staticmethod
    def compute_derivative_features(
        signal: np.ndarray,
        sr: int,
        n_fft: int = 2048
    ) -> Dict:
        """
        Compute spectral derivative features (delta features).

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate
        n_fft : int
            FFT size

        Returns:
        --------
        dict : Derivative features
        """
        stft_matrix = librosa.stft(signal, n_fft=n_fft)
        magnitude_spec = np.abs(stft_matrix)

        # Compute delta (first derivative)
        delta_spec = librosa.feature.delta(magnitude_spec)

        # Compute delta-delta (second derivative)
        delta_delta_spec = librosa.feature.delta(magnitude_spec, order=2)

        return {
            "delta_mean": float(np.mean(delta_spec)),
            "delta_std": float(np.std(delta_spec)),
            "delta_max": float(np.max(delta_spec)),
            "delta_delta_mean": float(np.mean(delta_delta_spec)),
            "delta_delta_std": float(np.std(delta_delta_spec)),
        }
