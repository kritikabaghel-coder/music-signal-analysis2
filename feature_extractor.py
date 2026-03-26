import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """Extract frequency-domain and spectral features from audio signals."""

    def __init__(self, sr: int = None, n_mfcc: int = 13, n_fft: int = 2048):
        """
        Parameters:
        -----------
        sr : int, optional
            Sample rate (None to use file's native)
        n_mfcc : int
            Number of MFCCs to extract
        n_fft : int
            FFT window size
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.errors: List[Dict] = []

    def compute_fft(self, signal: np.ndarray, sr: int) -> Dict:
        """
        Compute FFT and extract magnitude spectrum features.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : FFT features including dominant frequencies
        """
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result)
        frequency = np.fft.fftfreq(len(signal), 1/sr)

        # Use only positive frequencies
        positive_idx = frequency >= 0
        magnitude_positive = magnitude[positive_idx]
        frequency_positive = frequency[positive_idx]

        # Dominant frequencies (top 5)
        top_indices = np.argsort(magnitude_positive)[-5:][::-1]
        dominant_freqs = frequency_positive[top_indices]

        return {
            "fft_mean": float(np.mean(magnitude_positive)),
            "fft_std": float(np.std(magnitude_positive)),
            "fft_max": float(np.max(magnitude_positive)),
            "dominant_freq_1": float(dominant_freqs[0]) if len(dominant_freqs) > 0 else 0.0,
            "dominant_freq_2": float(dominant_freqs[1]) if len(dominant_freqs) > 1 else 0.0,
            "dominant_freq_3": float(dominant_freqs[2]) if len(dominant_freqs) > 2 else 0.0,
        }

    def compute_stft(self, signal: np.ndarray) -> Dict:
        """
        Compute STFT spectrogram features.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal

        Returns:
        --------
        dict : STFT-based spectrogram features
        """
        stft_matrix = librosa.stft(signal, n_fft=self.n_fft)
        magnitude_spec = np.abs(stft_matrix)

        return {
            "stft_mean": float(np.mean(magnitude_spec)),
            "stft_std": float(np.std(magnitude_spec)),
            "stft_max": float(np.max(magnitude_spec)),
            "stft_median": float(np.median(magnitude_spec)),
        }

    def compute_mel_spectrogram(self, signal: np.ndarray, sr: int) -> Dict:
        """
        Compute Mel spectrogram features.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Mel spectrogram features
        """
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=512,
            n_mels=128
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return {
            "mel_spec_mean": float(np.mean(mel_spec_db)),
            "mel_spec_std": float(np.std(mel_spec_db)),
            "mel_spec_max": float(np.max(mel_spec_db)),
            "mel_spec_min": float(np.min(mel_spec_db)),
        }

    def compute_mfcc(self, signal: np.ndarray, sr: int) -> Dict:
        """
        Compute MFCC (Mel-Frequency Cepstral Coefficients).

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Mean MFCC coefficients
        """
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft)

        # Time-averaged MFCCs
        mfcc_mean = np.mean(mfcc, axis=1)

        features = {}
        for i, coeff in enumerate(mfcc_mean):
            features[f"mfcc_{i}"] = float(coeff)

        return features

    def compute_spectral_centroid(self, signal: np.ndarray, sr: int) -> Dict:
        """
        Compute spectral centroid.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Spectral centroid statistics
        """
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=self.n_fft)

        return {
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
            "spectral_centroid_max": float(np.max(spectral_centroid)),
            "spectral_centroid_min": float(np.min(spectral_centroid)),
        }

    def compute_spectral_bandwidth(self, signal: np.ndarray, sr: int) -> Dict:
        """
        Compute spectral bandwidth.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Spectral bandwidth statistics
        """
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=self.n_fft)

        return {
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
            "spectral_bandwidth_max": float(np.max(spectral_bandwidth)),
            "spectral_bandwidth_min": float(np.min(spectral_bandwidth)),
        }

    def compute_zero_crossing_rate(self, signal: np.ndarray) -> Dict:
        """
        Compute zero-crossing rate.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal

        Returns:
        --------
        dict : Zero-crossing rate statistics
        """
        zcr = librosa.feature.zero_crossing_rate(signal)

        return {
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "zcr_max": float(np.max(zcr)),
            "zcr_min": float(np.min(zcr)),
        }

    def compute_chroma_features(self, signal: np.ndarray, sr: int) -> Dict:
        """
        Compute chroma features.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Chroma feature statistics for each pitch class
        """
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=self.n_fft)

        features = {}
        chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        for i, name in enumerate(chroma_names):
            features[f"chroma_{name}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{name}_std"] = float(np.std(chroma[i]))

        return features

    def compute_spectral_rolloff(self, signal: np.ndarray, sr: int) -> Dict:
        """
        Compute spectral rolloff frequency.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Spectral rolloff statistics
        """
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=self.n_fft)

        return {
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spectral_rolloff_std": float(np.std(spectral_rolloff)),
        }

    def extract_features(self, signal: np.ndarray, sr: int) -> Dict:
        """
        Extract all features from audio signal.

        Parameters:
        -----------
        signal : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        dict : Combined feature vector
        """
        features = {}

        try:
            features.update(self.compute_fft(signal, sr))
        except Exception as e:
            logger.warning(f"FFT extraction failed: {e}")

        try:
            features.update(self.compute_stft(signal))
        except Exception as e:
            logger.warning(f"STFT extraction failed: {e}")

        try:
            features.update(self.compute_mel_spectrogram(signal, sr))
        except Exception as e:
            logger.warning(f"Mel spectrogram extraction failed: {e}")

        try:
            features.update(self.compute_mfcc(signal, sr))
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")

        try:
            features.update(self.compute_spectral_centroid(signal, sr))
        except Exception as e:
            logger.warning(f"Spectral centroid extraction failed: {e}")

        try:
            features.update(self.compute_spectral_bandwidth(signal, sr))
        except Exception as e:
            logger.warning(f"Spectral bandwidth extraction failed: {e}")

        try:
            features.update(self.compute_zero_crossing_rate(signal))
        except Exception as e:
            logger.warning(f"Zero-crossing rate extraction failed: {e}")

        try:
            features.update(self.compute_chroma_features(signal, sr))
        except Exception as e:
            logger.warning(f"Chroma extraction failed: {e}")

        try:
            features.update(self.compute_spectral_rolloff(signal, sr))
        except Exception as e:
            logger.warning(f"Spectral rolloff extraction failed: {e}")

        return features
