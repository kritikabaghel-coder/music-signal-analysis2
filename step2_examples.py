"""
Step 2 Examples: Feature Extraction from Audio Signals

Demonstrates:
- FFT computation and magnitude spectrum extraction
- STFT and spectrogram analysis
- Mel spectrogram features
- MFCC extraction
- Spectral features (centroid, bandwidth, rolloff)
- Zero-crossing rate
- Chroma features
- Feature normalization
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path

from dataset_loader import GtganDatasetLoader
from feature_pipeline import FeatureExtractionPipeline
from feature_extractor import SpectralFeatureExtractor
from advanced_features import AdvancedFeatures, TimeFrequencyAnalysis
from config import GENRES_DIR


def example_1_fft_analysis():
    """Example: Compute FFT and analyze frequency content."""
    print("\n" + "="*80)
    print("EXAMPLE 1: FFT Analysis")
    print("="*80)

    # Create synthetic signal
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Mix of frequencies
    signal = (np.sin(2 * np.pi * 440 * t) +  # A4
              0.5 * np.sin(2 * np.pi * 880 * t) +  # A5
              0.3 * np.sin(2 * np.pi * 1320 * t))  # E6

    # Compute FFT
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    frequency = np.fft.fftfreq(len(signal), 1/sr)

    # Positive frequencies only
    positive_idx = frequency >= 0
    magnitude_positive = magnitude[positive_idx]
    frequency_positive = frequency[positive_idx]

    # Find peaks
    top_indices = np.argsort(magnitude_positive)[-3:][::-1]
    top_freqs = frequency_positive[top_indices]
    top_mags = magnitude_positive[top_indices]

    print(f"\nSignal composed of 3 sine waves:")
    print(f"  - 440 Hz (A4)")
    print(f"  - 880 Hz (A5)")
    print(f"  - 1320 Hz (E6)\n")

    print("FFT Analysis Results:")
    for i, (freq, mag) in enumerate(zip(top_freqs, top_mags), 1):
        print(f"  Peak {i}: {freq:7.1f} Hz (magnitude: {mag:8.1f})")

    print(f"\nFFT Statistics:")
    print(f"  Mean magnitude: {np.mean(magnitude_positive):.4f}")
    print(f"  Std magnitude:  {np.std(magnitude_positive):.4f}")
    print(f"  Max magnitude:  {np.max(magnitude_positive):.4f}")


def example_2_stft_spectrogram():
    """Example: Compute STFT and visualize spectrogram."""
    print("\n" + "="*80)
    print("EXAMPLE 2: STFT and Spectrogram Analysis")
    print("="*80)

    # Create chirp signal (frequency sweep)
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    f_start, f_end = 200, 2000

    # Linear frequency sweep
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
    signal = np.sin(phase)

    # Compute STFT
    n_fft = 2048
    stft_matrix = librosa.stft(signal, n_fft=n_fft)
    magnitude_spec = np.abs(stft_matrix)

    # Convert to dB scale
    magnitude_db = librosa.amplitude_to_db(magnitude_spec, ref=np.max)

    print(f"\nChirp Signal (frequency sweep from 200 Hz to 2000 Hz):")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration} s")
    print(f"  FFT size: {n_fft}\n")

    print(f"STFT Spectrogram shape: {magnitude_spec.shape}")
    print(f"  Frequency bins: {magnitude_spec.shape[0]}")
    print(f"  Time frames: {magnitude_spec.shape[1]}\n")

    print(f"Magnitude Statistics (linear scale):")
    print(f"  Mean: {np.mean(magnitude_spec):.4f}")
    print(f"  Std:  {np.std(magnitude_spec):.4f}")
    print(f"  Max:  {np.max(magnitude_spec):.4f}\n")

    print(f"Magnitude Statistics (dB scale):")
    print(f"  Mean: {np.mean(magnitude_db):.4f} dB")
    print(f"  Std:  {np.std(magnitude_db):.4f} dB")
    print(f"  Max:  {np.max(magnitude_db):.4f} dB")


def example_3_mel_spectrogram():
    """Example: Compute Mel spectrogram and convert to log scale."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Mel Spectrogram (Music Scale)")
    print("="*80)

    # Create synthetic signal
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Musical chord
    signal = (np.sin(2 * np.pi * 261.63 * t) +  # C4
              np.sin(2 * np.pi * 329.63 * t) +  # E4
              np.sin(2 * np.pi * 392.00 * t))   # G4

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    print(f"\nMusical Chord (C Major):")
    print(f"  C4 (261.63 Hz)")
    print(f"  E4 (329.63 Hz)")
    print(f"  G4 (392.00 Hz)\n")

    print(f"Mel Spectrogram shape: {mel_spec.shape}")
    print(f"  Mel bins: {mel_spec.shape[0]}")
    print(f"  Time frames: {mel_spec.shape[1]}\n")

    print(f"Mel Spectrogram Statistics (linear scale):")
    print(f"  Mean: {np.mean(mel_spec):.6f}")
    print(f"  Std:  {np.std(mel_spec):.6f}")
    print(f"  Max:  {np.max(mel_spec):.6f}\n")

    print(f"Mel Spectrogram Statistics (dB scale):")
    print(f"  Mean: {np.mean(mel_spec_db):.4f} dB")
    print(f"  Std:  {np.std(mel_spec_db):.4f} dB")
    print(f"  Min:  {np.min(mel_spec_db):.4f} dB")
    print(f"  Max:  {np.max(mel_spec_db):.4f} dB")


def example_4_mfcc_extraction():
    """Example: Extract MFCCs from audio signal."""
    print("\n" + "="*80)
    print("EXAMPLE 4: MFCC Extraction")
    print("="*80)

    # Create synthetic signal
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t) * (1 + 0.1 * np.sin(2 * np.pi * 5 * t))

    # Compute MFCCs
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=2048)

    print(f"\nMFCC Extraction (Mel-Frequency Cepstral Coefficients):")
    print(f"  Number of MFCCs: {n_mfcc}")
    print(f"  Sample rate: {sr} Hz\n")

    print(f"MFCC Matrix shape: {mfcc.shape}")
    print(f"  MFCC coefficients: {mfcc.shape[0]}")
    print(f"  Time frames: {mfcc.shape[1]}\n")

    # Average MFCCs over time
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    print(f"Time-Averaged MFCC Features:")
    for i, (mean, std) in enumerate(zip(mfcc_mean, mfcc_std)):
        print(f"  MFCC_{i:2d}: mean={mean:8.4f}, std={std:8.4f}")


def example_5_spectral_features():
    """Example: Extract spectral features (centroid, bandwidth, rolloff)."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Spectral Features")
    print("="*80)

    # Create synthetic signal with varying spectrum
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Low-frequency signal
    signal = np.sin(2 * np.pi * 200 * t)

    extractor = SpectralFeatureExtractor()

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    print(f"\nSpectral Centroid:")
    print(f"  Mean: {np.mean(spectral_centroid):.2f} Hz")
    print(f"  Std:  {np.std(spectral_centroid):.2f} Hz")
    print(f"  (Center of mass of the spectrum)")

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    print(f"\nSpectral Bandwidth:")
    print(f"  Mean: {np.mean(spectral_bandwidth):.2f} Hz")
    print(f"  Std:  {np.std(spectral_bandwidth):.2f} Hz")
    print(f"  (Width of the spectrum around centroid)")

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    print(f"\nSpectral Rolloff:")
    print(f"  Mean: {np.mean(spectral_rolloff):.2f} Hz")
    print(f"  Std:  {np.std(spectral_rolloff):.2f} Hz")
    print(f"  (Frequency below which 85% of energy is concentrated)")


def example_6_zero_crossing_rate():
    """Example: Compute zero-crossing rate."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Zero-Crossing Rate (ZCR)")
    print("="*80)

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Low frequency
    signal_low = np.sin(2 * np.pi * 100 * t)
    zcr_low = librosa.feature.zero_crossing_rate(signal_low)

    # High frequency
    signal_high = np.sin(2 * np.pi * 2000 * t)
    zcr_high = librosa.feature.zero_crossing_rate(signal_high)

    print(f"\nZero-Crossing Rate Analysis:")
    print(f"  (Measures how often signal crosses zero)\n")

    print(f"Low Frequency (100 Hz):")
    print(f"  Mean ZCR: {np.mean(zcr_low):.4f}")
    print(f"  Expected: ~{2*100/sr:.4f} (2*freq/sr)\n")

    print(f"High Frequency (2000 Hz):")
    print(f"  Mean ZCR: {np.mean(zcr_high):.4f}")
    print(f"  Expected: ~{2*2000/sr:.4f} (2*freq/sr)")


def example_7_chroma_features():
    """Example: Extract chroma features."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Chroma Features (Pitch Classes)")
    print("="*80)

    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # C major chord
    signal = (np.sin(2 * np.pi * 261.63 * t) +  # C4
              np.sin(2 * np.pi * 329.63 * t) +  # E4
              np.sin(2 * np.pi * 392.00 * t))   # G4

    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    print(f"\nChroma Features (12 Pitch Classes):")
    print(f"  C Major chord (C, E, G)\n")

    for name, value in zip(pitch_classes, chroma_mean):
        bar_length = int(value * 50)
        bar = "█" * bar_length
        print(f"  {name:3s}: {value:.4f} {bar}")


def example_8_feature_pipeline():
    """Example: Complete feature extraction pipeline."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Complete Feature Extraction Pipeline")
    print("="*80)

    # Load dataset
    print("\n[1/3] Loading dataset...")
    loader = GtganDatasetLoader(genres_dir=GENRES_DIR)
    df_dataset = loader.load_dataset()

    if len(df_dataset) == 0:
        print("Dataset is empty. Skipping example.")
        return

    print(f"Loaded {len(df_dataset)} files\n")

    # Extract features
    print("[2/3] Extracting features (first 5 files for demo)...")
    pipeline = FeatureExtractionPipeline(
        dataset_df=df_dataset.head(5),
        normalize=False
    )

    df_features = pipeline.process_dataset()

    # Normalize
    print("[3/3] Normalizing features...\n")
    df_features_normalized = pipeline.normalize_features(df_features)

    # Print summary
    print(f"Feature matrix shape: {df_features_normalized.shape}")
    print(f"Features extracted: {df_features_normalized.shape[1] - 2}")  # -2 for path and genre
    print(f"\nSample rows:")
    print(df_features_normalized.head())


def main():
    print("\n" + "="*80)
    print("STEP 2: FEATURE EXTRACTION EXAMPLES")
    print("="*80)

    examples = [
        ("FFT Analysis", example_1_fft_analysis),
        ("STFT and Spectrogram", example_2_stft_spectrogram),
        ("Mel Spectrogram", example_3_mel_spectrogram),
        ("MFCC Extraction", example_4_mfcc_extraction),
        ("Spectral Features", example_5_spectral_features),
        ("Zero-Crossing Rate", example_6_zero_crossing_rate),
        ("Chroma Features", example_7_chroma_features),
        ("Complete Pipeline", example_8_feature_pipeline),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example {i} ({name}) failed: {e}")

    print("\n" + "="*80)
    print("✓ All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
