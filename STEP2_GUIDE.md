# STEP 2: Feature Extraction using Signal Processing Techniques

## Overview

This step extracts frequency-domain and spectral features from audio signals using numpy, librosa, and scipy. The focus is on Signals and Systems concepts: FFT, STFT, spectral analysis, and perceptual audio features.

## Core Modules

### 1. `feature_extractor.py`
Main feature extraction class that computes:

#### Frequency Domain Features
- **FFT**: Fast Fourier Transform using `numpy.fft.fft()`
  - Magnitude spectrum
  - Dominant frequencies (top 5)
  - FFT statistics (mean, std, max)

- **STFT**: Short-Time Fourier Transform using `librosa.stft()`
  - Spectrogram (magnitude)
  - Time-frequency representation
  - STFT statistics

- **Mel Spectrogram**: Perceptual frequency scale using `librosa.feature.melspectrogram()`
  - Log scale conversion (dB)
  - 128 Mel bins
  - Statistics across time

#### Perceptual Audio Features
- **MFCC** (Mel-Frequency Cepstral Coefficients)
  - 13 coefficients (configurable)
  - Time-averaged over signal duration
  - Mimics human hearing

- **Spectral Centroid**: Center of mass of spectrum
  - Mean, std, min, max

- **Spectral Bandwidth**: Width around centroid
  - Spread of spectral energy

- **Spectral Rolloff**: Frequency where 85% of energy is concentrated
  - Mean and std

- **Zero-Crossing Rate (ZCR)**: How often signal crosses zero
  - Linked to tonality (voiced vs unvoiced)
  - Statistics: mean, std, min, max

- **Chroma Features**: Pitch class representation (12 pitches)
  - C, C#, D, D#, E, F, F#, G, G#, A, A#, B
  - Mean and std for each pitch class

### 2. `feature_pipeline.py`
Feature extraction pipeline that:
- Loads audio files from dataset
- Handles errors gracefully (try-except)
- Extracts all features per file
- Creates pandas DataFrame with metadata
- Normalizes features using StandardScaler
- Provides summary statistics and validation

### 3. `advanced_features.py`
Additional advanced features:
- **Spectral Contrast**: Energy contrast across sub-bands
- **Tempogram**: Rhythm-related features
- **RMSE**: Root Mean Square Energy
- **CQT**: Constant-Q Transform (music-friendly frequency scale)
- **Onset Strength**: Beat-related features
- **Energy Statistics**: Detailed energy analysis
- **Signal Statistics**: Basic signal properties
- **Delta Features**: Frame-to-frame changes

### 4. `feature_analysis.py`
Feature analysis and exploration:
- Feature statistics (mean, std, min, max, skew, kurtosis)
- Constant feature detection (low variance)
- Correlated feature detection (highly correlated pairs)
- Outlier detection (IQR and Z-score methods)
- Per-genre comparisons
- Discriminative feature ranking

## Usage

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline
python step2_pipeline.py

# 3. Run examples
python step2_examples.py

# 4. Extract features only
python extract_features.py
```

### Complete Feature Extraction

```python
from dataset_loader import GtganDatasetLoader
from feature_pipeline import FeatureExtractionPipeline

# Load dataset
loader = GtganDatasetLoader()
df_dataset = loader.load_dataset()

# Extract features
pipeline = FeatureExtractionPipeline(
    dataset_df=df_dataset,
    n_mfcc=13,
    n_fft=2048,
    normalize=True
)

# Process dataset
df_features = pipeline.process_dataset()

# Normalize
df_features = pipeline.normalize_features(df_features)

# Summary
pipeline.print_feature_summary(df_features)
```

### Individual Feature Extraction

```python
from feature_extractor import SpectralFeatureExtractor
import librosa

# Load signal
signal, sr = librosa.load('audio.wav')

# Extract features
extractor = SpectralFeatureExtractor(n_mfcc=13, n_fft=2048)
features = extractor.extract_features(signal, sr)

# Result: dictionary with ~80 features
```

### Feature Analysis

```python
from feature_analysis import FeatureAnalyzer

# Get statistics
stats = FeatureAnalyzer.get_feature_statistics(df_features)

# Find constant features
constant = FeatureAnalyzer.identify_constant_features(df_features)

# Find correlated features
correlated = FeatureAnalyzer.identify_correlated_features(df_features)

# Detect outliers
outliers = FeatureAnalyzer.detect_outliers(df_features, method='iqr')

# Print analysis report
FeatureAnalyzer.print_feature_analysis(df_features)
```

## Output Format

### Feature DataFrame Structure

```
file_path | genre | fft_mean | fft_std | fft_max | ... | mfcc_0 | ... | chroma_C_mean | ... |
----------|-------|----------|---------|---------|-----|--------|-----|---------------|-----|
path/to/file.wav | rock | 0.123 | 0.456 | 0.789 | ... | -12.5 | ... | 0.25 | ... |
```

### Feature List (Total: ~80-100 features)

**FFT Features (6)**
- fft_mean, fft_std, fft_max
- dominant_freq_1, dominant_freq_2, dominant_freq_3

**STFT Features (4)**
- stft_mean, stft_std, stft_max, stft_median

**Mel Spectrogram (4)**
- mel_spec_mean, mel_spec_std, mel_spec_max, mel_spec_min

**MFCC (13)**
- mfcc_0 through mfcc_12 (time-averaged)

**Spectral Centroid (4)**
- spectral_centroid_mean, spectral_centroid_std, spectral_centroid_max, spectral_centroid_min

**Spectral Bandwidth (4)**
- spectral_bandwidth_mean, spectral_bandwidth_std, spectral_bandwidth_max, spectral_bandwidth_min

**Zero-Crossing Rate (4)**
- zcr_mean, zcr_std, zcr_max, zcr_min

**Chroma Features (24)**
- chroma_C_mean, chroma_C_std, chroma_C#_mean, ... (12 pitch classes × 2)

**Spectral Rolloff (2)**
- spectral_rolloff_mean, spectral_rolloff_std

**Optional Advanced Features**
- spectral_contrast, tempogram, rmse, cqt, onset_strength, energy_stats

## Technical Details

### FFT Analysis
```
y[n] ---> FFT(n_fft) ---> |magnitude| ---> 
  dominant frequencies, spectral envelope
```

### STFT (Short-Time Fourier Transform)
```
y[n] ---> Window[n] ---> FFT(n_fft) @ each frame ---> 
  spectrogram (time-frequency representation)
```

### Mel Spectrogram
```
spectrogram ---> Mel-filterbank (108 Hz) ---> log scale ---> 
  perceptually-relevant frequency scale
```

### MFCC (Mel-Frequency Cepstral Coefficients)
```
mel_spectrogram ---> DCT (Discrete Cosine Transform) ---> 
  13 coefficients (mimics human ear)
```

### Normalization
- **Method**: StandardScaler (zero mean, unit variance)
- **Applied**: Per-feature normalization across all samples
- **Purpose**: Scale-invariant feature comparison

## Examples

### Example 1: FFT Analysis on Synthetic Signal
```python
# Multi-frequency signal: 440 Hz + 880 Hz + 1320 Hz
# FFT reveals dominant frequencies
# Output: Peaks at expected Hz values
```

### Example 2: STFT Spectrogram (Chirp Signal)
```python
# Frequency sweep from 200 Hz to 2000 Hz over 3 seconds
# STFT shows diagonal pattern in time-frequency plane
# Demonstrates resolution trade-off (time vs frequency)
```

### Example 3: Mel Spectrogram (Musical Chord)
```python
# C Major chord: C4 (261.63 Hz) + E4 (329.63 Hz) + G4 (392 Hz)
# Mel scale emphasizes perceptually relevant frequencies
# Output: 128 time-frequency bins
```

### Example 4: MFCC (Signal Representation)
```python
# 13 MFCCs capture spectral shape
# Time-averaged over signal duration
# Commonly used in speech/music recognition
```

## Error Handling

All feature extraction includes try-except blocks:
- Gracefully skips corrupted files
- Logs errors to file
- Reports statistics on failed files
- Returns complete feature matrix for valid files

## Output Files

### Generated Files
1. **features_extracted.csv**: Complete feature matrix
2. **feature_statistics.csv**: Feature statistics (mean, std, min, max, skew, kurtosis)
3. **features_{genre}.csv**: Per-genre feature statistics
4. **outlier_analysis.csv**: Outlier detection results
5. **dataset_loading.log**: Processing logs

## Performance Notes

- FFT size (n_fft): 2048 samples (default)
- Hop length: 512 samples
- Processing time: ~50-100 ms per 30-second audio file (depending on CPU)
- Memory: ~500 MB for 1000 files
- Feature vector size: ~80-100 features per file

## Constraints

✓ No ML models (just signal processing)
✓ Focus on clean feature extraction
✓ Modular, production-ready code
✓ Error handling throughout
✓ Comprehensive documentation

## Next Steps

After feature extraction:
- Feature selection/dimensionality reduction (Step 3)
- Model training for genre classification (Step 4)
- Beat detection using onset detection (Step 5)
