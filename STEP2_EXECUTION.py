"""
STEP 2: FEATURE EXTRACTION - EXECUTION GUIDE
"""

STEP_2_MODULES = {
    "Core Extraction": [
        "feature_extractor.py      - SpectralFeatureExtractor class",
        "feature_pipeline.py       - FeatureExtractionPipeline (orchestration)",
        "extract_features.py       - Main execution script",
    ],
    "Advanced Analysis": [
        "advanced_features.py      - Additional signal processing features",
        "feature_analysis.py       - Feature analysis and exploration",
    ],
    "Examples & Testing": [
        "step2_examples.py         - 8 comprehensive examples",
        "step2_pipeline.py         - Complete pipeline with reporting",
    ],
    "Documentation": [
        "STEP2_GUIDE.md           - Technical documentation",
    ]
}

FEATURE_CATEGORIES = {
    "FFT Features": [
        "fft_mean, fft_std, fft_max",
        "dominant_freq_1, dominant_freq_2, dominant_freq_3"
    ],
    "STFT Features": [
        "stft_mean, stft_std, stft_max, stft_median"
    ],
    "Mel Spectrogram": [
        "mel_spec_mean, mel_spec_std, mel_spec_max, mel_spec_min"
    ],
    "MFCC (13 coefficients)": [
        "mfcc_0, mfcc_1, ..., mfcc_12"
    ],
    "Spectral Centroid": [
        "spectral_centroid_mean, spectral_centroid_std",
        "spectral_centroid_max, spectral_centroid_min"
    ],
    "Spectral Bandwidth": [
        "spectral_bandwidth_mean, spectral_bandwidth_std",
        "spectral_bandwidth_max, spectral_bandwidth_min"
    ],
    "Zero-Crossing Rate": [
        "zcr_mean, zcr_std, zcr_max, zcr_min"
    ],
    "Chroma Features (12 pitch classes)": [
        "chroma_C_mean, chroma_C_std",
        "chroma_C#_mean, chroma_C#_std, ..., chroma_B_mean, chroma_B_std"
    ],
    "Spectral Rolloff": [
        "spectral_rolloff_mean, spectral_rolloff_std"
    ]
}

EXECUTION_STEPS = """
╔════════════════════════════════════════════════════════════════════════════╗
║                      STEP 2: EXECUTION GUIDE                              ║
╚════════════════════════════════════════════════════════════════════════════╝

PREREQUISITE:
─────────────
✓ Step 1 complete: GTZAN dataset downloaded and loaded
✓ Output: data/dataset_metadata.csv with file paths


QUICK START (3 OPTIONS):
──────────────────────────────────────────────────────────────────────────────

OPTION 1: Full Pipeline (Recommended)
    python step2_pipeline.py
    
    - Loads dataset
    - Extracts all spectral features
    - Normalizes features
    - Analyzes features
    - Generates report
    - Saves: features_extracted.csv


OPTION 2: Examples Only (Learning)
    python step2_examples.py
    
    - 8 standalone examples
    - FFT, STFT, Mel spectrogram, MFCC
    - Spectral features, ZCR, Chroma
    - Complete pipeline demo


OPTION 3: Custom Extraction
    python extract_features.py
    
    - Basic feature extraction
    - Minimal normalization
    - Quick processing


DETAILED EXECUTION:
──────────────────────────────────────────────────────────────────────────────

1. VERIFY SETUP
   python verify_setup.py
   ✓ Checks Python version
   ✓ Verifies all dependencies installed
   ✓ Checks directory structure


2. LOAD DATASET
   from dataset_loader import GtganDatasetLoader
   loader = GtganDatasetLoader()
   df = loader.load_dataset()
   
   Output: DataFrame with 1000 audio files (10 genres × 100)


3. CREATE PIPELINE
   from feature_pipeline import FeatureExtractionPipeline
   
   pipeline = FeatureExtractionPipeline(
       dataset_df=df,
       n_mfcc=13,
       n_fft=2048,
       normalize=True
   )


4. EXTRACT FEATURES
   df_features = pipeline.process_dataset()
   
   - Processes each audio file individually
   - Extracts ~80-100 features per file
   - Handles errors gracefully
   - ~ 50-100 ms per 30-second file


5. NORMALIZE FEATURES
   df_features = pipeline.normalize_features(df_features)
   
   - StandardScaler normalization
   - Zero mean, unit variance
   - Applied per feature


6. ANALYZE RESULTS
   pipeline.print_feature_summary(df_features)
   
   Output:
   - Feature matrix shape: (1000, 82)
   - Files per genre: 100 each
   - Missing values: 0
   - Feature statistics


OUTPUT FILES:
──────────────────────────────────────────────────────────────────────────────
data/
├── features_extracted.csv           [1000 files × 82 features]
├── feature_statistics.csv           [Statistics per feature]
├── features_rock.csv                [Per-genre statistics]
├── features_jazz.csv
├── ... (10 genres total)
└── outlier_analysis.csv             [Outlier detection results]


FEATURE MATRIX STRUCTURE:
──────────────────────────────────────────────────────────────────────────────
Rows:    1000 audio files
Columns: 82 total
  - 80 spectral/perceptual features
  - 1 file_path
  - 1 genre label

Example row:
file_path: data/genres/rock/rock.00001.au
genre: rock
fft_mean: 0.1234
fft_std: 0.5678
... (78 more feature columns)


FEATURE DESCRIPTIONS:
──────────────────────────────────────────────────────────────────────────────

FFT (Fast Fourier Transform):
  - Decomposes signal into frequency components
  - fft_mean, fft_std, fft_max: Statistics of magnitude spectrum
  - dominant_freq_X: Top 5 frequencies by magnitude

STFT (Short-Time Fourier Transform):
  - Time-frequency representation
  - Shows how frequency content changes over time
  - Used to create spectrograms

Mel Spectrogram:
  - Frequency scale based on human hearing
  - Emphasizes lower frequencies more than higher
  - Converted to dB (decibel) scale

MFCC (Mel-Frequency Cepstral Coefficients):
  - 13 coefficients capture spectral shape
  - Mimics human auditory system
  - Widely used in speech/music recognition

Spectral Centroid:
  - Center of mass of the spectrum
  - Higher centroid = brighter tone
  - Lower centroid = darker tone

Spectral Bandwidth:
  - Width of spectrum around centroid
  - High bandwidth = noisy signal
  - Low bandwidth = tonal signal

Spectral Rolloff:
  - Frequency where 85% of energy is concentrated
  - High rolloff = more high-frequency content

Zero-Crossing Rate (ZCR):
  - How often signal crosses zero
  - High ZCR = noisy/unvoiced
  - Low ZCR = tonal/voiced

Chroma Features:
  - Pitch class representation (C, C#, D, ..., B)
  - 12 pitch classes total
  - Useful for music analysis


ANALYSIS TOOLS:
──────────────────────────────────────────────────────────────────────────────
from feature_analysis import FeatureAnalyzer, FeatureComparison

# Get feature statistics
stats = FeatureAnalyzer.get_feature_statistics(df_features)

# Find constant features
constant = FeatureAnalyzer.identify_constant_features(df_features)

# Find correlated features
correlated = FeatureAnalyzer.identify_correlated_features(df_features)

# Detect outliers
outliers = FeatureAnalyzer.detect_outliers(df_features)

# Find discriminative features
discriminative = FeatureComparison.find_discriminative_features(df_features)


TROUBLESHOOTING:
──────────────────────────────────────────────────────────────────────────────
Problem: ModuleNotFoundError: No module named 'librosa'
Solution: pip install -r requirements.txt

Problem: Dataset is empty
Solution: Run setup_dataset.py first to download GTZAN dataset

Problem: Feature extraction is slow
Solution: Process subset of files with limit_files parameter:
          pipeline.process_dataset() # reduces batch
          Or use step2_pipeline.py with limit_files=100

Problem: NaN values in features
Solution: Handled by pipeline; logged in dataset_loading.log

Problem: Memory error with large dataset
Solution: Process in batches:
          for i in range(0, len(df), 100):
              batch = df.iloc[i:i+100]
              features_batch = pipeline.process_dataset()


NEXT STEPS:
──────────────────────────────────────────────────────────────────────────────
After Step 2 completion:

Step 3: Feature Selection & Dimensionality Reduction
  - Select discriminative features
  - PCA for dimensionality reduction
  - Visualize feature distributions

Step 4: Model Training
  - Train genre classification model
  - Validate with cross-validation
  - Evaluate performance

Step 5: Beat Detection
  - Onset detection
  - Tempo estimation
  - Beat tracking


═════════════════════════════════════════════════════════════════════════════
For detailed technical documentation, see: STEP2_GUIDE.md
═════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(EXECUTION_STEPS)
    
    print("\n\nMODULE STRUCTURE:")
    for category, modules in STEP_2_MODULES.items():
        print(f"\n{category}:")
        for module in modules:
            print(f"  - {module}")
    
    print("\n\nFEATURE CATEGORIES ({} total features):".format(
        sum(len(v) for v in FEATURE_CATEGORIES.values())
    ))
    for category, features in FEATURE_CATEGORIES.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  - {feature}")
