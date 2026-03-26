#!/usr/bin/env python3
"""
Quick start guide for Music Genre Classification project
"""

def print_quickstart():
    guide = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    QUICK START GUIDE                                      ║
║   Music Genre Classification + Beat Detection Project                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

STEP 1: Install Dependencies
─────────────────────────────────────────────────────────────────────────────
    pip install -r requirements.txt


STEP 2: Download and Setup GTZAN Dataset
─────────────────────────────────────────────────────────────────────────────
Option A (Automatic):
    python setup_dataset.py

Option B (Manual):
    1. Download: http://marsyasweb.appspot.com/download/genres.tar.gz
    2. Save to:  data/genres.tar.gz
    3. Run:      python setup_dataset.py


STEP 3: Load and Validate Dataset
─────────────────────────────────────────────────────────────────────────────
    python dataset_loader.py

Expected Output:
    - Total number of audio files
    - Number of genres (10)
    - Duration statistics
    - Sample dataset rows
    - Saved metadata to: data/dataset_metadata.csv


STEP 4: Run Examples
─────────────────────────────────────────────────────────────────────────────
    python examples.py

Includes:
    - Basic dataset loading
    - Signal validation
    - Genre statistics
    - Sample rate analysis
    - Metadata export


PROJECT STRUCTURE
─────────────────────────────────────────────────────────────────────────────
│
├── config.py                 ← Configuration and settings
├── dataset_loader.py         ← Main dataset loading class
├── signal_utils.py           ← Signal validation utilities
├── examples.py               ← Usage examples
├── setup_dataset.py          ← Dataset download/extraction
│
├── data/
│   ├── genres/               ← Audio files organized by genre
│   │   ├── rock/
│   │   ├── jazz/
│   │   ├── classical/
│   │   ├── hiphop/
│   │   ├── pop/
│   │   ├── blues/
│   │   ├── country/
│   │   ├── disco/
│   │   ├── metal/
│   │   └── reggae/
│   └── dataset_metadata.csv  ← Generated dataset index
│
└── logs/
    └── dataset_loading.log   ← Detailed loading logs


CORE FEATURES
─────────────────────────────────────────────────────────────────────────────
✓ Automatic GTZAN dataset download
✓ Librosa-based audio signal loading
✓ Robust error handling for corrupted files
✓ Pandas DataFrame with metadata
✓ Duration and sample rate statistics
✓ Production-ready code
✓ Comprehensive logging


GENERATED OUTPUT
─────────────────────────────────────────────────────────────────────────────
CSV Columns:
    - file_path: Full path to audio file
    - genre: Music genre label
    - sample_rate: Sampling frequency (Hz)
    - duration: Audio length (seconds)
    - num_samples: Total number of audio samples


TROUBLESHOOTING
─────────────────────────────────────────────────────────────────────────────
Problem: Download fails
Solution: Download manually from http://marsyasweb.appspot.com/download/genres.tar.gz
          Place in data/ folder and run setup_dataset.py

Problem: Library import errors
Solution: Ensure all packages are installed: pip install -r requirements.txt

Problem: No files found
Solution: Verify genre folders contain .wav files
          Check logs/dataset_loading.log for details


═════════════════════════════════════════════════════════════════════════════
Next: Step 2 - Spectral Analysis & Feature Extraction (Beat Detection)
═════════════════════════════════════════════════════════════════════════════
"""
    print(guide)


if __name__ == "__main__":
    print_quickstart()
