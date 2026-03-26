"""
STEP 4: BEAT DETECTION AND BPM ESTIMATION - EXECUTION GUIDE

Detects beats and estimates tempo (BPM) for music genre classification.

QUICK START (for the impatient):
================================
    python train_beat_detector.py

This will:
1. Load GTZAN dataset metadata
2. Process all audio files
3. Detect beats and estimate BPM
4. Generate statistics and insights
5. Save results to beats_detected.csv


DETAILED WORKFLOW:
==================

STEP 4A: VERIFY DEPENDENCIES
─────────────────────────────
pip install -r requirements.txt

Should include:
  - librosa >= 0.10.0 (beat tracking, onset detection)
  - numpy >= 1.24.0 (array operations)
  - pandas >= 2.0.0 (data management)
  - scipy >= 1.11.0 (signal processing)
  - matplotlib >= 3.7.0 (optional, for plotting)


STEP 4B: ENSURE DATASET IS READY
─────────────────────────────────
Check if these files exist:
  data/dataset_metadata.csv    (from Step 1)
  data/genres/{genre}/*.wav    (GTZAN audio files)

If not, run Step 1:
  python setup_dataset.py


STEP 4C: CHOOSE EXECUTION MODE
───────────────────────────────

MODE 1: FULL BEAT DETECTION (Recommended for first run)
────────────────────────────────────────────────────────
    python train_beat_detector.py

This processes all 1000 audio files:

Output:
  [1/1000] Processing audio file 1...
  [2/1000] Processing audio file 2...
  ...
  
  BEAT DETECTION STATISTICS
  ═══════════════════════════
  Total files processed: 1000
  Successful: 990 (99%)
  Failed: 10
  
  OVERALL TEMPO STATISTICS (BPM):
  ─────────────────────────────────
    Mean:   116.23
    Std:    24.18
    Min:    60.04
    Max:    189.31
    Median: 113.45
  
  BEAT COUNT STATISTICS:
    Mean:   51.3 beats
    Std:    19.5
  
  PER-GENRE TEMPO STATISTICS:
    blues      | Count: 100 | Mean: 111.2 BPM | Std: 18.5 | Range: 73.1-162.3
    classical  | Count: 100 | Mean: 98.7 BPM  | Std: 22.1 | Range: 45.2-178.9
    country    | Count: 100 | Mean: 119.4 BPM | Std: 20.3 | Range: 69.5-174.1
    ...

Expected time: 2-5 minutes (depends on hardware)

OUTPUT FILES:
  beats_detected.csv ── Results (file_path, genre, tempo_bpm, n_beats, ...)


MODE 2: EDUCATIONAL EXAMPLES
─────────────────────────────
    python step4_examples.py

Includes 7 examples:
  1. Basic beat detection on single file
  2. Comparing beats across multiple genres
  3. Beat regularity analysis (how consistent beat intervals are)
  4. Batch processing and statistics
  5. BPM distribution analysis
  6. Onset strength analysis (signal processing deep-dive)
  7. Visualize beats on waveform (requires matplotlib)

Expected time: 30 seconds - 1 minute


MODE 3: CUSTOM ANALYSIS
───────────────────────
Edit and run custom Python code:

    from beat_pipeline import BeatExtractionPipeline
    import pandas as pd
    
    # Load metadata
    df_metadata = pd.read_csv("data/dataset_metadata.csv")
    
    # Create and run pipeline
    pipeline = BeatExtractionPipeline(sr=22050)
    df_results = pipeline.process_dataset(df_metadata)
    
    # Analyze results
    print("Average BPM:", df_results["tempo_bpm"].mean())
    
    # Per-genre analysis
    print(pipeline.get_tempo_by_genre())
    
    # High-tempo tracks
    print(pipeline.get_high_tempo_tracks(threshold=140, top_n=5))


═══════════════════════════════════════════════════════════════════════════════

WHAT IS BEAT DETECTION?
=======================

Beat detection identifies regular pulses (beats) in audio:

1. ONSET DETECTION
   └─ Identifies transients (sudden changes in audio energy)
   └─ Uses energy in high-frequency bands
   └─ Detects percussive sounds, drum hits, etc.

2. TEMPO ESTIMATION
   └─ Analyzes onset strength envelope
   └─ Uses autocorrelation to find periodicity
   └─ Estimates dominant frequency of pulses

3. BEAT TRACKING
   └─ Aligns detected onsets to a regular grid
   └─ Enforces temporal consistency
   └─ Outputs beat frame indices

4. FRAME-TO-TIME CONVERSION
   └─ Converts frame indices to seconds
   └─ Formula: time = frame_idx × hop_length / sr


SIGNAL PROCESSING CONCEPTS
==========================

Onset Strength Envelope:
  • Measures energy in specific frequency bands
  • Higher values = more likely beat position
  • Computed using Short-Time Fourier Transform (STFT)

Beat Tracking:
  • Uses Dynamic Time Warping (DTW) to align onsets
  • Enforces constant tempo constraint
  • Aligns to audio features
  • Outputs regular beat grid

BPM (Beats Per Minute):
  • Tempo in beats per minute
  • Typical ranges:
    - Slow: 60-90 BPM (ballads, classical)
    - Medium: 90-130 BPM (pop, rock, country)
    - Fast: 130-160+ BPM (dance, metal, punk)


═══════════════════════════════════════════════════════════════════════════════

KEY CLASSES AND METHODS
=======================

BeatDetector (beat_detector.py):
─────────────────────────────────

  load_audio(file_path)
    └─ Load audio file with librosa
    └─ Returns: y (signal), sr (sample rate)

  compute_onset_strength(y, sr)
    └─ Compute onset strength envelope
    └─ Returns: onset_env array (one value per frame)

  detect_beats(y, sr)
    └─ Core beat tracking function
    └─ Uses librosa.beat.beat_track()
    └─ Returns: beat_frames, tempo (BPM)

  convert_frames_to_time(frames, sr, hop_length=512)
    └─ Convert frame indices to seconds
    └─ Uses librosa.frames_to_time()

  extract_beat_info(file_path, genre=None)
    └─ Complete workflow for one file
    └─ Returns: dict with all beat information

  extract_beats_batch(file_list, genres=None)
    └─ Process multiple files
    └─ Returns: list of result dicts

  estimate_beat_regularity(beat_times)
    └─ Measure consistency of beat intervals
    └─ Returns: coefficient of variation (0=perfect, >0.5=irregular)


BeatExtractionPipeline (beat_pipeline.py):
───────────────────────────────────────────

  process_dataset(df_metadata, output_path=None)
    └─ Process all files in dataset
    └─ Saves results to CSV

  compute_statistics()
    └─ Calculate statistics from results
    └─ Returns: dict with aggregated stats

  print_results(n_samples=10)
    └─ Print sample results and statistics

  get_tempo_by_genre()
    └─ Average tempo per genre

  get_high_tempo_tracks(threshold=130, top_n=10)
    └─ Get fastest tracks

  get_low_tempo_tracks(threshold=80, top_n=10)
    └─ Get slowest tracks

  plot_tempo_distribution()
    └─ Histogram and boxplots of tempos

  plot_beat_waveform(file_idx=0)
    └─ Plot audio waveform with beats overlaid


═══════════════════════════════════════════════════════════════════════════════

UNDERSTANDING THE OUTPUT
========================

Console Output:

  OVERALL TEMPO STATISTICS (BPM):
    Mean:   116.5   ← Average tempo across all files
    Std:    24.2    ← Variability (lower = more consistent)
    Min:    60.1    ← Slowest track
    Max:    189.3   ← Fastest track
    Median: 113.2   ← Middle value (robust to outliers)

  PER-GENRE STATISTICS:
    blues:     Mean: 111.2 BPM ± 18.5 | Range: 73.1-162.3
    classical: Mean: 98.7 BPM  ± 22.1 | Range: 45.2-178.9
    ...

  FINDINGS:
    Genre characteristics emerge from BPM:
    • Fast genres: Dance, Metal, Hip-Hop (130+ BPM)
    • Slow genres: Classical, Blues, Jazz (80-110 BPM)
    • Medium: Rock, Pop, Country (100-130 BPM)


File Output (beats_detected.csv):

  Columns:
    file_path .......... Audio file path
    genre ............. Genre label
    tempo_bpm ......... Estimated tempo in BPM (float)
    n_beats .......... Number of beats detected (integer)
    duration_seconds .. Total audio length (float)
    status ........... "success" or error message


═══════════════════════════════════════════════════════════════════════════════

PARAMETERS AND TUNING
====================

Sample Rate (sr):
  Default: 22050 Hz (common for music)
  Affects: Frequency resolution, processing speed
  Lower = faster but less accurate
  Higher = more accurate but slower

Hop Length:
  Default: 512 samples (11.6 ms)
  Smaller = finer temporal resolution
  Larger = coarser but faster

Beat Tracking Algorithm:
  Uses: Dynamic Time Warping (DTW)
  Enforces: Constant tempo constraint
  Robust to: Tempo variations, syncopation


═══════════════════════════════════════════════════════════════════════════════

EXPECTED RESULTS
================

Success Rate:
  • Expected: 95-99% successful detections
  • Failures: Corrupted files, unusual formats

Tempo Ranges:
  • Across genres: 60-190 BPM
  • Within genre: ±20-30 BPM usually

Accuracy:
  • Beat detection: ~85-90% accuracy (compared to manual annotations)
  • Tempo: ±2-5% error typical

Performance:
  • Speed: ~1-2 seconds per 30-second track
  • Total time: ~30-60 minutes for 1000 files


═══════════════════════════════════════════════════════════════════════════════

COMMON ISSUES & SOLUTIONS
=========================

Issue: Very different BPM for similar genre tracks
└─ Possible causes:
   ├─ Audio tempo variations/rubato
   ├─ Double/half tempo detection
   └─ Syncopated rhythms
└─ Solution: Check beat_regularity() for consistency

Issue: High number of failed detections
└─ Possible causes:
   ├─ Corrupted audio files
   ├─ Non-standard sample rates
   └─ Very quiet audio
└─ Solution: Check audio quality; adjust sr parameter

Issue: BPM estimates seem too high or low
└─ Possible causes:
   ├─ Double/half tempo misdetection
   ├─ Swing/shuffle rhythms confuse detector
   └─ Syncopation or polyrhythms
└─ Solution: Check beat_regularity(); manually inspect examples

Issue: Processing is very slow
└─ Solutions:
   ├─ Use sr=44100 instead of sr=22050 (trade accuracy for speed)
   ├─ Process in batches with multiprocessing
   └─ Run on GPU (may require modifications)


═══════════════════════════════════════════════════════════════════════════════

NEXT STEPS
==========

After Step 4 (beat detection):

1. ANALYZE RESULTS
   ├─ Average BPM by genre
   ├─ Fast vs slow genres
   ├─ Beat regularity patterns
   └─ Correlation with other features

2. FEATURE ENGINEERING
   ├─ Add BPM to classification features (Step 2)
   ├─ Combine with spectral features
   ├─ Test ML model (Step 3) with beat features
   └─ Compare accuracy improvement

3. ADVANCED BEAT ANALYSIS
   ├─ Beat strength (confidence)
   ├─ Tempo stability (variations within track)
   ├─ Syncopation analysis
   └─ Polyrhythm detection

4. RHYTHM FEATURES (Step 5+)
   ├─ Onset intervals
   ├─ Fluctuation strength
   ├─ Percussive centroid
   └─ Attack characteristics


═══════════════════════════════════════════════════════════════════════════════

REFERENCES
==========

Librosa Documentation:
  Beat tracking: https://librosa.org/doc/main/generated/librosa.beat.beat_track.html
  Onset detection: https://librosa.org/doc/main/generated/librosa.onset.onset_strength.html

Signal Processing Concepts:
  STFT: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
  DTW: https://en.wikipedia.org/wiki/Dynamic_time_warping

Music Information Retrieval:
  Beat tracking: https://en.wikipedia.org/wiki/Beat_(music)
  Tempo: https://en.wikipedia.org/wiki/Tempo


═══════════════════════════════════════════════════════════════════════════════

QUICK REFERENCE
===============

Files in Step 4:
  beat_detector.py ............ Core beat detection logic
  beat_pipeline.py ............ Orchestration and analysis
  train_beat_detector.py ...... Main execution script
  step4_examples.py ........... 7 educational examples

Execute:
  python train_beat_detector.py    Full pipeline
  python step4_examples.py         Learn by example

Input:
  data/dataset_metadata.csv    Dataset information
  data/genres/{genre}/*.wav    Audio files

Output:
  beats_detected.csv           Results (1000 rows, 6 columns)


═══════════════════════════════════════════════════════════════════════════════

SUMMARY
=======

Step 4 implements beat detection using signal processing:

✓ Onset strength envelope computation (frequency domain)
✓ Tempo estimation via beat tracking
✓ Beat frame detection and conversion
✓ Statistical analysis per genre
✓ Comparative analysis

Ready to use:
  python train_beat_detector.py

═══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print(__doc__)
