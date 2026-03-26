"""
Step 4 Examples: Beat Detection and BPM Analysis

Demonstrates various beat detection use cases and analysis techniques.
"""

import pandas as pd
import numpy as np
import librosa
from pathlib import Path

from beat_detector import BeatDetector
from beat_pipeline import BeatExtractionPipeline
from config import GENRES_DIR


# ==============================================================================
# EXAMPLE 1: Basic Beat Detection on Single File
# ==============================================================================
def example_1_single_file_beat_detection():
    """Detect beats on a single audio file."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Beat Detection on Single File")
    print("="*80)

    # Find a sample audio file
    genre_dir = GENRES_DIR / "blues"
    if not genre_dir.exists():
        print("✗ GTZAN dataset not found. Please run setup_dataset.py first.")
        return

    audio_files = list(genre_dir.glob("*.wav"))
    if not audio_files:
        print("✗ No audio files found.")
        return

    file_path = str(audio_files[0])
    print(f"\nAnalyzing: {Path(file_path).name}")

    # Initialize detector
    detector = BeatDetector(sr=22050)

    # Extract beats
    result = detector.extract_beat_info(file_path, genre="blues")

    print(f"\nResults:")
    print(f"  Tempo:           {result['tempo_bpm']:.1f} BPM")
    print(f"  Beats detected:  {result['n_beats']}")
    print(f"  Duration:        {result['duration_seconds']:.2f} seconds")
    print(f"  Beat times (sec): {np.array(result['beat_times'])[:5]} ...")

    return result


# ==============================================================================
# EXAMPLE 2: Comparing Beats Across Multiple Files
# ==============================================================================
def example_2_compare_multiple_genres():
    """Compare beat properties across different genres."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Comparing Beats Across Genres")
    print("="*80)

    # Sample files from different genres
    genres = ["blues", "classical", "disco", "metal", "reggae"]
    detector = BeatDetector(sr=22050)

    results = []

    for genre in genres:
        genre_dir = GENRES_DIR / genre
        if not genre_dir.exists():
            continue

        audio_files = list(genre_dir.glob("*.wav"))
        if not audio_files:
            continue

        # Analyze first file of each genre
        file_path = str(audio_files[0])
        result = detector.extract_beat_info(file_path, genre=genre)

        results.append({
            "genre": genre,
            "tempo_bpm": result["tempo_bpm"],
            "n_beats": result["n_beats"],
            "duration": result["duration_seconds"],
            "beat_density": result["n_beats"] / result["duration_seconds"]
        })

        print(f"  {genre:12s}: {result['tempo_bpm']:6.1f} BPM | "
              f"{result['n_beats']:3d} beats | "
              f"Density: {result['n_beats']/result['duration_seconds']:.2f} beats/sec")

    df_comparison = pd.DataFrame(results)
    print("\nComparison DataFrame:")
    print(df_comparison.to_string(index=False))

    return df_comparison


# ==============================================================================
# EXAMPLE 3: Beat Regularity Analysis
# ==============================================================================
def example_3_beat_regularity():
    """Analyze beat regularity (how consistent are beat intervals)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Beat Regularity Analysis")
    print("="*80)

    genre_dir = GENRES_DIR / "disco"  # Usually has very regular beats
    if not genre_dir.exists():
        print("✗ Genre directory not found.")
        return

    audio_files = list(genre_dir.glob("*.wav"))
    if not audio_files:
        print("✗ No audio files found.")
        return

    detector = BeatDetector(sr=22050)
    file_path = str(audio_files[0])

    y, sr = librosa.load(file_path, sr=22050)
    beat_frames, tempo = detector.detect_beats(y, sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Compute inter-beat intervals
    intervals = np.diff(beat_times)

    print(f"\nFile: {Path(file_path).name}")
    print(f"Tempo: {tempo:.1f} BPM")
    print(f"Number of beats: {len(beat_times)}")

    print(f"\nInter-beat Intervals (seconds):")
    print(f"  Mean:          {np.mean(intervals):.4f} s")
    print(f"  Std:           {np.std(intervals):.4f} s")
    print(f"  Min:           {np.min(intervals):.4f} s")
    print(f"  Max:           {np.max(intervals):.4f} s")

    # Regularity metric
    regularity = detector.estimate_beat_regularity(beat_times)
    print(f"\nBeat Regularity (CV):")
    print(f"  Coefficient of Variation: {regularity:.3f}")
    print(f"  (Lower = more regular, <0.1 = very regular)")

    return {
        "intervals": intervals,
        "regularity": regularity,
        "beat_times": beat_times
    }


# ==============================================================================
# EXAMPLE 4: Batch Processing and Statistics
# ==============================================================================
def example_4_batch_processing():
    """Process multiple files and compute statistics."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Processing Complete Dataset")
    print("="*80)

    metadata_path = GENRES_DIR.parent / "dataset_metadata.csv"

    if not metadata_path.exists():
        print("✗ Metadata not found. Run setup_dataset.py first.")
        return

    df_metadata = pd.read_csv(metadata_path)

    # Process only first 50 files for speed
    df_small = df_metadata.head(50)

    print(f"Processing {len(df_small)} files...")

    pipeline = BeatExtractionPipeline(sr=22050)
    df_results = pipeline.process_dataset(df_small, verbose=False)

    # Statistics
    stats = pipeline.compute_statistics()

    print(f"\nResults:")
    print(f"  Successful: {stats['successful']}/{stats['total_files']}")
    print(f"  Mean tempo: {stats['mean_tempo_bpm']:.1f} ± {stats['std_tempo_bpm']:.1f} BPM")
    print(f"  Tempo range: {stats['min_tempo_bpm']:.1f} - {stats['max_tempo_bpm']:.1f} BPM")

    return df_results


# ==============================================================================
# EXAMPLE 5: BPM Distribution Analysis
# ==============================================================================
def example_5_bpm_distribution():
    """Analyze BPM distribution across genres."""
    print("\n" + "="*80)
    print("EXAMPLE 5: BPM Distribution Analysis")
    print("="*80)

    beats_path = GENRES_DIR.parent / "beats_detected.csv"

    if not beats_path.exists():
        print("✗ Beat detection results not found.")
        print("  Run train_beat_detector.py first.")
        return

    df_beats = pd.read_csv(beats_path)

    # Filter successful detections
    df_success = df_beats[df_beats["status"] == "success"]

    print(f"\nAnalyzing {len(df_success)} successful detections:\n")

    for genre in sorted(df_success["genre"].unique()):
        genre_data = df_success[df_success["genre"] == genre]["tempo_bpm"]

        print(f"{genre:12s}:")
        print(f"  Count:   {len(genre_data):3d} files")
        print(f"  Mean:    {genre_data.mean():6.1f} BPM")
        print(f"  Median:  {genre_data.median():6.1f} BPM")
        print(f"  Std:     {genre_data.std():6.1f}")
        print(f"  Range:   {genre_data.min():6.1f} - {genre_data.max():6.1f} BPM")
        print()

    return df_success


# ==============================================================================
# EXAMPLE 6: Onset Strength Analysis
# ==============================================================================
def example_6_onset_strength():
    """Analyze onset strength envelope."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Onset Strength Analysis")
    print("="*80)

    genre_dir = GENRES_DIR / "disco"
    if not genre_dir.exists():
        print("✗ Genre directory not found.")
        return

    audio_files = list(genre_dir.glob("*.wav"))
    if not audio_files:
        print("✗ No audio files found.")
        return

    detector = BeatDetector(sr=22050)
    file_path = str(audio_files[0])

    y, sr = librosa.load(file_path, sr=22050)

    # Compute onset strength
    onset_env = detector.compute_onset_strength(y, sr)

    print(f"\nFile: {Path(file_path).name}")
    print(f"\nOnset Strength Statistics:")
    print(f"  Length: {len(onset_env)} frames (~{len(onset_env)*512/sr:.1f} sec)")
    print(f"  Mean:   {np.mean(onset_env):.4f}")
    print(f"  Max:    {np.max(onset_env):.4f}")
    print(f"  Min:    {np.min(onset_env):.4f}")

    # Find peaks
    peaks = np.where(onset_env > np.mean(onset_env) + 2*np.std(onset_env))[0]
    print(f"  Prominent peaks (>mean+2std): {len(peaks)}")

    return onset_env


# ==============================================================================
# EXAMPLE 7: Visualize Beats on Waveform
# ==============================================================================
def example_7_plot_beats():
    """Plot waveform with detected beats."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Visualize Beats on Waveform")
    print("="*80)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("✗ matplotlib not available. Install with: pip install matplotlib")
        return

    beats_path = GENRES_DIR.parent / "beats_detected.csv"

    if not beats_path.exists():
        print("✗ Beat detection results not found.")
        return

    pipeline = BeatExtractionPipeline(sr=22050)
    pipeline.results_df = pd.read_csv(beats_path)

    # Plot for first successful result
    successful_idx = None
    for idx, row in pipeline.results_df.iterrows():
        if row["status"] == "success":
            successful_idx = idx
            break

    if successful_idx is None:
        print("✗ No successful beat detections found.")
        return

    print(f"\nGenerating waveform plot...")
    fig, ax = pipeline.plot_beat_waveform(file_idx=successful_idx)

    if fig is not None:
        plt.savefig(GENRES_DIR.parent / "beat_waveform_example.png", dpi=100)
        print(f"✓ Plot saved to beat_waveform_example.png")
        plt.close()

    return fig


# ==============================================================================
# MAIN: Run all examples
# ==============================================================================
def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("STEP 4: BEAT DETECTION EXAMPLES")
    print("="*80)

    try:
        example_1_single_file_beat_detection()
        example_2_compare_multiple_genres()
        example_3_beat_regularity()
        example_4_batch_processing()
        example_5_bpm_distribution()
        example_6_onset_strength()
        example_7_plot_beats()

        print("\n" + "="*80)
        print("✓ All examples completed successfully!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
