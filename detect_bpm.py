"""
BPM (Tempo) Detection Script

Extracts tempo (BPM) from audio files using librosa beat tracking.
Processes one sample per genre and saves results to a CSV file.
"""

import librosa
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bpm_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 22050
OUTPUT_DIR = Path("outputs")
METADATA_FILE = Path("data/dataset_metadata.csv")
OUTPUT_CSV = OUTPUT_DIR / "bpm_results.csv"
MAX_DURATION = 10  # Limit audio duration to 10 seconds for safety

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# BPM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio_file(file_path, sr=22050, max_duration=10):
    """
    Load audio file using librosa with duration limit.
    
    Args:
        file_path (str or Path): Path to audio file
        sr (int): Sample rate for resampling (Hz)
        max_duration (float): Maximum duration in seconds
    
    Returns:
        tuple: (audio_data, sample_rate) or (None, None) on error
    """
    try:
        # Calculate maximum number of samples
        max_samples = int(sr * max_duration)
        
        # Load audio with duration limit
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        
        # Truncate if exceeds max duration
        if len(y) > max_samples:
            y = y[:max_samples]
            logger.info(f"Truncated audio to {max_duration}s: {file_path}")
        
        logger.info(f"Loaded audio: {file_path} (duration: {len(y)/sr:.2f}s)")
        return y, sr
    except Exception as e:
        logger.error(f"Failed to load audio: {file_path} - {str(e)}")
        return None, None


def detect_bpm(y, sr):
    """
    Detect BPM (tempo) from audio signal.
    
    Args:
        y (np.ndarray): Audio time series data
        sr (int): Sample rate (Hz)
    
    Returns:
        float or str: BPM value or "N/A" on error
    """
    try:
        # Estimate tempo using beat tracking
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Ensure tempo is a valid number
        bpm = float(tempo)
        logger.info(f"Detected BPM: {bpm:.1f}")
        return bpm
    
    except Exception as e:
        logger.warning(f"BPM detection failed: {str(e)}")
        return "N/A"


def extract_bpm_for_file(file_path, genre, sr=22050, max_duration=10):
    """
    Extract BPM for a single audio file.
    
    Args:
        file_path (str or Path): Path to audio file
        genre (str): Genre label
        sr (int): Sample rate (Hz)
        max_duration (float): Maximum duration in seconds
    
    Returns:
        dict: Result dictionary with file_name, genre, bpm
    """
    try:
        # Load audio
        y, sr_actual = load_audio_file(file_path, sr=sr, max_duration=max_duration)
        if y is None:
            logger.error(f"Failed to load audio: {file_path}")
            return {
                'file_name': Path(file_path).name,
                'genre': genre,
                'bpm': 'N/A'
            }
        
        # Detect BPM
        bpm = detect_bpm(y, sr_actual)
        
        result = {
            'file_name': Path(file_path).name,
            'genre': genre,
            'bpm': bpm
        }
        
        logger.info(f"✓ {genre}/{Path(file_path).name}: {bpm} BPM")
        return result
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {
            'file_name': Path(file_path).name,
            'genre': genre,
            'bpm': 'N/A'
        }


def process_sample_per_genre(metadata_file, sr=22050, max_duration=10):
    """
    Process one sample audio file from each genre.
    
    Args:
        metadata_file (Path): Path to dataset_metadata.csv
        sr (int): Sample rate (Hz)
        max_duration (float): Maximum duration in seconds
    
    Returns:
        dict: Statistics and results
    """
    try:
        # Load metadata
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return None
        
        df = pd.read_csv(metadata_file)
        logger.info(f"Loaded metadata with {len(df)} files")
        
        # Get unique genres
        genres = df['genre'].unique()
        logger.info(f"Found {len(genres)} genres: {sorted(genres)}")
        
        # Results list
        results = []
        total_processed = 0
        successful = 0
        failed = 0
        
        # Process one sample per genre
        for genre in sorted(genres):
            logger.info(f"\nProcessing genre: {genre}")
            
            # Get first file from this genre
            genre_df = df[df['genre'] == genre]
            if len(genre_df) == 0:
                logger.warning(f"No files found for genre: {genre}")
                continue
            
            # Get first file
            first_file = genre_df.iloc[0]
            file_path = first_file['file_path']
            
            logger.info(f"Processing: {file_path}")
            
            # Extract BPM
            result = extract_bpm_for_file(file_path, genre, sr=sr, max_duration=max_duration)
            results.append(result)
            total_processed += 1
            
            # Count successes/failures
            if result['bpm'] != 'N/A':
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("BPM DETECTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Genres: {len(genres)}")
        logger.info(f"Successfully Processed: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Output File: {OUTPUT_CSV.absolute()}")
        logger.info("="*80)
        
        return {
            'total_genres': len(genres),
            'successful': successful,
            'failed': failed,
            'results': results,
            'output_file': str(OUTPUT_CSV.absolute())
        }
        
    except Exception as e:
        logger.error(f"Error processing samples: {str(e)}")
        return None


def save_results_to_csv(results, output_file):
    """
    Save results to CSV file.
    
    Args:
        results (list): List of result dictionaries
        output_file (Path): Path to output CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to: {output_file}")
        logger.info(f"Total rows: {len(df)}")
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        return False


def print_summary(stats):
    """Print summary in console-friendly format."""
    if stats is None:
        print("❌ No statistics available")
        return
    
    print("\n" + "="*80)
    print("🎵 BPM DETECTION SUMMARY")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   • Total Genres Processed: {stats['total_genres']}")
    print(f"   • Successful: {stats['successful']}")
    print(f"   • Failed: {stats['failed']}")
    
    print(f"\n📁 Output File:")
    print(f"   {stats['output_file']}")
    
    print(f"\n🎼 BPM Results (per genre):")
    for result in stats['results']:
        bpm_display = f"{result['bpm']:.1f}" if result['bpm'] != 'N/A' else result['bpm']
        print(f"   • {result['genre'].upper()}: {bpm_display} BPM ({result['file_name']})")
    
    # Calculate and display statistics
    bpm_values = [r['bpm'] for r in stats['results'] if r['bpm'] != 'N/A']
    if bpm_values:
        print(f"\n📈 BPM Statistics:")
        print(f"   • Mean: {np.mean(bpm_values):.1f} BPM")
        print(f"   • Min: {np.min(bpm_values):.1f} BPM")
        print(f"   • Max: {np.max(bpm_values):.1f} BPM")
        print(f"   • Std Dev: {np.std(bpm_values):.1f} BPM")
    
    print("\n" + "="*80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting BPM detection...")
    logger.info(f"Sample Rate: {SAMPLE_RATE} Hz")
    logger.info(f"Max Duration: {MAX_DURATION} seconds")
    logger.info(f"Output File: {OUTPUT_CSV.absolute()}")
    
    # Process samples
    stats = process_sample_per_genre(METADATA_FILE, sr=SAMPLE_RATE, max_duration=MAX_DURATION)
    
    if stats and stats['results']:
        # Save to CSV
        save_results_to_csv(stats['results'], OUTPUT_CSV)
        
        # Print summary
        print_summary(stats)
    else:
        logger.error("No results to save")
        print("❌ BPM detection failed - no results")
    
    logger.info("BPM detection complete!")
