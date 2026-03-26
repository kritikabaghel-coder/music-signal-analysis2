"""
Waveform Visualization Generator

Generates waveform plots for sample audio files (1 per genre).
Uses librosa for audio loading and matplotlib for visualization.
"""

import librosa
import matplotlib.pyplot as plt
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
        logging.FileHandler('logs/waveform_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 22050
OUTPUT_DIR = Path("outputs/waveforms")
METADATA_FILE = Path("data/dataset_metadata.csv")
DPI = 100  # Resolution for saved images
FIGSIZE = (14, 6)  # Figure size (width, height)

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# WAVEFORM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio_file(file_path, sr=22050):
    """
    Load audio file using librosa.
    
    Args:
        file_path (str or Path): Path to audio file
        sr (int): Sample rate for resampling (Hz)
    
    Returns:
        tuple: (audio_data, sample_rate) or (None, None) on error
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
        logger.info(f"Loaded audio: {file_path} (duration: {len(y)/sr:.2f}s)")
        return y, sr
    except Exception as e:
        logger.error(f"Failed to load audio: {file_path} - {str(e)}")
        return None, None


def generate_waveform_plot(y, sr, file_path, output_path):
    """
    Generate and save waveform plot.
    
    Args:
        y (np.ndarray): Audio time series data
        sr (int): Sample rate (Hz)
        file_path (str): Original audio file path (for naming)
        output_path (Path): Where to save the plot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        
        # Generate time axis (in seconds)
        time_axis = np.arange(len(y)) / sr
        
        # Plot waveform
        ax.plot(time_axis, y, linewidth=0.5, color='steelblue')
        
        # Set labels and title
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
        
        # Extract filename for title
        filename = Path(file_path).stem
        genre = Path(file_path).parent.name
        ax.set_title(f'Waveform: {genre.upper()} - {filename}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add duration info
        duration = len(y) / sr
        ax.text(0.98, 0.95, f'Duration: {duration:.2f}s | SR: {sr} Hz',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved waveform: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate waveform plot: {str(e)}")
        return False


def process_sample_per_genre(metadata_file):
    """
    Process one sample audio file from each genre.
    
    Args:
        metadata_file (Path): Path to dataset_metadata.csv
    
    Returns:
        dict: Statistics of processed files
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
        
        # Statistics
        total_processed = 0
        successful = 0
        failed = 0
        genre_samples = {}
        
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
            
            # Load audio
            y, sr = load_audio_file(file_path, sr=SAMPLE_RATE)
            if y is None:
                failed += 1
                genre_samples[genre] = {'status': 'FAILED', 'reason': 'Load error'}
                continue
            
            # Generate output filename
            filename_stem = Path(file_path).stem
            output_filename = f"{genre}_{filename_stem}.png"
            output_path = OUTPUT_DIR / output_filename
            
            # Generate waveform plot
            if generate_waveform_plot(y, sr, file_path, output_path):
                successful += 1
                total_processed += 1
                genre_samples[genre] = {
                    'status': 'SUCCESS',
                    'file': file_path,
                    'output': str(output_path),
                    'duration': len(y) / sr
                }
                logger.info(f"✓ {genre}: {output_filename}")
            else:
                failed += 1
                genre_samples[genre] = {'status': 'FAILED', 'reason': 'Plot generation error'}
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("WAVEFORM GENERATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Genres: {len(genres)}")
        logger.info(f"Successfully Processed: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Output Directory: {OUTPUT_DIR.absolute()}")
        logger.info("="*80)
        
        return {
            'total_genres': len(genres),
            'successful': successful,
            'failed': failed,
            'output_dir': str(OUTPUT_DIR.absolute()),
            'genre_samples': genre_samples
        }
        
    except Exception as e:
        logger.error(f"Error processing samples: {str(e)}")
        return None


def print_summary(stats):
    """Print summary in console-friendly format."""
    if stats is None:
        print("❌ No statistics available")
        return
    
    print("\n" + "="*80)
    print("🎵 WAVEFORM VISUALIZATION SUMMARY")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   • Total Genres Processed: {stats['total_genres']}")
    print(f"   • Successful: {stats['successful']}")
    print(f"   • Failed: {stats['failed']}")
    print(f"\n📁 Output Directory:")
    print(f"   {stats['output_dir']}")
    
    print(f"\n🎼 Per-Genre Results:")
    for genre, result in stats['genre_samples'].items():
        status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"   {status_icon} {genre.upper()}: {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"      Duration: {result['duration']:.2f}s")
            print(f"      Output: {Path(result['output']).name}")
    
    print("\n" + "="*80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting waveform visualization generation...")
    logger.info(f"Sample Rate: {SAMPLE_RATE} Hz")
    logger.info(f"Output Directory: {OUTPUT_DIR.absolute()}")
    
    # Process samples
    stats = process_sample_per_genre(METADATA_FILE)
    
    # Print summary
    print_summary(stats)
    
    logger.info("Waveform generation complete!")
