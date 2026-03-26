"""
Spectrogram Visualization Generator

Generates spectrogram plots for sample audio files (1 per genre).
Uses librosa for audio loading and STFT computation.
Uses matplotlib with librosa.display for visualization.
"""

import librosa
import librosa.display
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
        logging.FileHandler('logs/spectrogram_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 22050
OUTPUT_DIR = Path("outputs/spectrograms")
METADATA_FILE = Path("data/dataset_metadata.csv")
DPI = 100  # Resolution for saved images
FIGSIZE = (14, 8)  # Figure size (width, height)
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Number of samples between successive frames

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SPECTROGRAM GENERATION
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


def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """
    Compute STFT-based spectrogram in decibel scale.
    
    Args:
        y (np.ndarray): Audio time series data
        sr (int): Sample rate (Hz)
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
    
    Returns:
        np.ndarray: Spectrogram in dB scale (magnitude)
    """
    try:
        # Compute STFT (Short-Time Fourier Transform)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Convert to magnitude
        magnitude = np.abs(stft)
        
        # Convert to decibel scale
        spectrogram_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        logger.info(f"Computed spectrogram: shape {spectrogram_db.shape}")
        return spectrogram_db
    
    except Exception as e:
        logger.error(f"Failed to compute spectrogram: {str(e)}")
        return None


def generate_spectrogram_plot(y, sr, spectrogram_db, file_path, output_path, 
                             n_fft=2048, hop_length=512):
    """
    Generate and save spectrogram plot.
    
    Args:
        y (np.ndarray): Audio time series data
        sr (int): Sample rate (Hz)
        spectrogram_db (np.ndarray): Spectrogram in dB scale
        file_path (str): Original audio file path (for naming)
        output_path (Path): Where to save the plot
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        
        # Plot spectrogram using librosa.display.specshow
        img = librosa.display.specshow(
            spectrogram_db,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='log',  # Log frequency scale for better visualization
            ax=ax,
            cmap='magma'  # Color map
        )
        
        # Add colorbar
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Magnitude (dB)', fontsize=11, fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        
        # Extract filename for title
        filename = Path(file_path).stem
        genre = Path(file_path).parent.name
        ax.set_title(f'Spectrogram: {genre.upper()} - {filename}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add info box
        duration = len(y) / sr
        ax.text(0.98, 0.98, f'Duration: {duration:.2f}s | SR: {sr} Hz | FFT: {n_fft}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved spectrogram: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate spectrogram plot: {str(e)}")
        return False


def process_sample_per_genre(metadata_file, n_fft=2048, hop_length=512):
    """
    Process one sample audio file from each genre.
    
    Args:
        metadata_file (Path): Path to dataset_metadata.csv
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
    
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
            
            # Compute spectrogram
            spectrogram_db = compute_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length)
            if spectrogram_db is None:
                failed += 1
                genre_samples[genre] = {'status': 'FAILED', 'reason': 'Spectrogram computation error'}
                continue
            
            # Generate output filename
            filename_stem = Path(file_path).stem
            output_filename = f"{genre}_{filename_stem}.png"
            output_path = OUTPUT_DIR / output_filename
            
            # Generate spectrogram plot
            if generate_spectrogram_plot(y, sr, spectrogram_db, file_path, output_path, 
                                        n_fft=n_fft, hop_length=hop_length):
                successful += 1
                total_processed += 1
                genre_samples[genre] = {
                    'status': 'SUCCESS',
                    'file': file_path,
                    'output': str(output_path),
                    'duration': len(y) / sr,
                    'n_frequencies': spectrogram_db.shape[0],
                    'n_frames': spectrogram_db.shape[1]
                }
                logger.info(f"✓ {genre}: {output_filename}")
            else:
                failed += 1
                genre_samples[genre] = {'status': 'FAILED', 'reason': 'Plot generation error'}
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("SPECTROGRAM GENERATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Genres: {len(genres)}")
        logger.info(f"Successfully Processed: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Output Directory: {OUTPUT_DIR.absolute()}")
        logger.info(f"FFT Window Size: {n_fft}")
        logger.info(f"Hop Length: {hop_length}")
        logger.info("="*80)
        
        return {
            'total_genres': len(genres),
            'successful': successful,
            'failed': failed,
            'output_dir': str(OUTPUT_DIR.absolute()),
            'n_fft': n_fft,
            'hop_length': hop_length,
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
    print("🎵 SPECTROGRAM VISUALIZATION SUMMARY")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   • Total Genres Processed: {stats['total_genres']}")
    print(f"   • Successful: {stats['successful']}")
    print(f"   • Failed: {stats['failed']}")
    print(f"\n⚙️  Parameters:")
    print(f"   • FFT Window Size: {stats['n_fft']}")
    print(f"   • Hop Length: {stats['hop_length']}")
    
    print(f"\n📁 Output Directory:")
    print(f"   {stats['output_dir']}")
    
    print(f"\n🎼 Per-Genre Results:")
    for genre, result in stats['genre_samples'].items():
        status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"   {status_icon} {genre.upper()}: {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"      Duration: {result['duration']:.2f}s")
            print(f"      Frequencies: {result['n_frequencies']} | Frames: {result['n_frames']}")
            print(f"      Output: {Path(result['output']).name}")
    
    print("\n" + "="*80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting spectrogram visualization generation...")
    logger.info(f"Sample Rate: {SAMPLE_RATE} Hz")
    logger.info(f"Output Directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"FFT Window: {N_FFT} | Hop Length: {HOP_LENGTH}")
    
    # Process samples
    stats = process_sample_per_genre(METADATA_FILE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Print summary
    print_summary(stats)
    
    logger.info("Spectrogram generation complete!")
