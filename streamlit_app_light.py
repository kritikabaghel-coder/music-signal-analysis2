"""
🎵 Music Signal Analyzer - Lightweight Streamlit App
Deployment-optimized version for real-time audio analysis.
"""

import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import io
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🎵 Music Signal Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 22050
MAX_DURATION = 5  # Process only first 5 seconds
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB in bytes
FIGSIZE = (14, 6)
DPI = 100

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_session_state():
    """Initialize session state."""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'sr' not in st.session_state:
        st.session_state.sr = None
    if 'filename' not in st.session_state:
        st.session_state.filename = None


def load_audio_file(file_path, sr=22050, max_duration=5):
    """
    Load audio file with duration limit.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate (Hz)
        max_duration: Maximum duration in seconds
    
    Returns:
        tuple: (audio_data, sample_rate) or (None, None) on error
    """
    try:
        # Calculate max samples
        max_samples = int(sr * max_duration)
        
        # Load audio with duration limit
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        
        # Truncate if exceeds max duration
        if len(y) > max_samples:
            y = y[:max_samples]
        
        return y, sr
    except Exception as e:
        st.error(f"❌ Failed to load audio: {str(e)}")
        return None, None


def detect_bpm(y, sr):
    """
    Detect BPM using librosa beat tracking.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        float or str: BPM value or "N/A" on error
    """
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception as e:
        st.warning(f"⚠️ BPM detection failed: {str(e)}")
        return "N/A"


def generate_waveform_plot(y, sr):
    """
    Generate waveform visualization.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        matplotlib figure or None on error
    """
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        
        # Generate time axis
        time_axis = np.arange(len(y)) / sr
        
        # Plot waveform
        ax.plot(time_axis, y, linewidth=0.8, color='steelblue')
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax.set_title('🌊 Waveform (Time Domain)', fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        
        # Add metadata box
        duration = len(y) / sr
        metadata_text = f'Duration: {duration:.2f}s\nSample Rate: {sr} Hz'
        ax.text(0.98, 0.97, metadata_text, 
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"❌ Waveform generation failed: {str(e)}")
        return None


def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """
    Compute spectrogram using STFT.
    
    Args:
        y: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        2D array or None on error
    """
    try:
        # Compute STFT
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Extract magnitude
        magnitude = np.abs(stft)
        
        # Convert to dB
        spectrogram_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        return spectrogram_db
    except Exception as e:
        st.error(f"❌ Spectrogram computation failed: {str(e)}")
        return None


def generate_spectrogram_plot(y, sr, spectrogram_db):
    """
    Generate spectrogram visualization.
    
    Args:
        y: Audio time series
        sr: Sample rate
        spectrogram_db: Computed spectrogram in dB
    
    Returns:
        matplotlib figure or None on error
    """
    try:
        if spectrogram_db is None:
            return None
        
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        
        # Display spectrogram
        img = librosa.display.specshow(spectrogram_db, sr=sr, hop_length=512,
                                       x_axis='time', y_axis='log', ax=ax, cmap='magma')
        
        ax.set_title('🎼 Spectrogram (Frequency Domain)', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Magnitude (dB)', fontsize=10)
        
        # Add metadata box
        duration = len(y) / sr
        freq_bins = spectrogram_db.shape[0]
        frames = spectrogram_db.shape[1]
        metadata_text = f'Duration: {duration:.2f}s\nFreq Bins: {freq_bins}\nFrames: {frames}'
        ax.text(0.98, 0.97, metadata_text, 
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"❌ Spectrogram plotting failed: {str(e)}")
        return None


def process_audio_file(uploaded_file):
    """
    Process uploaded audio file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        dict: Processing results
    """
    try:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File size exceeds 5MB limit. Your file: {uploaded_file.size / (1024*1024):.2f}MB")
            return None
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Load audio
            y, sr = load_audio_file(tmp_path, sr=SAMPLE_RATE, max_duration=MAX_DURATION)
            if y is None or sr is None:
                return None
            
            # Compute spectrogram
            spectrogram_db = compute_spectrogram(y, sr)
            if spectrogram_db is None:
                return None
            
            # Detect BPM
            bpm = detect_bpm(y, sr)
            
            # Generate plots
            waveform_fig = generate_waveform_plot(y, sr)
            spectrogram_fig = generate_spectrogram_plot(y, sr, spectrogram_db)
            
            if waveform_fig is None or spectrogram_fig is None:
                return None
            
            return {
                'y': y,
                'sr': sr,
                'bpm': bpm,
                'waveform_fig': waveform_fig,
                'spectrogram_fig': spectrogram_fig,
                'duration': len(y) / sr,
                'filename': uploaded_file.name
            }
        
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    except Exception as e:
        st.error(f"❌ Unexpected error processing audio: {str(e)}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("# 🎵 Music Signal Analyzer")
    st.markdown("**Real-time audio analysis with waveform, spectrogram, and tempo detection**")
    st.divider()
    
    # Sidebar info
    st.sidebar.title("📋 About")
    st.sidebar.info(
        """
        **🎵 Music Signal Analyzer**
        
        Lightweight audio analysis tool for:
        • 🌊 Waveform visualization
        • 🎼 Spectrogram analysis
        • ⏱️ Tempo (BPM) detection
        
        **Specifications:**
        • Max file size: 5 MB
        • Processing length: First 5 seconds
        • Formats: MP3, WAV, and more
        • Sample rate: 22,050 Hz
        """
    )
    
    st.sidebar.divider()
    st.sidebar.info(
        """
        **⚡ Performance:**
        • Processing time: <5 seconds
        • Memory efficient
        • Deployment ready
        """
    )
    
    # Main content - Tabs layout
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Visualizations", "📈 Results"])
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: UPLOAD
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("📤 Upload Audio File")
        st.write("Upload an audio file to analyze. Supported formats: MP3, WAV, OGG, FLAC, and more.")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "ogg", "flac", "m4a"],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
            
            # Process button
            if st.button("🔍 Analyze Audio", use_container_width=True, type="primary"):
                with st.spinner("Processing audio... This may take a few seconds..."):
                    result = process_audio_file(uploaded_file)
                    
                    if result is not None:
                        st.session_state.processed = True
                        st.session_state.result = result
                        st.success("✅ Audio processed successfully!")
                        st.info(f"⏱️ Processed first {MAX_DURATION} seconds of audio")
                    else:
                        st.error("❌ Failed to process audio file")
                        st.session_state.processed = False
        else:
            st.info("👆 Upload an audio file to get started")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("📊 Signal Visualizations")
        
        if st.session_state.processed and 'result' in st.session_state:
            result = st.session_state.result
            
            # Waveform
            st.markdown("### 🌊 Waveform (Time Domain)")
            st.markdown("Amplitude over time - shows the raw audio signal")
            st.pyplot(result['waveform_fig'], use_container_width=True)
            
            st.divider()
            
            # Spectrogram
            st.markdown("### 🎼 Spectrogram (Frequency Domain)")
            st.markdown("Frequency content over time - shows which frequencies are present")
            st.markdown("- **Bright areas:** High energy")
            st.markdown("- **Dark areas:** Low energy")
            st.markdown("- **Color intensity:** Magnitude strength in dB")
            st.pyplot(result['spectrogram_fig'], use_container_width=True)
        else:
            st.info("👈 Upload and process audio in the Upload tab to see visualizations")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("📈 Analysis Results")
        
        if st.session_state.processed and 'result' in st.session_state:
            result = st.session_state.result
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📁 File Information")
                st.write(f"**Filename:** {result['filename']}")
                st.write(f"**Duration (processed):** {result['duration']:.2f} seconds")
                st.write(f"**Sample Rate:** {result['sr']} Hz")
            
            with col2:
                st.markdown("### ⏱️ Tempo Analysis")
                if result['bpm'] != 'N/A':
                    st.metric(label="Detected BPM", value=f"{result['bpm']:.1f}")
                    
                    # BPM interpretation
                    bpm_val = float(result['bpm'])
                    if bpm_val < 80:
                        tempo_desc = "🎶 Slow (Ballads, Classical)"
                    elif bpm_val < 100:
                        tempo_desc = "🎵 Moderate (Blues, Rock)"
                    elif bpm_val < 130:
                        tempo_desc = "🎸 Fast (Pop, Hip-hop, Disco)"
                    else:
                        tempo_desc = "⚡ Very Fast (Metal, Electronic)"
                    
                    st.write(f"**Tempo Category:** {tempo_desc}")
                else:
                    st.warning("⚠️ BPM detection failed for this audio")
            
            st.divider()
            
            # Spectrogram details
            st.markdown("### 🎼 Spectrogram Details")
            col1, col2, col3 = st.columns(3)
            
            freq_bins = result['spectrogram_fig'].get_axes()[0].get_yticks().shape[0] if result['spectrogram_fig'] else 1025
            
            with col1:
                st.metric(label="FFT Size", value="2048")
            with col2:
                st.metric(label="Hop Length", value="512")
            with col3:
                st.metric(label="Frequency Bins", value="1025")
            
            st.divider()
            
            # Export options
            st.markdown("### 💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export waveform
                buf_waveform = io.BytesIO()
                result['waveform_fig'].savefig(buf_waveform, format='png', dpi=DPI, bbox_inches='tight')
                buf_waveform.seek(0)
                st.download_button(
                    label="📥 Download Waveform (PNG)",
                    data=buf_waveform,
                    file_name="waveform.png",
                    mime="image/png"
                )
            
            with col2:
                # Export spectrogram
                buf_spectrogram = io.BytesIO()
                result['spectrogram_fig'].savefig(buf_spectrogram, format='png', dpi=DPI, bbox_inches='tight')
                buf_spectrogram.seek(0)
                st.download_button(
                    label="📥 Download Spectrogram (PNG)",
                    data=buf_spectrogram,
                    file_name="spectrogram.png",
                    mime="image/png"
                )
        else:
            st.info("👈 Upload and process audio in the Upload tab to see results")


if __name__ == "__main__":
    main()
