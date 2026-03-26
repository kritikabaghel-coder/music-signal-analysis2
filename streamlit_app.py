"""
Music Signal Analysis System - Streamlit App

Lightweight deployment version with waveform, spectrogram, and BPM detection.
No external model files required.
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PAGE_CONFIG = {
    "page_title": "🎵 Music Signal Analyzer",
    "page_icon": "🎵",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

st.set_page_config(**PAGE_CONFIG)

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio_file(file_path, sr=22050):
    """Load audio file with librosa."""
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        logger.info(f"Loaded audio: duration={len(y)/sr:.2f}s")
        return y, sr
    except Exception as e:
        logger.error(f"Audio loading error: {e}")
        st.error(f"❌ Failed to load audio: {str(e)}")
        return None, None
    
    if features is None:
        return None
    
    return features, y, sr


def detect_bpm(y, sr):
    """Detect BPM using beat tracking."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception as e:
        logger.error(f"BPM detection error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_waveform(y, sr):
    """Plot audio waveform."""
    fig, ax = plt.subplots(figsize=(12, 4))
    time_axis = np.arange(len(y)) / sr
    ax.plot(time_axis, y, color="steelblue", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(y, sr):
    """Plot mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, hop_length=512)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main Streamlit app - Lightweight deployment version."""
    
    # Title and description
    st.markdown("# 🎵 Music Signal Analysis System")
    st.markdown("Lightweight demo: Analyze waveforms, spectrograms, and detect BPM")
    
    # Info box - Genre classification not available in demo
    st.info("ℹ️ **Genre Classification:** Not available in this deployment version. Use ML model version for genre prediction.")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        sample_rate = st.slider("Sample Rate (Hz)", 8000, 44100, 22050, step=1000)
        max_file_size = st.number_input("Max File Size (MB)", 1, 100, 30)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📤 Upload & Analysis", "📊 Visualizations"])
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: UPLOAD & ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "ogg", "flac"],
            help="Supported formats: MP3, WAV, OGG, FLAC"
        )
        
        if uploaded_file is not None:
            # Check file size
            if uploaded_file.size > max_file_size * 1024 * 1024:
                st.error(f"File too large. Max size: {max_file_size}MB")
            else:
                with st.spinner("Processing audio..."):
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Load audio
                        y, sr = librosa.load(tmp_path, sr=sample_rate, mono=True, duration=10)
                        
                        # Analysis results
                        st.subheader("Analysis Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Duration", f"{len(y) / sr:.2f}s")
                        
                        with col2:
                            bpm = detect_bpm(y, sr)
                            if bpm:
                                st.metric("BPM (Tempo)", f"{bpm:.1f}")
                            else:
                                st.metric("BPM (Tempo)", "N/A")
                        
                        with col3:
                            st.metric("Genre", "Unknown")
                        
                        with col4:
                            st.metric("Confidence", "N/A")
                        
                        # Audio player
                        st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix.strip('.')}")
                        
                        # Store in session state for other tab
                        st.session_state.y = y
                        st.session_state.sr = sr
                        st.session_state.bpm = bpm
                        
                        # Clean up
                        Path(tmp_path).unlink()
                    
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
                        logger.error(f"File processing error: {e}")
                        try:
                            Path(tmp_path).unlink()
                        except:
                            pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        if "y" in st.session_state:
            st.subheader("Audio Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.pyplot(plot_waveform(st.session_state.y, st.session_state.sr))
            
            with col2:
                st.pyplot(plot_mel_spectrogram(st.session_state.y, st.session_state.sr))
        else:
            st.info("Upload an audio file first to see visualizations")


if __name__ == "__main__":
    main()
