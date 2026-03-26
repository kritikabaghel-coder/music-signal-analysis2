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
        y, sr = librosa.load(file_path, sr=sr, mono=True, duration=10)
        logger.info(f"Loaded audio: duration={len(y)/sr:.2f}s")
        return y, sr
    except Exception as e:
        logger.error(f"Audio loading error: {e}")
        st.error(f"❌ Failed to load audio: {str(e)}")
        return None, None


def detect_bpm(y, sr):
    """
    Detect BPM using beat tracking with librosa.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
    
    Returns:
        int: BPM value or "Not detected" if detection fails
    """
    try:
        # Ensure audio is mono (should already be from load_audio_file)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Use only first 15 seconds for faster computation
        max_samples = sr * 15  # 15 seconds
        y_trunc = y[:max_samples]
        
        # Detect tempo using librosa beat tracking
        tempo, _ = librosa.beat.beat_track(y=y_trunc, sr=sr)
        
        # Convert to integer and ensure valid range
        bpm = int(tempo)
        if bpm < 0 or bpm > 300:  # Sanity check
            logger.warning(f"BPM out of range: {bpm} (expected 0-300)")
            return "Not detected"
        
        logger.info(f"BPM detected: {bpm}")
        return bpm
    
    except Exception as e:
        logger.error(f"BPM detection error: {e}")
        return "Not detected"


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


def plot_spectral_centroid(y, sr):
    """Plot spectral centroid over time (brightness of sound)."""
    # Compute spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroid))
    t = librosa.frames_to_time(frames, sr=sr)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, spectral_centroid, color="#FF6B6B", linewidth=2)
    ax.fill_between(t, spectral_centroid, alpha=0.2, color="#FF6B6B")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spectral Centroid (Hz)")
    ax.set_title("Spectral Centroid Over Time (Brightness)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectral_bandwidth(y, sr):
    """Plot spectral bandwidth over time."""
    # Compute spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    frames = range(len(spectral_bandwidth))
    t = librosa.frames_to_time(frames, sr=sr)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, spectral_bandwidth, color="#4ECDC4", linewidth=2)
    ax.fill_between(t, spectral_bandwidth, alpha=0.2, color="#4ECDC4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spectral Bandwidth (Hz)")
    ax.set_title("Spectral Bandwidth Over Time")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_rms_energy(y, sr):
    """Plot RMS energy over time (signal energy)."""
    # Compute RMS energy
    rms_energy = librosa.feature.rms(y=y)[0]
    frames = range(len(rms_energy))
    t = librosa.frames_to_time(frames, sr=sr)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, rms_energy, color="#95E1D3", linewidth=2)
    ax.fill_between(t, rms_energy, alpha=0.2, color="#95E1D3")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title("RMS Energy Over Time (Signal Energy)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_fft(y, sr):
    """Plot frequency distribution using FFT."""
    # Compute FFT
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequencies = np.fft.fftfreq(len(y), 1/sr)
    
    # Take only positive frequencies
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(frequencies[:len(frequencies)//4], magnitude[:len(magnitude)//4], color="#A8E6CF", linewidth=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (log scale)")
    ax.set_title("Frequency Distribution (FFT)")
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim([0, sr//2])
    plt.tight_layout()
    return fig


def plot_chroma_features(y, sr):
    """Plot chromagram (musical content)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax, hop_length=512)
    fig.colorbar(img, ax=ax, format="%+2.0f")
    ax.set_title("Chromagram (Musical Notes)")
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
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analysis", "📊 Visualizations", "📊 Advanced Signal Analysis"])
    
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
                            if isinstance(bpm, int) and bpm > 0:
                                st.metric("Tempo (BPM)", bpm)
                            else:
                                st.metric("Tempo (BPM)", bpm if isinstance(bpm, str) else "N/A")
                        
                        with col3:
                            st.metric("Genre", "Unknown")
                        
                        with col4:
                            st.metric("Confidence", "N/A")
                        
                        # Audio player
                        st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix.strip('.')}")
                        
                        # Store in session state for other tabs
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: ADVANCED SIGNAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        if "y" in st.session_state:
            st.subheader("📊 Advanced Signal Analysis")
            
            # Create expandable sections for performance optimization
            with st.expander("🔴 Spectral Centroid (Brightness)", expanded=True):
                try:
                    st.pyplot(plot_spectral_centroid(st.session_state.y, st.session_state.sr))
                    st.caption("Shows the brightness of the sound over time. Higher values = brighter sound.")
                except Exception as e:
                    st.error(f"Error computing spectral centroid: {e}")
                    logger.error(f"Spectral centroid error: {e}")
            
            with st.expander("🟢 Spectral Bandwidth", expanded=True):
                try:
                    st.pyplot(plot_spectral_bandwidth(st.session_state.y, st.session_state.sr))
                    st.caption("Shows the width of frequencies present in the signal.")
                except Exception as e:
                    st.error(f"Error computing spectral bandwidth: {e}")
                    logger.error(f"Spectral bandwidth error: {e}")
            
            with st.expander("🔵 RMS Energy (Signal Energy)", expanded=True):
                try:
                    st.pyplot(plot_rms_energy(st.session_state.y, st.session_state.sr))
                    st.caption("Shows the energy level (loudness) of the audio over time.")
                except Exception as e:
                    st.error(f"Error computing RMS energy: {e}")
                    logger.error(f"RMS energy error: {e}")
            
            with st.expander("🟡 Frequency Distribution (FFT)", expanded=False):
                try:
                    st.pyplot(plot_fft(st.session_state.y, st.session_state.sr))
                    st.caption("Shows which frequencies are present in the signal (computed using Fast Fourier Transform).")
                except Exception as e:
                    st.error(f"Error computing FFT: {e}")
                    logger.error(f"FFT error: {e}")
            
            with st.expander("⚫ Chromagram (Musical Notes)", expanded=False):
                try:
                    st.pyplot(plot_chroma_features(st.session_state.y, st.session_state.sr))
                    st.caption("Shows the intensity of 12 musical note classes over time.")
                except Exception as e:
                    st.error(f"Error computing chromagram: {e}")
                    logger.error(f"Chromagram error: {e}")
        else:
            st.info("Upload an audio file first to see advanced signal analysis")


if __name__ == "__main__":
    main()
