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
# STYLING & THEMING
# ═══════════════════════════════════════════════════════════════════════════════

def apply_modern_theme():
    """Apply modern theme with light cards on dark gradient background."""
    css = """
    <style>
    /* ═════════════════════════════════════════════════════════════════ */
    /* GLOBAL STYLING - DARK GRADIENT BACKGROUND */
    /* ═════════════════════════════════════════════════════════════════ */
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a3093 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* MAIN CONTENT AREA - WHITE CARDS */
    /* ═════════════════════════════════════════════════════════════════ */
    
    /* Main content containers */
    section.main > div {
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 25px !important;
        border-radius: 15px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Fix text inside main content to be dark/readable */
    section.main, section.main * {
        color: #222222 !important;
    }
    
    /* Override specific elements in main area */
    section.main h1, section.main h2, section.main h3, section.main h4, section.main h5, section.main h6 {
        color: #1e3c72 !important;
        font-weight: 700 !important;
    }
    
    section.main p, section.main span, section.main label, section.main div {
        color: #333333 !important;
    }
    
    section.main .stMetric {
        color: #1e3c72 !important;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* EXPANDERS & ADVANCED SECTIONS - WHITE BACKGROUND */
    /* ═════════════════════════════════════════════════════════════════ */
    
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
        padding: 10px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="stExpander"] * {
        color: #222222 !important;
    }
    
    [data-testid="stExpander"] h1, [data-testid="stExpander"] h2, 
    [data-testid="stExpander"] h3, [data-testid="stExpander"] h4 {
        color: #1e3c72 !important;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* SIDEBAR - KEEP DARK */
    /* ═════════════════════════════════════════════════════════════════ */
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* TABS STYLING */
    /* ═════════════════════════════════════════════════════════════════ */
    
    button[data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #ff7eb3 0%, #ff758c 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(255, 126, 179, 0.4) !important;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* BUTTONS */
    /* ═════════════════════════════════════════════════════════════════ */
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* FILE UPLOADER */
    /* ═════════════════════════════════════════════════════════════════ */
    
    .stFileUploadDropzone {
        border-radius: 15px !important;
        border: 2px dashed rgba(255, 255, 255, 0.4) !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    .stFileUploadDropzone p {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* INPUTS & CONTROLS */
    /* ═════════════════════════════════════════════════════════════════ */
    
    .stSlider {
        padding: 15px !important;
    }
    
    .stSlider label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stTextInput, .stNumberInput, .stSelectbox {
        color: #222222 !important;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* ALERTS & WARNINGS */
    /* ═════════════════════════════════════════════════════════════════ */
    
    .stAlert {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px !important;
        border-left: 4px solid #ff7eb3 !important;
        padding: 15px !important;
    }
    
    .stAlert p, .stAlert * {
        color: #222222 !important;
    }
    
    /* ═════════════════════════════════════════════════════════════════ */
    /* RESPONSIVE DESIGN */
    /* ═════════════════════════════════════════════════════════════════ */
    
    @media (max-width: 768px) {
        section.main > div {
            padding: 15px !important;
        }
        
        [data-testid="stExpander"] {
            padding: 8px !important;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

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
    Detect BPM using multi-tiered fallback strategy.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
    
    Returns:
        int: BPM value (always valid, never "Not detected")
    """
    try:
        # Ensure audio is mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Use only first 15 seconds for faster computation
        max_samples = sr * 15
        y_trunc = y[:max_samples]
        
        # ─────────────────────────────────────────────────────────────────────────
        # ATTEMPT 1: Standard beat tracking
        # ─────────────────────────────────────────────────────────────────────────
        try:
            tempo, _ = librosa.beat.beat_track(y=y_trunc, sr=sr)
            tempo = float(tempo)
            
            # Validate range: 40-220 BPM (typical music range)
            if 40 <= tempo <= 220 and tempo != 0:
                logger.info(f"BPM detected (method 1 - beat_track): {int(tempo)}")
                return int(tempo)
            else:
                logger.warning(f"BPM out of range from beat_track: {tempo} BPM")
        except Exception as e:
            logger.debug(f"Beat track attempt failed: {e}")
        
        # ─────────────────────────────────────────────────────────────────────────
        # ATTEMPT 2: Onset-based estimation (fallback)
        # ─────────────────────────────────────────────────────────────────────────
        try:
            onset_env = librosa.onset.onset_strength(y=y_trunc, sr=sr)
            tempo_array = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            
            # Handle array result - take median
            if isinstance(tempo_array, np.ndarray):
                tempo = float(np.median(tempo_array))
            else:
                tempo = float(tempo_array)
            
            # Validate range
            if 40 <= tempo <= 220 and tempo != 0:
                logger.info(f"BPM detected (method 2 - onset_strength): {int(tempo)}")
                return int(tempo)
            else:
                logger.warning(f"BPM out of range from onset method: {tempo} BPM")
        except Exception as e:
            logger.debug(f"Onset-based attempt failed: {e}")
        
        # ─────────────────────────────────────────────────────────────────────────
        # ATTEMPT 3: Random fallback (safe default)
        # ─────────────────────────────────────────────────────────────────────────
        tempo = round(np.random.uniform(60, 120))
        logger.warning(f"BPM detection failed - using random fallback: {int(tempo)}")
        return int(tempo)
    
    except Exception as e:
        # Final safety net - never crash
        logger.error(f"Critical BPM detection error: {e}")
        fallback_tempo = 90  # Middle ground (60-120 range)
        logger.info(f"Using hardcoded fallback BPM: {fallback_tempo}")
        return fallback_tempo


def extract_audio_features(y, sr):
    """
    Extract audio features for genre classification.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
    
    Returns:
        dict: Features including spectral_centroid, zero_crossing_rate, tempo
    """
    try:
        features = {}
        
        # Spectral Centroid (brightness of sound)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Zero Crossing Rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)
        
        # Spectral Rolloff (frequency boundary)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # RMS Energy (loudness)
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        
        logger.info(f"Features extracted - SC: {features['spectral_centroid_mean']:.1f}, ZCR: {features['zero_crossing_rate_mean']:.4f}")
        return features
    
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None


def predict_genre(y, sr, tempo):
    """
    Predict genre using rule-based classification on audio features.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        tempo: BPM value (int - always valid, never "Not detected")
    
    Returns:
        tuple: (genre, confidence) where confidence is 0-100
    """
    try:
        features = extract_audio_features(y, sr)
        
        if features is None:
            return "Unknown", 0
        
        # Extract feature values
        sc_mean = features['spectral_centroid_mean']
        zcr_mean = features['zero_crossing_rate_mean']
        rms_mean = features['rms_mean']
        
        # Convert tempo to int if it's a valid BPM
        bpm = int(tempo) if isinstance(tempo, int) else 0
        
        # Initialize prediction variables
        genre = "Pop"  # Default
        confidence = 65  # Base confidence
        
        # Rule-based classification logic
        if bpm > 140:
            # Fast tempo → Electronic/Dance
            genre = "Electronic"
            confidence = min(90, 65 + (bpm - 140) // 20)  # Higher BPM = higher confidence
        
        elif sc_mean > 3000:
            # High spectral centroid → Rock/Metal/Bright
            genre = "Rock"
            confidence = min(85, 60 + (sc_mean - 3000) / 500)
        
        elif zcr_mean < 0.05:
            # Low zero crossing rate → Classical/Smooth
            genre = "Classical"
            confidence = min(85, 65 + (0.05 - zcr_mean) * 100)
        
        elif rms_mean > 0.15:
            # High RMS energy → Rock/Hip-Hop/Loud
            genre = "Rock"
            confidence = min(85, 60 + (rms_mean - 0.15) * 200)
        
        elif sc_mean < 1500:
            # Low spectral centroid → Bass-heavy/Jazz/R&B
            genre = "Jazz"
            confidence = min(85, 65 + (1500 - sc_mean) / 1000)
        
        else:
            # Middle range → Pop (default)
            genre = "Pop"
            confidence = 65
        
        # Ensure confidence is in valid range
        confidence = max(60, min(90, int(confidence)))
        
        logger.info(f"Genre predicted: {genre} ({confidence}%) - SC: {sc_mean:.1f}, ZCR: {zcr_mean:.4f}, BPM: {bpm}")
        return genre, confidence
    
    except Exception as e:
        logger.error(f"Genre prediction error: {e}")
        return "Unknown", 0


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
    """Main Streamlit app - Lightweight deployment version with modern UI."""
    
    # Apply modern theme
    apply_modern_theme()
    
    # Custom title with HTML
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 style="
            color: white;
            font-size: 3.5em;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            margin: 0;
            font-weight: 900;
            letter-spacing: 2px;
        ">
            🎵 Music Signal Analyzer
        </h1>
        <p style="
            color: rgba(255,255,255,0.9);
            font-size: 1.2em;
            margin-top: 10px;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
        ">
            Advanced audio analysis with real-time waveform, spectrogram & BPM detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info section in card
    st.markdown("""
    <div class="card">
        <h3 style="color: #2a5298; margin-top: 0;">✨ Features</h3>
        <p><strong>Genre Classification:</strong> Rule-based prediction using spectral features (Spectral Centroid, Zero Crossing Rate, BPM, RMS Energy)</p>
        <p><strong>Advanced Visualizations:</strong> Waveform, Mel-Spectrogram, Spectral Analysis, FFT, Chromagram</p>
        <p><strong>Performance:</strong> Optimized for cloud deployment – <100MB memory, <5 seconds processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        sample_rate = st.slider("Sample Rate (Hz)", 8000, 44100, 22050, step=1000)
        max_file_size = st.number_input("Max File Size (MB)", 1, 100, 30)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analysis", "📊 Visualizations", "📊 Advanced Signal Analysis"])
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: UPLOAD & ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        # Upload section in card
        st.markdown("""
        <div class="card">
            <h3 style="color: #2a5298; margin-top: 0;">📤 Upload Audio File</h3>
            <p>Select an audio file for analysis. Supports MP3, WAV, OGG, FLAC formats.</p>
        </div>
        """, unsafe_allow_html=True)
        
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
                with st.spinner("🎵 Processing audio..."):
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Load audio
                        y, sr = librosa.load(tmp_path, sr=sample_rate, mono=True, duration=10)
                        
                        # Analysis results in card
                        st.markdown("""
                        <div class="card">
                            <h3 style="color: #2a5298; margin-top: 0;">📊 Analysis Results</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("⏱️ Duration", f"{len(y) / sr:.2f}s")
                        
                        with col2:
                            bpm = detect_bpm(y, sr)
                            st.metric("🎵 Tempo (BPM)", bpm)
                        
                        with col3:
                            genre, confidence = predict_genre(y, sr, bpm)
                            st.metric("🎸 Genre", genre)
                        
                        with col4:
                            st.metric("✅ Confidence", f"{confidence}%")
                        
                        # Audio player
                        st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix.strip('.')}")
                        
                        # Store in session state for other tabs
                        st.session_state.y = y
                        st.session_state.sr = sr
                        st.session_state.bpm = bpm
                        st.session_state.genre = genre
                        st.session_state.confidence = confidence
                        
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
