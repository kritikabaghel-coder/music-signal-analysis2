"""
Music Signal Analysis System - Streamlit App

Complete integration of feature extraction, genre classification, BPM detection,
and song recommendation using pre-trained models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import joblib
from scipy.spatial.distance import cosine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PAGE_CONFIG = {
    "page_title": "Music Signal Analysis System",
    "page_icon": "🎵",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

st.set_page_config(**PAGE_CONFIG)

# Cache model loading
@st.cache_resource
def load_models():
    """Load pre-trained models."""
    try:
        model = joblib.load("genre_model.joblib")
        encoder = joblib.load("label_encoder.pkl")
        features_df = pd.read_csv("features.csv")
        return model, encoder, features_df
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (EXACT SAME AS TRAINING - STEP 2)
# ═══════════════════════════════════════════════════════════════════════════════

class AudioFeatureExtractor:
    """Extract same features as training pipeline (Step 2)."""

    def __init__(self, sr=22050):
        self.sr = sr
        self.hop_length = 512

    def compute_fft(self, y):
        """FFT magnitude spectrum features."""
        fft = np.abs(librosa.stft(y, hop_length=self.hop_length))
        S = librosa.power_to_db(fft ** 2, ref=np.max)
        
        features = {
            "fft_mean": np.mean(S),
            "fft_std": np.std(S),
            "fft_max": np.max(S),
            "fft_min": np.min(S),
            "fft_median": np.median(S),
            "fft_sum": np.sum(np.abs(fft))
        }
        return features

    def compute_stft_envelope(self, y):
        """STFT-based envelope features."""
        stft = np.abs(librosa.stft(y, hop_length=self.hop_length))
        S_db = librosa.power_to_db(stft ** 2, ref=np.max)
        
        # Time-domain statistics
        magnitude_env = np.mean(stft, axis=0)
        
        features = {
            "stft_mean": np.mean(S_db),
            "stft_std": np.std(S_db),
            "stft_max": np.max(S_db),
            "stft_energy": np.sum(stft ** 2),
            "envelope_mean": np.mean(magnitude_env),
            "envelope_std": np.std(magnitude_env)
        }
        return features

    def compute_mel_spectrogram(self, y):
        """Mel spectrogram features."""
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Temporal statistics
        temporal_avg = np.mean(mel_db, axis=1)
        
        features = {
            "mel_mean": np.mean(mel_db),
            "mel_std": np.std(mel_db),
            "mel_max": np.max(mel_db),
            "mel_min": np.min(mel_db)
        }
        
        # Mel band statistics
        for i in range(min(13, len(temporal_avg))):
            features[f"mel_band_{i}_avg"] = temporal_avg[i]
        
        return features

    def compute_mfcc(self, y):
        """MFCC features."""
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13, hop_length=self.hop_length)
        
        features = {}
        for i in range(mfcc.shape[0]):
            mfcc_coeff = mfcc[i]
            features[f"mfcc_{i}_mean"] = np.mean(mfcc_coeff)
            features[f"mfcc_{i}_std"] = np.std(mfcc_coeff)
        
        return features

    def compute_spectral_centroid(self, y):
        """Spectral centroid."""
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)[0]
        
        features = {
            "spectral_centroid_mean": np.mean(spec_cent),
            "spectral_centroid_std": np.std(spec_cent),
            "spectral_centroid_max": np.max(spec_cent),
            "spectral_centroid_min": np.min(spec_cent)
        }
        return features

    def compute_spectral_bandwidth(self, y):
        """Spectral bandwidth."""
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, hop_length=self.hop_length)[0]
        
        features = {
            "spectral_bandwidth_mean": np.mean(spec_bw),
            "spectral_bandwidth_std": np.std(spec_bw),
            "spectral_bandwidth_max": np.max(spec_bw),
            "spectral_bandwidth_min": np.min(spec_bw)
        }
        return features

    def compute_zero_crossing_rate(self, y):
        """Zero-crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        
        features = {
            "zcr_mean": np.mean(zcr),
            "zcr_std": np.std(zcr),
            "zcr_max": np.max(zcr),
            "zcr_min": np.min(zcr)
        }
        return features

    def compute_chroma_features(self, y):
        """Chroma features (pitch class distribution)."""
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, hop_length=self.hop_length)
        
        features = {}
        for i in range(chroma.shape[0]):
            features[f"chroma_{i}_mean"] = np.mean(chroma[i])
            features[f"chroma_{i}_std"] = np.std(chroma[i])
        
        return features

    def extract_features(self, y):
        """Extract all features in same order as training."""
        try:
            features = {}
            
            # Extract all feature types
            features.update(self.compute_fft(y))
            features.update(self.compute_stft_envelope(y))
            features.update(self.compute_mel_spectrogram(y))
            features.update(self.compute_mfcc(y))
            features.update(self.compute_spectral_centroid(y))
            features.update(self.compute_spectral_bandwidth(y))
            features.update(self.compute_zero_crossing_rate(y))
            features.update(self.compute_chroma_features(y))
            
            return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio_file(file_path, sr=22050):
    """Load audio file with librosa."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        logger.error(f"Audio loading error: {e}")
        st.error(f"Failed to load audio: {e}")
        return None, None


def extract_features_from_file(file_path, sr=22050):
    """Extract features from audio file."""
    y, sr = load_audio_file(file_path, sr)
    if y is None:
        return None
    
    extractor = AudioFeatureExtractor(sr=sr)
    features = extractor.extract_features(y)
    
    if features is None:
        return None
    
    return features, y, sr


def get_genre_prediction(features_dict, model, encoder):
    """Predict genre with confidence."""
    try:
        # Convert to ordered array matching training
        feature_array = np.array([list(features_dict.values())])
        
        # Predict
        genre_encoded = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        confidence = np.max(probabilities)
        genre = encoder.inverse_transform([genre_encoded])[0]
        
        return genre, confidence
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"Prediction failed: {e}")
        return None, None


def detect_bpm(y, sr):
    """Detect BPM using beat tracking."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception as e:
        logger.error(f"BPM detection error: {e}")
        return None


def find_similar_songs(features_dict, features_df, top_n=5):
    """Find similar songs using cosine similarity."""
    try:
        # Get feature array from input
        input_features = np.array(list(features_dict.values()))
        
        # Get training features (skip file_path and genre columns)
        training_features = features_df.iloc[:, 2:].values
        
        # Compute similarities
        similarities = []
        for i, train_feat in enumerate(training_features):
            sim = 1 - cosine(input_features, train_feat)
            similarities.append(sim)
        
        # Get top N
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            results.append({
                "file_path": features_df.iloc[idx]["file_path"],
                "genre": features_df.iloc[idx]["genre"],
                "similarity": float(similarities[idx])
            })
        
        return results
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
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
    """Main Streamlit app."""
    
    # Title and description
    st.markdown("# 🎵 Music Signal Analysis System")
    st.markdown("Analyze music genre, detect BPM, and find similar songs")
    
    # Load models
    model, encoder, features_df = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        sample_rate = st.slider("Sample Rate (Hz)", 8000, 44100, 22050, step=1000)
        max_file_size = st.number_input("Max File Size (MB)", 1, 100, 30)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analysis", "📊 Visualizations", "🔍 Recommendations"])
    
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
                        # Extract features
                        result = extract_features_from_file(tmp_path, sr=sample_rate)
                        
                        if result is not None:
                            features_dict, y, sr = result
                            
                            # Genre prediction
                            st.subheader("Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                genre, confidence = get_genre_prediction(features_dict, model, encoder)
                                if genre:
                                    st.metric("Predicted Genre", genre)
                                    st.metric("Confidence", f"{confidence*100:.1f}%")
                            
                            with col2:
                                bpm = detect_bpm(y, sr)
                                if bpm:
                                    st.metric("BPM (Tempo)", f"{bpm:.1f}")
                                else:
                                    st.metric("BPM (Tempo)", "N/A")
                            
                            with col3:
                                duration = len(y) / sr
                                st.metric("Duration", f"{duration:.2f}s")
                            
                            # Audio player
                            st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix.strip('.')}")
                            
                            # Store in session state for other tabs
                            st.session_state.features_dict = features_dict
                            st.session_state.y = y
                            st.session_state.sr = sr
                            st.session_state.genre = genre
                            st.session_state.bpm = bpm
                            
                        # Clean up
                        Path(tmp_path).unlink()
                    
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
                        Path(tmp_path).unlink()
    
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
    # TAB 3: RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        if "features_dict" in st.session_state:
            st.subheader("Similar Songs")
            
            with st.spinner("Finding similar songs..."):
                similar = find_similar_songs(st.session_state.features_dict, features_df, top_n=5)
                
                if similar:
                    recommendations_df = pd.DataFrame(similar)
                    recommendations_df["similarity"] = recommendations_df["similarity"].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(
                        recommendations_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "file_path": st.column_config.TextColumn("Song"),
                            "genre": st.column_config.TextColumn("Genre"),
                            "similarity": st.column_config.TextColumn("Similarity Score")
                        }
                    )
                else:
                    st.error("Could not find similar songs")
        else:
            st.info("Upload an audio file first to get recommendations")


if __name__ == "__main__":
    main()
