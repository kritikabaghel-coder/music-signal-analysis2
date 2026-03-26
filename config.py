import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
GENRES_DIR = DATA_DIR / "genres"
LOGS_DIR = PROJECT_ROOT / "logs"

GENRES = [
    "rock", "jazz", "classical", "hiphop", "pop",
    "blues", "country", "disco", "metal", "reggae"
]

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
DEFAULT_SR = None
