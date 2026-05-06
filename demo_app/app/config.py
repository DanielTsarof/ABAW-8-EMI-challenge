from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    device: str = "cuda"

    audio_encoder_dir: Path
    audio_head_pt: Path
    text_encoder_dir: Path
    text_head_pt: Path
    face_head_pt: Path
    fusion_pt: Path
    calib_path: Optional[Path] = None
    norm_path: Optional[Path] = None

    whisper_model: str = "large-v3"
    whisper_lang: Optional[str] = None

    target_sr: int = 16000
    audio_max_sec: float = 20.0
    target_fps: int = 5
    num_bins: int = 100
    text_max_len: int = 128
    face_min_prob: float = 0.90
    face_min_valid: int = 5


settings = Settings()
