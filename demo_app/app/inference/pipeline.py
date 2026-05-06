from .audio import run_audio
from .face import run_face
from .fusion import run_fusion
from .text import run_text


def run_pipeline(video_path: str) -> bytes:
    from app.config import settings
    cfg = settings

    print(f"[pipeline] video={video_path}")
    audio_pred, audio_avail, audio_quality = run_audio(video_path, cfg)
    text_pred, text_avail, text_quality = run_text(video_path, cfg)
    face_pred, face_avail, face_quality = run_face(video_path, cfg)

    png_bytes = run_fusion(
        preds=[audio_pred, text_pred, face_pred],
        avails=[audio_avail, text_avail, face_avail],
        qualities=[audio_quality, text_quality, face_quality],
        cfg=cfg,
    )
    print("[pipeline] done")
    return png_bytes
