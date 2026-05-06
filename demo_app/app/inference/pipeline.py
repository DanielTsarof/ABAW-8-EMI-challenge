from .audio import run_audio
from .face import run_face
from .fusion import run_fusion, run_fusion_json
from .text import run_text


def _run_modalities(video_path: str, cfg):
    print(f"[pipeline] video={video_path}")
    audio_pred, audio_avail, audio_quality = run_audio(video_path, cfg)
    text_pred, text_avail, text_quality = run_text(video_path, cfg)
    face_pred, face_avail, face_quality = run_face(video_path, cfg)
    return (
        [audio_pred, text_pred, face_pred],
        [audio_avail, text_avail, face_avail],
        [audio_quality, text_quality, face_quality],
    )


def run_pipeline(video_path: str) -> bytes:
    from app.config import settings
    cfg = settings
    preds, avails, qualities = _run_modalities(video_path, cfg)
    png_bytes = run_fusion(preds=preds, avails=avails, qualities=qualities, cfg=cfg)
    print("[pipeline] done")
    return png_bytes


def run_pipeline_json(video_path: str) -> dict:
    from app.config import settings
    cfg = settings
    preds, avails, qualities = _run_modalities(video_path, cfg)
    result = run_fusion_json(preds=preds, avails=avails, qualities=qualities, cfg=cfg)
    print("[pipeline] done (json)")
    return result
