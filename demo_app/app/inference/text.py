import numpy as np
import torch
import torch.nn as nn
from faster_whisper import WhisperModel as FasterWhisper
from transformers import AutoModel, AutoTokenizer

from .models import free_gpu

E = 6


def run_text(video_path: str, cfg) -> tuple:
    compute_type = "float16" if cfg.device == "cuda" else "int8"
    print(f"[text] Loading Whisper {cfg.whisper_model} ({compute_type}) ...")
    whisper = FasterWhisper(
        cfg.whisper_model,
        device="cuda" if cfg.device == "cuda" else "cpu",
        compute_type=compute_type,
    )
    segments, info = whisper.transcribe(str(video_path), language=cfg.whisper_lang)
    transcript = " ".join(s.text.strip() for s in segments).strip() or "[NO SPEECH]"
    free_gpu(whisper)
    print(f"[text] Whisper freed  |  lang={getattr(info, 'language', '?')}  |  {len(transcript)} chars")

    print("[text] Loading GTE encoder ...")
    gte = AutoModel.from_pretrained(str(cfg.text_encoder_dir), trust_remote_code=True).to(cfg.device).eval()
    tok = AutoTokenizer.from_pretrained(str(cfg.text_encoder_dir), use_fast=True)

    inputs = tok(
        [transcript], padding=True, truncation=True,
        max_length=cfg.text_max_len, return_tensors="pt",
    )
    inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = gte(**inputs)
        cls_emb = out.last_hidden_state[:, 0, :].float()  # [1, 768]
        text_quality = float(torch.tanh(cls_emb.norm() / 10.0))
        cls_cpu = cls_emb.cpu()

    free_gpu(gte, out, inputs)
    print(f"[text] GTE freed  |  quality={text_quality:.3f}")

    ckpt_t = torch.load(cfg.text_head_pt, map_location="cpu")
    state = ckpt_t.get("model_state", ckpt_t)
    head_sd = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
    if not head_sd:
        raise ValueError("No 'head.*' keys found in text checkpoint")

    D_TEXT = int(cls_cpu.shape[-1])
    text_head = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(D_TEXT, 512), nn.GELU(),
        nn.Dropout(0.2), nn.Linear(512, E),
    ).eval()
    text_head.load_state_dict(head_sd, strict=True)

    with torch.no_grad():
        text_pred = text_head(cls_cpu).numpy()[0]  # [E]

    text_avail = transcript != "[NO SPEECH]"
    free_gpu(text_head, cls_cpu)
    print(f"[text] head freed  |  avail={text_avail}")

    return text_pred, text_avail, text_quality
