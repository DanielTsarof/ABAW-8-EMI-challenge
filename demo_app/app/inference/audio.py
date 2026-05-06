import subprocess

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from .models import AudioWavLMModel, free_gpu, hidden_to_fps_bins


def run_audio(video_path: str, cfg) -> tuple:
    hop_sec = 1.0 / cfg.target_fps

    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(cfg.target_sr),
            "-t", str(cfg.audio_max_sec),
            "-f", "f32le", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')}")

    wav_np = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if wav_np.size == 0:
        raise RuntimeError("ffmpeg returned empty audio — video may have no audio stream")

    wav = torch.from_numpy(wav_np)
    print(f"[audio] {wav.shape[0] / cfg.target_sr:.1f}s extracted")

    wavlm = WavLMModel.from_pretrained(str(cfg.audio_encoder_dir)).to(cfg.device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained(str(cfg.audio_encoder_dir))

    fe = feat_ext(wav.numpy(), sampling_rate=cfg.target_sr, return_tensors="pt")
    fe = {k: v.to(cfg.device) for k, v in fe.items()}

    with torch.no_grad():
        attn_mask = fe.get(
            "attention_mask",
            torch.ones(1, fe["input_values"].shape[1], device=cfg.device, dtype=torch.long),
        )
        input_lengths = attn_mask.sum(-1)
        output_lengths = wavlm._get_feat_extract_output_lengths(input_lengths)
        out = wavlm(**fe)
        hidden = out.last_hidden_state  # [1, Lmax, 1024]
        audio_bins, audio_mask = hidden_to_fps_bins(
            hidden, input_lengths, output_lengths, cfg.target_sr, hop_sec, cfg.num_bins,
        )

    audio_bins = audio_bins.cpu().float()
    audio_mask = audio_mask.cpu()
    audio_quality = float(audio_mask[0].float().mean())

    free_gpu(wavlm, out, hidden, fe)
    print(f"[audio] WavLM freed  |  quality={audio_quality:.3f}")

    pkg = torch.load(cfg.audio_head_pt, map_location="cpu")
    h_in = int(pkg["h_in"])
    train_bins = int(pkg["num_bins"])

    audio_head = AudioWavLMModel(h_in=h_in).to(cfg.device).eval()
    audio_head.load_state_dict(pkg["head_state"], strict=True)

    T = audio_bins.shape[1]
    if T < train_bins:
        pad = train_bins - T
        audio_bins = torch.cat([audio_bins, torch.zeros(1, pad, h_in)], dim=1)
        audio_mask = torch.cat([audio_mask, torch.zeros(1, pad, dtype=torch.bool)], dim=1)
    elif T > train_bins:
        audio_bins = audio_bins[:, :train_bins]
        audio_mask = audio_mask[:, :train_bins]

    with torch.no_grad():
        audio_pred = audio_head(audio_bins.to(cfg.device), audio_mask.to(cfg.device))
        audio_pred = audio_pred.cpu().numpy()[0]  # [E]

    audio_avail = bool(audio_mask[0].any().item())
    free_gpu(audio_head)
    print(f"[audio] head freed  |  avail={audio_avail}")

    return audio_pred, audio_avail, audio_quality
