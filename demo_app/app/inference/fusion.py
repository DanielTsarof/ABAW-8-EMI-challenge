import io

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import QualityAwareGatedFusion

EMOTIONS = ["Admiration", "Amusement", "Determination", "Empathic Pain", "Excitement", "Joy"]


def run_fusion(preds: list, avails: list, qualities: list, cfg) -> bytes:
    preds_arr = np.stack(preds, axis=0)  # [3, E]
    avails_arr = np.array(avails, dtype=bool)
    quals_arr = np.array(qualities, dtype=np.float32)

    if cfg.calib_path and cfg.calib_path.exists():
        cal = np.load(cfg.calib_path, allow_pickle=False)
        preds_arr[0] = preds_arr[0] * cal["a_audio"] + cal["b_audio"]
        preds_arr[1] = preds_arr[1] * cal["a_text"] + cal["b_text"]
        preds_arr[2] = preds_arr[2] * cal["a_video"] + cal["b_video"]
        print("[fusion] calibration applied")

    if not cfg.fusion_pt.exists():
        raise FileNotFoundError(
            f"Fusion checkpoint not found: {cfg.fusion_pt}\n"
            "Run multimodal_fusion_emi_face_v1.ipynb first."
        )

    fusion = QualityAwareGatedFusion(hidden=32).eval()
    ckpt = torch.load(cfg.fusion_pt, map_location="cpu")
    fusion.load_state_dict(ckpt["model_state"], strict=True)

    P_t = torch.from_numpy(preds_arr).unsqueeze(0).float()
    A_t = torch.from_numpy(avails_arr).unsqueeze(0)
    Q_t = torch.from_numpy(quals_arr).unsqueeze(0).float()

    with torch.no_grad():
        fused_norm, weights = fusion(P_t, A_t, Q_t)
    fused_norm = fused_norm.numpy()[0]
    weights = weights.numpy()[0]  # [3, E]

    if cfg.norm_path and cfg.norm_path.exists():
        norm = np.load(cfg.norm_path, allow_pickle=False)
        fused = fused_norm * norm["y_std"] + norm["y_mean"]
        print("[fusion] target denormalization applied")
    else:
        fused = fused_norm
        print("[fusion] norm file not found — showing z-score scale")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Emotion Recognition", fontsize=13, fontweight="bold")

    cmap = plt.cm.RdYlGn
    vnorm = plt.Normalize(fused.min(), fused.max())
    colors = [cmap(vnorm(v)) for v in fused]
    bars = ax1.barh(EMOTIONS, fused, color=colors, edgecolor="grey", linewidth=0.5)
    ax1.set_xlabel("Predicted Score", fontsize=11)
    ax1.set_title("Emotion Vector", fontsize=12)
    span = max(abs(fused).max() * 1.18, 0.05)
    ax1.set_xlim(-span if fused.min() < 0 else 0, span)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, fused):
        xpos = bar.get_width() + span * 0.02
        ax1.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

    mod_names = ["Audio", "Text", "Face"]
    mod_cols = ["#4CAF50", "#2196F3", "#FF9800"]
    mean_w = weights.mean(axis=1)  # [3]
    ax2.bar(mod_names, mean_w, color=mod_cols, edgecolor="grey", linewidth=0.5)
    ax2.set_ylabel("Mean Gate Weight", fontsize=11)
    ax2.set_title("Modality Contributions", fontsize=12)
    ax2.set_ylim(0, max(1.0, mean_w.max() * 1.35))
    for i, (w, av) in enumerate(zip(mean_w, avails)):
        label = f"{w:.3f}" + ("" if av else "\n(unavail.)")
        ax2.text(i, w + 0.01, label, ha="center", fontsize=10, color="black" if av else "gray")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
