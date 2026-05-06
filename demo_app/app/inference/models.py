import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

E = 6  # number of target emotions


class TCNBlock(nn.Module):
    def __init__(self, d, kernel, dilation, dropout):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(d, d, kernel_size=kernel, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(d, d, kernel_size=kernel, dilation=dilation, padding=padding)
        self.norm1 = nn.GroupNorm(1, d)
        self.norm2 = nn.GroupNorm(1, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.drop(F.gelu(self.norm1(self.conv1(x))))
        x = self.drop(F.gelu(self.norm2(self.conv2(x))))
        return x + res


class TCNEncoder(nn.Module):
    def __init__(self, d, layers, kernel, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([TCNBlock(d, kernel, 2**i, dropout) for i in range(layers)])

    def forward(self, x):  # [B, T, d]
        x = x.transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        return x.transpose(1, 2)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, d, attn_hidden, dropout, temp=1.5):
        super().__init__()
        self.temp = temp
        self.attn = nn.Sequential(
            nn.Linear(d, attn_hidden), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(attn_hidden, 1),
        )

    def forward(self, x, mask):  # x: [B, T, d], mask: [B, T] bool
        logits = self.attn(x).squeeze(-1) / self.temp
        logits = logits.masked_fill(~mask, -1e4)
        w = torch.softmax(logits, dim=1)
        w = w * mask.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        w = w.unsqueeze(-1)
        mu = (w * x).sum(dim=1)
        var = (w * (x - mu.unsqueeze(1)).pow(2)).sum(dim=1)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mu, std], dim=-1)


class AudioWavLMModel(nn.Module):
    def __init__(self, h_in, d_model=192, tcn_layers=6, tcn_kernel=3, dropout=0.3, attn_hidden=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(h_in, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout),
        )
        self.enc = TCNEncoder(d_model, tcn_layers, tcn_kernel, dropout)
        self.pool = AttentiveStatsPooling(d_model, attn_hidden, dropout, temp=1.5)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d_model, E),
        )

    def forward(self, x, mask):
        x = self.proj(x)
        x = self.enc(x)
        z = self.pool(x, mask)
        return self.head(z)


class EmiFaceHead(nn.Module):
    def __init__(self, in_dim, num_targets):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_targets)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class QualityAwareGatedFusion(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(E + 2, hidden), nn.GELU(), nn.Linear(hidden, E),
        )
        self.bias = nn.Parameter(torch.zeros(E))

    def forward(self, P, avail, quality):
        # P: [B, 3, E], avail: [B, 3] bool, quality: [B, 3] float
        feat = torch.cat([P, quality.unsqueeze(-1), avail.unsqueeze(-1).float()], dim=-1)
        logits = self.mlp(feat)
        mask = avail.unsqueeze(-1).expand_as(logits)
        logits = logits.masked_fill(~mask, -1e4)
        w = torch.softmax(logits, dim=1)
        w = w * mask.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        fused = (w * P).sum(dim=1) + self.bias
        return fused, w


def hidden_to_fps_bins(hidden, input_lengths, output_lengths, sr, hop_sec, num_bins):
    B, Lmax, H = hidden.shape
    device = hidden.device
    dur = input_lengths.float() / float(sr)
    x_out, m_out = [], []
    for b in range(B):
        Lb = int(output_lengths[b].item())
        if Lb <= 0:
            x_out.append(torch.zeros((num_bins, H), device=device, dtype=hidden.dtype))
            m_out.append(torch.zeros((num_bins,), device=device, dtype=torch.bool))
            continue
        hb = hidden[b, :Lb, :]
        stride = dur[b] / float(Lb)
        t = torch.arange(Lb, device=device, dtype=torch.float32) * stride
        idx = torch.floor(t / float(hop_sec)).long()
        valid = (idx >= 0) & (idx < num_bins)
        sums = torch.zeros((num_bins, H), device=device, dtype=hidden.dtype)
        cnts = torch.zeros((num_bins,), device=device, dtype=torch.int32)
        if valid.any():
            ii = idx[valid]
            sums.index_add_(0, ii, hb[valid])
            cnts.index_add_(0, ii, torch.ones_like(ii, dtype=torch.int32))
        xb = sums / cnts.clamp_min(1).unsqueeze(1).to(sums.dtype)
        mb = cnts > 0
        x_out.append(xb)
        m_out.append(mb)
    return torch.stack(x_out), torch.stack(m_out)


def pool_face_emb(emb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = emb[valid].astype(np.float32) if valid.any() else emb.astype(np.float32)
    mean_f = x.mean(0)
    std_f = x.std(0) if x.shape[0] > 1 else np.zeros_like(mean_f)
    min_f = x.min(0)
    max_f = x.max(0)
    return np.concatenate([mean_f, std_f, min_f, max_f]).astype(np.float32)


def crop_face(img_rgb: np.ndarray, box: np.ndarray, margin: float = 0.20) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * margin
    ph = (y2 - y1) * margin
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(W, int(x2 + pw))
    y2 = min(H, int(y2 + ph))
    return img_rgb[y1:y2, x1:x2]


def free_gpu(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
