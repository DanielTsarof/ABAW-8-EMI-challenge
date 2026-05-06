import gc

import cv2
import numpy as np
import torch
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from facenet_pytorch import MTCNN
from PIL import Image

from .models import EmiFaceHead, crop_face, free_gpu, pool_face_emb

E = 6
FACE_MARGIN = 0.20
MTCNN_BATCH = 16
FACE_ENC_BATCH = 64


def run_face(video_path: str, cfg) -> tuple:
    cap = cv2.VideoCapture(str(video_path))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps_src / cfg.target_fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    print(f"[face] {len(frames)} frames extracted  (src_fps={fps_src:.1f}, stride={stride})")

    mtcnn = MTCNN(
        keep_all=True, device=cfg.device,
        thresholds=[0.6, 0.7, 0.7], min_face_size=40, post_process=False,
    )

    def _largest(boxes):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(areas))

    face_imgs = []
    face_probs = []

    for b0 in range(0, len(frames), MTCNN_BATCH):
        batch = frames[b0:b0 + MTCNN_BATCH]
        pil_batch = [Image.fromarray(f) for f in batch]
        boxes_l, probs_l = mtcnn.detect(pil_batch)
        for frame_np, boxes, probs in zip(batch, boxes_l, probs_l):
            if boxes is None or probs is None:
                continue
            keep = probs >= cfg.face_min_prob
            if not keep.any():
                continue
            bk = boxes[keep]
            pk = probs[keep]
            k = _largest(bk)
            crop = crop_face(frame_np, bk[k], FACE_MARGIN)
            if crop.size == 0:
                continue
            face_imgs.append(crop)
            face_probs.append(float(pk[k]))

    del mtcnn
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[face] MTCNN freed  |  {len(face_imgs)} faces detected")

    encoder = EmotiEffLibRecognizer(engine="torch", model_name="enet_b0_8_va_mtl", device=cfg.device)
    emb_chunks = []
    for b0 in range(0, len(face_imgs), FACE_ENC_BATCH):
        feats = encoder.extract_features(face_imgs[b0:b0 + FACE_ENC_BATCH])
        emb_chunks.append(feats)
    del encoder
    torch.cuda.empty_cache()
    gc.collect()
    print("[face] EmotiEffNet freed")

    if emb_chunks:
        face_emb_arr = np.concatenate(emb_chunks, axis=0).astype(np.float32)
        face_valid_arr = np.ones(len(face_emb_arr), dtype=bool)
        face_pool = pool_face_emb(face_emb_arr, face_valid_arr)
        face_mean_prob = float(np.mean(face_probs))
    else:
        face_pool = None
        face_mean_prob = 0.0

    emi_ckpt = torch.load(cfg.face_head_pt, map_location="cpu")
    in_dim = int(emi_ckpt["in_dim"])
    face_head = EmiFaceHead(in_dim, E).eval()
    face_head.load_state_dict(emi_ckpt["model_state"], strict=True)

    if face_pool is not None and len(face_imgs) >= cfg.face_min_valid:
        xt = torch.from_numpy(face_pool).unsqueeze(0)
        with torch.no_grad():
            face_pred = face_head(xt).numpy()[0]  # [E]
        face_avail = True
        face_quality = face_mean_prob
    else:
        face_pred = np.zeros(E, dtype=np.float32)
        face_avail = False
        face_quality = 0.0

    free_gpu(face_head)
    print(f"[face] head freed  |  avail={face_avail}  |  quality={face_quality:.3f}")

    return face_pred, face_avail, face_quality
