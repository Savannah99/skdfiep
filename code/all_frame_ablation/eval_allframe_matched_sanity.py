"""Sanity check: evaluate the ALL-FRAME-trained checkpoints under MATCHED
(no-subsampling, all-frame) inference too, to separate 'train/infer mismatch'
from 'all-frame training is just weak for this architecture'.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import jaccard_score

sys.path.insert(0, "/home/luning/CodeRepo/SurgKeyMAE")
from model_tecno import TeCNO
from model_trans_svnet import TransSVNetStage
import torch.nn as nn

WINDOW, STRIDE = 16, 16
FEAT_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/vit_small")
SPLIT_DIR = Path("/home/luning/CodeRepo/SurgKeyMAE/cholec80/splits")
TECNO_CKPT = Path(f"/home/luning/CodeRepo/SurgKeyMAE/runs/mstcn/tecno_window{WINDOW}_cholec80_s0/best.pth")
DEVICE = "cuda"
NUM_CLASSES, D_FEAT = 7, 384

def load_labels(vid): return np.load(str(FEAT_ROOT / vid / "labels.npy")).astype(np.int64)
def load_feats(vid): return np.load(str(FEAT_ROOT / vid / "features.npy")).astype(np.float32)

def window_majority_labels(labels, window, stride):
    meta = []
    for s in range(0, len(labels) - window + 1, stride):
        chunk = labels[s:s+window]; meta.append(int(np.bincount(chunk, minlength=NUM_CLASSES).argmax()))
    return np.array(meta, dtype=np.int64)

def window_mean_feats(feats, window, stride):
    T = len(feats); out = []
    for s in range(0, T - window + 1, stride):
        out.append(feats[s:s+window].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

test_ids = (SPLIT_DIR / "test.txt").read_text().strip().splitlines()
all_raw_feats, all_labels_meta = {}, {}
for vid in test_ids:
    all_raw_feats[vid] = load_feats(vid)
    all_labels_meta[vid] = window_majority_labels(load_labels(vid), WINDOW, STRIDE)

class GRUModel(nn.Module):
    def __init__(self, d_feat=384, hidden_dim=256, num_layers=2, num_classes=7, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(d_feat, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        h = self.proj(x); h, _ = self.gru(h)
        return self.head(h).permute(0, 2, 1)

@torch.no_grad()
def eval_bigru(seed):
    model = GRUModel().to(DEVICE)
    ckpt = torch.load(f"/tmp/ckpt_bigru_allframe_cholec80_seed{seed}.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["best_state"]); model.eval()
    all_preds, all_gt = [], []
    for vid in test_ids:
        meta_lab = all_labels_meta[vid]
        if len(meta_lab) == 0: continue
        meta_feat = window_mean_feats(all_raw_feats[vid], WINDOW, STRIDE)
        x = torch.from_numpy(meta_feat).float().unsqueeze(0).to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)[0].cpu().numpy()
        all_preds.extend(preds); all_gt.extend(meta_lab)
    return jaccard_score(all_gt, all_preds, average="macro", zero_division=0)

@torch.no_grad()
def eval_tecno(seed):
    model = TeCNO(input_dim=D_FEAT, num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(f"/tmp/ckpt_tecno_allframe_cholec80_seed{seed}.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    all_preds, all_gt = [], []
    for vid in test_ids:
        meta_lab = all_labels_meta[vid]
        if len(meta_lab) == 0: continue
        meta_feat = window_mean_feats(all_raw_feats[vid], WINDOW, STRIDE)
        x = torch.from_numpy(meta_feat).float().unsqueeze(0).permute(0, 2, 1).to(DEVICE)
        logits = model(x)[-1]
        preds = logits.argmax(dim=1)[0].cpu().numpy()
        all_preds.extend(preds); all_gt.extend(meta_lab)
    return jaccard_score(all_gt, all_preds, average="macro", zero_division=0)

@torch.no_grad()
def eval_transsvnet(seed):
    tecno = TeCNO(input_dim=D_FEAT, num_classes=NUM_CLASSES).to(DEVICE)
    tckpt = torch.load(TECNO_CKPT, map_location=DEVICE, weights_only=False)
    tecno.load_state_dict(tckpt["model_state"]); tecno.eval()
    model = TransSVNetStage(num_classes=NUM_CLASSES, spatial_dim=D_FEAT, len_q=30).to(DEVICE)
    ckpt = torch.load(f"/tmp/ckpt_transsvnet_allframe_cholec80_seed{seed}.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    all_preds, all_gt = [], []
    for vid in test_ids:
        meta_lab = all_labels_meta[vid]
        if len(meta_lab) == 0: continue
        meta_feat = window_mean_feats(all_raw_feats[vid], WINDOW, STRIDE)
        x = torch.from_numpy(meta_feat).float().unsqueeze(0).permute(0, 2, 1).to(DEVICE)
        tl = tecno(x)[-1].permute(0, 2, 1)
        sp = torch.from_numpy(meta_feat).float().unsqueeze(0).to(DEVICE)
        out = model(tl, sp)
        preds = out.argmax(dim=-1).squeeze(0).cpu().numpy()
        all_preds.extend(preds); all_gt.extend(meta_lab)
    return jaccard_score(all_gt, all_preds, average="macro", zero_division=0)

for seed in [0, 1, 2]:
    b = eval_bigru(seed); t = eval_tecno(seed); ts = eval_transsvnet(seed)
    print(f"seed={seed}  matched-allframe-eval:  bigru={b:.4f} tecno={t:.4f} transsvnet={ts:.4f}")
