"""Evaluate an EXISTING BiGRU crossover checkpoint (trained under random
resampling, from the 324-config valselect run) under MGSampler's and
SCSampler-argmax's own selection rules at inference -- no retraining, since
BiGRU's crossover checkpoints already exist and are training-signal-neutral.
usage: python eval_bigru_mg_sc.py <dataset> <window> <top_k> <seed>
  dataset in {cholec80, autolaparo, heichole}
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score

DATASET = sys.argv[1]; WINDOW = int(sys.argv[2]); TOP_K = int(sys.argv[3]); SEED = int(sys.argv[4])
STRIDE = WINDOW
N_EVAL_DRAWS = 5
BASE = "/tmp/claude-1000/-home-luning-CodeRepo-SurgKeyMAE/a87fa977-49a1-424d-9851-a2558df0370d/scratchpad/multidata"
PATHS = {
    "cholec80": dict(feat_root="/home/luning/CodeRepo/SurgKeyMAE/features/vit_small",
                      flow_root="/home/luning/CodeRepo/SurgKeyMAE/features/flow_mag",
                      sc_root="/home/luning/CodeRepo/SurgKeyMAE/features/scsampler_score",
                      split_dir="/home/luning/CodeRepo/SurgKeyMAE/cholec80/splits"),
    "autolaparo": dict(feat_root="/mnt/ssd4t/SurgKeyMAE/features/autolaparo/vit_small",
                        flow_root=f"{BASE}/autolaparo/flow_mag", sc_root=f"{BASE}/autolaparo/scsampler_score",
                        split_dir=f"{BASE}/autolaparo/splits"),
    "heichole": dict(feat_root="/mnt/ssd4t/SurgKeyMAE/features/heichole/vit_small",
                      flow_root=f"{BASE}/heichole/flow_mag", sc_root=f"{BASE}/heichole/scsampler_score",
                      split_dir=f"{BASE}/heichole/splits"),
}[DATASET]
FEAT_ROOT = Path(PATHS["feat_root"]); FLOW_ROOT = Path(PATHS["flow_root"])
SCSAMPLER_ROOT = Path(PATHS["sc_root"]); SPLIT_DIR = Path(PATHS["split_dir"])
DEVICE = "cuda"
NUM_CLASSES = 7
CKPT_IN = Path(f"/tmp/ckpt_bigru_{DATASET}_w{WINDOW}_topk{TOP_K}_seed{SEED}.pt")

def load_labels(vid): return np.load(str(FEAT_ROOT / vid / "labels.npy")).astype(np.int64)
def load_feats(vid): return np.load(str(FEAT_ROOT / vid / "features.npy")).astype(np.float32)
def load_flow(vid): return np.load(str(FLOW_ROOT / vid / "flow.npy")).astype(np.float32)[:, 0]
def load_scsampler(vid): return np.load(str(SCSAMPLER_ROOT / vid / "score.npy")).astype(np.float32)

def window_majority_labels(labels, window, stride):
    meta = []
    for s in range(0, len(labels) - window + 1, stride):
        chunk = labels[s:s+window]
        meta.append(int(np.bincount(chunk, minlength=NUM_CLASSES).argmax()))
    return np.array(meta, dtype=np.int64)

def rand_feats(feats, rng):
    T = len(feats); out = []
    for s in range(0, T - WINDOW + 1, STRIDE):
        idx = []
        for k in range(TOP_K):
            lo = k*WINDOW//TOP_K; hi = (k+1)*WINDOW//TOP_K if k < TOP_K-1 else WINDOW
            idx.append(s + rng.integers(lo, hi))
        out.append(feats[idx].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

def fixed_feats(feats, offset):
    T = len(feats); out = []
    for s in range(0, T - WINDOW + 1, STRIDE):
        idx = []
        for k in range(TOP_K):
            lo = k*WINDOW//TOP_K; hi = (k+1)*WINDOW//TOP_K if k < TOP_K-1 else WINDOW
            bs = hi - lo; pos = min(offset, bs - 1)
            idx.append(s + lo + pos)
        out.append(feats[idx].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

def mgsampler_feats(feats, flow_mag, window, stride, top_k):
    T = len(feats); out = []
    for s in range(0, T - window + 1, stride):
        local_motion = np.clip(flow_mag[s:s+window], 0, None) + 1e-6
        cdf = np.cumsum(local_motion); cdf = cdf / cdf[-1]
        idx = []
        for i in range(top_k):
            q = (i + 0.5) / top_k
            j = int(np.searchsorted(cdf, q)); j = min(j, window - 1)
            idx.append(s + j)
        idx = sorted(set(idx))
        while len(idx) < top_k:
            for cand in range(window):
                if (s + cand) not in idx:
                    idx.append(s + cand); break
        out.append(feats[idx[:top_k]].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

def argmax_feats(feats, score_z):
    T = len(feats); out = []
    for s in range(0, T - WINDOW + 1, STRIDE):
        idx = []
        for k in range(TOP_K):
            lo = k*WINDOW//TOP_K; hi = (k+1)*WINDOW//TOP_K if k < TOP_K-1 else WINDOW
            local = score_z[s+lo:s+hi]; idx.append(s + lo + int(np.argmax(local)))
        out.append(feats[idx].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

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

rng = np.random.default_rng(SEED)
train_ids = (SPLIT_DIR / "train.txt").read_text().strip().splitlines()
val_ids   = (SPLIT_DIR / "val.txt").read_text().strip().splitlines()
test_ids  = (SPLIT_DIR / "test.txt").read_text().strip().splitlines()

all_raw_feats, all_labels_meta = {}, {}
for vid in train_ids + val_ids + test_ids:
    all_raw_feats[vid] = load_feats(vid)
    all_labels_meta[vid] = window_majority_labels(load_labels(vid), WINDOW, STRIDE)

all_flow_raw = {v: load_flow(v) for v in test_ids}
all_sc_raw = {v: load_scsampler(v) for v in train_ids + test_ids}
sc_mu, sc_sd = (lambda c: (c.mean(), c.std() + 1e-6))(np.concatenate([all_sc_raw[v] for v in train_ids]))
all_sc_z = {v: (all_sc_raw[v] - sc_mu) / sc_sd for v in all_sc_raw}

model = GRUModel().to(DEVICE)
ckpt = torch.load(CKPT_IN, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["best_state"]); model.eval()

@torch.no_grad()
def eval_pool(video_ids, pool_fn, n_draws=1):
    model.eval()
    jaccs = []
    for _ in range(n_draws):
        all_preds, all_gt = [], []
        for vid in video_ids:
            meta_lab = all_labels_meta[vid]
            if len(meta_lab) == 0: continue
            meta_feat = pool_fn(vid)
            x = torch.from_numpy(meta_feat).float().unsqueeze(0).to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)[0].cpu().numpy()
            all_preds.extend(preds); all_gt.extend(meta_lab)
        jaccs.append(jaccard_score(all_gt, all_preds, average="macro", zero_division=0))
    return float(np.mean(jaccs))

bin_size = WINDOW // TOP_K
val_random = eval_pool(val_ids, lambda vid: rand_feats(all_raw_feats[vid], rng), n_draws=N_EVAL_DRAWS)
val_fixed0 = eval_pool(val_ids, lambda vid: fixed_feats(all_raw_feats[vid], 0))
val_fixed1 = eval_pool(val_ids, lambda vid: fixed_feats(all_raw_feats[vid], bin_size // 2))
candidates = {"random": val_random, "fixed0": val_fixed0, "fixed1": val_fixed1}
winner = max(candidates, key=candidates.get)

test_random = eval_pool(test_ids, lambda vid: rand_feats(all_raw_feats[vid], rng), n_draws=N_EVAL_DRAWS)
test_fixed0 = eval_pool(test_ids, lambda vid: fixed_feats(all_raw_feats[vid], 0))
test_fixed1 = eval_pool(test_ids, lambda vid: fixed_feats(all_raw_feats[vid], bin_size // 2))
test_by_name = {"random": test_random, "fixed0": test_fixed0, "fixed1": test_fixed1}
test_valsel = test_by_name[winner]

test_mg = eval_pool(test_ids, lambda vid: mgsampler_feats(all_raw_feats[vid], all_flow_raw[vid], WINDOW, STRIDE, TOP_K))
test_sc = eval_pool(test_ids, lambda vid: argmax_feats(all_raw_feats[vid], all_sc_z[vid]))

print(f"RESULT bigru_mgsc_{DATASET} window={WINDOW} topk={TOP_K} seed={SEED}  winner={winner} test_valsel={test_valsel:.4f} test_MG={test_mg:.4f} test_SC={test_sc:.4f}  (val: random={val_random:.4f} fixed0={val_fixed0:.4f} fixed1={val_fixed1:.4f})")
