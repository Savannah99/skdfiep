"""BiGRU trained on ALL frames in the window (full mean-pool, no top_k
subsampling at train time), then evaluated at inference under several
top_k=8 selection strategies: random, fixed0, fixed1(mid), MobileNet-TS+
(argmax), MGSampler (deterministic), SCSampler-style (argmax + stochastic).
This isolates train/inference distribution mismatch from selection-strategy
quality -- the current 324-cell grid always matches train/inference budget.
usage: python grid_bigru_allframe.py <seed>
"""
import sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import jaccard_score

SEED = int(sys.argv[1])
WINDOW, TOP_K, STRIDE, TEMPERATURE = 16, 8, 16, 1.0
FEAT_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/vit_small")
SCORE_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/mobilenet_multitask_score")
FLOW_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/flow_mag")
SCSAMPLER_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/scsampler_score")
SPLIT_DIR = Path("/home/luning/CodeRepo/SurgKeyMAE/cholec80/splits")
DEVICE = "cuda"
NUM_CLASSES, D_FEAT, EPOCHS = 7, 384, 50
N_EVAL_DRAWS = 5
CKPT_OUT = Path(f"/tmp/ckpt_bigru_allframe_cholec80_seed{SEED}.pt")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_labels(vid): return np.load(str(FEAT_ROOT / vid / "labels.npy")).astype(np.int64)
def load_feats(vid): return np.load(str(FEAT_ROOT / vid / "features.npy")).astype(np.float32)
def load_score(vid): return np.load(str(SCORE_ROOT / vid / "score.npy")).astype(np.float32)
def load_flow(vid): return np.load(str(FLOW_ROOT / vid / "flow.npy")).astype(np.float32)[:, 0]
def load_scsampler(vid): return np.load(str(SCSAMPLER_ROOT / vid / "score.npy")).astype(np.float32)

def window_majority_labels(labels, window, stride):
    meta = []
    for s in range(0, len(labels) - window + 1, stride):
        chunk = labels[s:s+window]
        meta.append(int(np.bincount(chunk, minlength=NUM_CLASSES).argmax()))
    return np.array(meta, dtype=np.int64)

def window_mean_feats(feats, window, stride):
    T = len(feats); out = []
    for s in range(0, T - window + 1, stride):
        out.append(feats[s:s+window].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

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

def argmax_feats(feats, score_z):
    T = len(feats); out = []
    for s in range(0, T - WINDOW + 1, STRIDE):
        idx = []
        for k in range(TOP_K):
            lo = k*WINDOW//TOP_K; hi = (k+1)*WINDOW//TOP_K if k < TOP_K-1 else WINDOW
            local = score_z[s+lo:s+hi]; idx.append(s + lo + int(np.argmax(local)))
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

def sc_stoch_feats(feats, score_z, window, stride, top_k, temperature, rng):
    T = len(feats); out = []
    for s in range(0, T - window + 1, stride):
        idx = []
        for k in range(top_k):
            lo = k*window//top_k; hi = (k+1)*window//top_k if k < top_k-1 else window
            local = score_z[s+lo:s+hi]; local = local - local.max()
            probs = np.exp(local / temperature); probs = probs / probs.sum()
            idx.append(s + lo + rng.choice(len(probs), p=probs))
        out.append(feats[idx].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

def median_frequency_weights(train_ids, all_labels_meta):
    counts = np.zeros(NUM_CLASSES)
    for vid in train_ids:
        for c in all_labels_meta[vid]: counts[c] += 1
    freqs = counts / counts.sum()
    median_freq = np.median(freqs[freqs > 0])
    weights = np.where(freqs > 0, median_freq / np.maximum(freqs, 1e-12), 0.0)
    return torch.tensor(weights, dtype=torch.float32)

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

set_seed(SEED)
rng = np.random.default_rng(SEED)
train_ids = (SPLIT_DIR / "train.txt").read_text().strip().splitlines()
val_ids   = (SPLIT_DIR / "val.txt").read_text().strip().splitlines()
test_ids  = (SPLIT_DIR / "test.txt").read_text().strip().splitlines()

all_raw_feats, all_labels_meta = {}, {}
for vid in train_ids + val_ids + test_ids:
    all_raw_feats[vid] = load_feats(vid)
    all_labels_meta[vid] = window_majority_labels(load_labels(vid), WINDOW, STRIDE)

all_score_raw = {v: load_score(v) for v in train_ids + val_ids + test_ids}
score_mu, score_sd = (lambda c: (c.mean(), c.std() + 1e-6))(np.concatenate([all_score_raw[v][1:] for v in train_ids]))
all_score_z = {v: (all_score_raw[v] - score_mu) / score_sd for v in all_score_raw}

all_flow_raw = {v: load_flow(v) for v in test_ids}

all_sc_raw = {v: load_scsampler(v) for v in train_ids + test_ids}
sc_mu, sc_sd = (lambda c: (c.mean(), c.std() + 1e-6))(np.concatenate([all_sc_raw[v] for v in train_ids]))
all_sc_z = {v: (all_sc_raw[v] - sc_mu) / sc_sd for v in all_sc_raw}

class_weights = median_frequency_weights(train_ids, all_labels_meta).to(DEVICE)

@torch.no_grad()
def eval_pool(model, video_ids, pool_fn, n_draws=1, seed_base=0):
    model.eval()
    jaccs = []
    for i in range(n_draws):
        all_preds, all_gt = [], []
        for vid in video_ids:
            meta_lab = all_labels_meta[vid]
            if len(meta_lab) == 0: continue
            meta_feat = pool_fn(vid, i)
            x = torch.from_numpy(meta_feat).float().unsqueeze(0).to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)[0].cpu().numpy()
            all_preds.extend(preds); all_gt.extend(meta_lab)
        jaccs.append(jaccard_score(all_gt, all_preds, average="macro", zero_division=0))
    return float(np.mean(jaccs))

model = GRUModel().to(DEVICE)
opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)
best_jacc, best_state = 0.0, None
for epoch in range(1, EPOCHS + 1):
    model.train()
    for vid in random.sample(train_ids, len(train_ids)):
        meta_lab = all_labels_meta[vid]
        if len(meta_lab) == 0: continue
        meta_feat = window_mean_feats(all_raw_feats[vid], WINDOW, STRIDE)
        x = torch.from_numpy(meta_feat).float().unsqueeze(0).to(DEVICE)
        lab = torch.from_numpy(meta_lab).long().unsqueeze(0).to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[0].T, lab[0], weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    val_jacc = eval_pool(model, val_ids, lambda vid, i: window_mean_feats(all_raw_feats[vid], WINDOW, STRIDE))
    sched.step(val_jacc)
    if val_jacc > best_jacc:
        best_jacc = val_jacc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save({"best_state": best_state, "seed": SEED, "window": WINDOW, "top_k": TOP_K, "train_mode": "allframe"}, CKPT_OUT)

test_random = eval_pool(model, test_ids, lambda vid, i: rand_feats(all_raw_feats[vid], np.random.default_rng(1000+SEED*10+i)), n_draws=N_EVAL_DRAWS)
test_fixed0 = eval_pool(model, test_ids, lambda vid, i: fixed_feats(all_raw_feats[vid], 0))
test_fixed1 = eval_pool(model, test_ids, lambda vid, i: fixed_feats(all_raw_feats[vid], 1))
test_tsplus = eval_pool(model, test_ids, lambda vid, i: argmax_feats(all_raw_feats[vid], all_score_z[vid]))
test_mg     = eval_pool(model, test_ids, lambda vid, i: mgsampler_feats(all_raw_feats[vid], all_flow_raw[vid], WINDOW, STRIDE, TOP_K))
test_sc_a   = eval_pool(model, test_ids, lambda vid, i: argmax_feats(all_raw_feats[vid], all_sc_z[vid]))
test_sc_s   = eval_pool(model, test_ids, lambda vid, i: sc_stoch_feats(all_raw_feats[vid], all_sc_z[vid], WINDOW, STRIDE, TOP_K, TEMPERATURE, np.random.default_rng(2000+SEED*10+i)), n_draws=N_EVAL_DRAWS)

print(f"RESULT bigru_allframe seed={SEED}  random={test_random:.4f} fixed0={test_fixed0:.4f} fixed1={test_fixed1:.4f} TSplus={test_tsplus:.4f} MGSampler={test_mg:.4f} SCSampler_argmax={test_sc_a:.4f} SCSampler_stoch={test_sc_s:.4f}")
