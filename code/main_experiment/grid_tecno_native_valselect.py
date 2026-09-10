"""TeCNO native training (train=eval=MobileNet-TS+ argmax) on Cholec80,
with validation-set-selected baseline and checkpoint saving.
usage: python grid_tecno_native_valselect.py <window> <top_k> <seed>
"""
import sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import jaccard_score

sys.path.insert(0, "/home/luning/CodeRepo/SurgKeyMAE")
from model_tecno import TeCNO

WINDOW = int(sys.argv[1]); TOP_K = int(sys.argv[2]); SEED = int(sys.argv[3])
STRIDE = WINDOW
FEAT_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/vit_small")
SCORE_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/mobilenet_multitask_score")
SPLIT_DIR = Path("/home/luning/CodeRepo/SurgKeyMAE/cholec80/splits")
DEVICE = "cuda"
NUM_CLASSES, D_FEAT, EPOCHS, LR = 7, 384, 50, 1e-3
N_EVAL_DRAWS = 5
CKPT_OUT = Path(f"/tmp/ckpt_tecno_cholec80_w{WINDOW}_topk{TOP_K}_seed{SEED}.pt")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_labels(vid): return np.load(str(FEAT_ROOT / vid / "labels.npy")).astype(np.int64)
def load_feats(vid): return np.load(str(FEAT_ROOT / vid / "features.npy")).astype(np.float32)
def load_score(vid): return np.load(str(SCORE_ROOT / vid / "score.npy")).astype(np.float32)

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
            lo = k * WINDOW // TOP_K; hi = (k+1) * WINDOW // TOP_K if k < TOP_K-1 else WINDOW
            idx.append(s + rng.integers(lo, hi))
        out.append(feats[idx].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

def fixed_feats(feats, offset):
    T = len(feats); out = []
    for s in range(0, T - WINDOW + 1, STRIDE):
        idx = []
        for k in range(TOP_K):
            lo = k * WINDOW // TOP_K; hi = (k+1) * WINDOW // TOP_K if k < TOP_K-1 else WINDOW
            bs = hi - lo; pos = min(offset, bs - 1)
            idx.append(s + lo + pos)
        out.append(feats[idx].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, feats.shape[1]), dtype=np.float32)

def argmax_feats(feats, score_z):
    T = len(feats); out = []
    for s in range(0, T - WINDOW + 1, STRIDE):
        idx = []
        for k in range(TOP_K):
            lo = k * WINDOW // TOP_K; hi = (k+1) * WINDOW // TOP_K if k < TOP_K-1 else WINDOW
            local = score_z[s+lo:s+hi]
            idx.append(s + lo + int(np.argmax(local)))
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

def weighted_multistage_loss(outputs, labels, weights):
    total = 0.0
    for out in outputs:
        log_probs = F.log_softmax(out, dim=1)
        nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        w = weights[labels]
        total = total + (nll * w).sum() / w.sum().clamp_min(1e-8)
    return total / len(outputs)

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
train_concat = np.concatenate([all_score_raw[v][1:] for v in train_ids])
score_mu, score_sd = train_concat.mean(), train_concat.std() + 1e-6
all_score_z = {v: (all_score_raw[v] - score_mu) / score_sd for v in all_score_raw}

class_weights = median_frequency_weights(train_ids, all_labels_meta).to(DEVICE)

@torch.no_grad()
def eval_pool(model, video_ids, pool_fn, n_draws=1):
    model.eval()
    jaccs = []
    for _ in range(n_draws):
        all_preds, all_gt = [], []
        for vid in video_ids:
            meta_lab = all_labels_meta[vid]
            if len(meta_lab) == 0: continue
            meta_feat = pool_fn(vid)
            x = torch.from_numpy(meta_feat).float().unsqueeze(0).permute(0, 2, 1).to(DEVICE)
            logits = model(x)[-1]
            preds = logits.argmax(dim=1)[0].cpu().numpy()
            all_preds.extend(preds); all_gt.extend(meta_lab)
        jaccs.append(jaccard_score(all_gt, all_preds, average="macro", zero_division=0))
    return float(np.mean(jaccs))

model = TeCNO(input_dim=D_FEAT, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
train_pool_fn = lambda vid: argmax_feats(all_raw_feats[vid], all_score_z[vid])

best_jacc, best_state = 0.0, None
for epoch in range(1, EPOCHS + 1):
    model.train()
    random.shuffle(train_ids)
    for vid in train_ids:
        meta_lab = all_labels_meta[vid]
        if len(meta_lab) == 0: continue
        meta_feat = train_pool_fn(vid)
        x = torch.from_numpy(meta_feat).float().unsqueeze(0).permute(0, 2, 1).to(DEVICE)
        lab = torch.from_numpy(meta_lab).long().unsqueeze(0).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = weighted_multistage_loss(outputs, lab, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    val_jacc = eval_pool(model, val_ids, train_pool_fn)
    scheduler.step(val_jacc)
    if val_jacc > best_jacc:
        best_jacc = val_jacc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save({"model_state": best_state, "seed": SEED, "window": WINDOW, "top_k": TOP_K}, CKPT_OUT)

bin_size = WINDOW // TOP_K
val_random = eval_pool(model, val_ids, lambda vid: rand_feats(all_raw_feats[vid], rng), n_draws=N_EVAL_DRAWS)
val_fixed0 = eval_pool(model, val_ids, lambda vid: fixed_feats(all_raw_feats[vid], 0))
val_fixed1 = eval_pool(model, val_ids, lambda vid: fixed_feats(all_raw_feats[vid], bin_size // 2))
candidates = {"random": val_random, "fixed0": val_fixed0, "fixed1": val_fixed1}
winner = max(candidates, key=candidates.get)

test_random = eval_pool(model, test_ids, lambda vid: rand_feats(all_raw_feats[vid], rng), n_draws=N_EVAL_DRAWS)
test_fixed0 = eval_pool(model, test_ids, lambda vid: fixed_feats(all_raw_feats[vid], 0))
test_fixed1 = eval_pool(model, test_ids, lambda vid: fixed_feats(all_raw_feats[vid], bin_size // 2))
test_by_name = {"random": test_random, "fixed0": test_fixed0, "fixed1": test_fixed1}
test_valsel = test_by_name[winner]
test_leaky = max(test_random, test_fixed0, test_fixed1)
test_tsplus = eval_pool(model, test_ids, lambda vid: argmax_feats(all_raw_feats[vid], all_score_z[vid]))

print(f"RESULT tecno_native_valselect window={WINDOW} topk={TOP_K} seed={SEED}  winner={winner} test_valsel={test_valsel:.4f} test_leaky={test_leaky:.4f} test_TSplus={test_tsplus:.4f}  (val: random={val_random:.4f} fixed0={val_fixed0:.4f} fixed1={val_fixed1:.4f})")
