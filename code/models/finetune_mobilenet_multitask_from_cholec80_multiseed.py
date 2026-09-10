"""Same as finetune_mobilenet_multitask_from_cholec80.py but parameterized by
seed and writing to a seed-specific score directory, so the downstream
crossover result can be reported as a proper 3-seed mean+-std instead of a
single seed=0 run.
usage: python finetune_mobilenet_multitask_from_cholec80_multiseed.py <dataset> <epochs> <seed>
"""
import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from concurrent.futures import ThreadPoolExecutor

DATASET = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
BASE = "/tmp/claude-1000/-home-luning-CodeRepo-SurgKeyMAE/a87fa977-49a1-424d-9851-a2558df0370d/scratchpad/multidata"
PATHS = {
    "autolaparo": dict(frames_root="/mnt/ssd4t/SurgKeyMAE/autolaparo_sf/frames",
                        feat_root="/mnt/ssd4t/SurgKeyMAE/features/autolaparo/vit_small",
                        flow_root=f"{BASE}/autolaparo/flow_mag", split_dir=f"{BASE}/autolaparo/splits", ext="png"),
    "heichole": dict(frames_root="/mnt/ssd4t/SurgKeyMAE/heichole_sf/frames_1fps",
                      feat_root="/mnt/ssd4t/SurgKeyMAE/features/heichole/vit_small",
                      flow_root=f"{BASE}/heichole/flow_mag", split_dir=f"{BASE}/heichole/splits", ext="jpg"),
}[DATASET]
FRAMES_ROOT = Path(PATHS["frames_root"]); FEAT_ROOT = Path(PATHS["feat_root"])
FLOW_ROOT = Path(PATHS["flow_root"]); SPLIT_DIR = Path(PATHS["split_dir"]); EXT = PATHS["ext"]
OUT_ROOT = Path(f"{BASE}/{DATASET}/mobilenet_multitask_ft_score_seed{SEED}")
DEVICE = "cuda"
BATCH = 64
NUM_CLASSES, WINDOW, STRIDE = 7, 16, 16
BACKBONE_CKPT = "/tmp/mobilenet_ts_plus_cholec80_backbone.pt"

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_labels(vid): return np.load(str(FEAT_ROOT / vid / "labels.npy")).astype(np.int64)
def load_flow_mag(vid): return np.load(str(FLOW_ROOT / vid / "flow.npy")).astype(np.float32)[:, 0]
def frame_paths(vid, T): return [FRAMES_ROOT / vid / f"{i:05d}.{EXT}" for i in range(T)]

def per_frame_window_labels(labels, window, stride):
    T = len(labels)
    n_windows = (T - window) // stride + 1 if T >= window else 0
    frame_labels = np.full(T, -1, dtype=np.int64)
    for wi in range(n_windows):
        s = wi * stride
        chunk = labels[s:s+window]
        lab = int(np.bincount(chunk, minlength=NUM_CLASSES).argmax())
        frame_labels[s:s+window] = lab
    return frame_labels

def load_frames_batch(paths):
    def _l(p): return transform(Image.open(p).convert("RGB"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        return torch.stack(list(ex.map(_l, paths)))

class MobileNetMultiTask(nn.Module):
    def __init__(self, freeze_before=3, num_classes=7):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        for i, block in enumerate(self.features):
            trainable = i >= freeze_before
            for p in block.parameters(): p.requires_grad = trainable
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(nn.Linear(576, 64), nn.ReLU(inplace=True))
        self.cls_head = nn.Linear(64, num_classes)
        self.flow_head = nn.Linear(64, 1)

    def forward(self, x):
        f = self.features(x)
        f = self.pool(f).flatten(1)
        h = self.trunk(f)
        return self.cls_head(h), self.flow_head(h).squeeze(-1)

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)
set_seed(SEED)

train_ids = (SPLIT_DIR / "train.txt").read_text().strip().splitlines()
val_ids   = (SPLIT_DIR / "val.txt").read_text().strip().splitlines()
test_ids  = (SPLIT_DIR / "test.txt").read_text().strip().splitlines()
all_ids = train_ids + val_ids + test_ids

all_frame_labels = {vid: per_frame_window_labels(load_labels(vid), WINDOW, STRIDE) for vid in all_ids}
all_flow_raw = {vid: load_flow_mag(vid) for vid in all_ids}
train_flow_concat = np.concatenate([all_flow_raw[v][1:] for v in train_ids])
flow_mu, flow_sd = train_flow_concat.mean(), train_flow_concat.std() + 1e-6
all_flow_z = {v: (all_flow_raw[v] - flow_mu) / flow_sd for v in all_flow_raw}

train_y_concat = np.concatenate([all_frame_labels[v][all_frame_labels[v] >= 0] for v in train_ids])
counts = np.bincount(train_y_concat, minlength=NUM_CLASSES).astype(np.float64)
freqs = counts / counts.sum()
median_freq = np.median(freqs[freqs > 0])
class_weights = torch.tensor(np.where(freqs > 0, median_freq / np.maximum(freqs, 1e-12), 0.0), dtype=torch.float32).to(DEVICE)

backbone_ckpt = torch.load(BACKBONE_CKPT, map_location=DEVICE, weights_only=False)
FREEZE_BEFORE = backbone_ckpt["freeze_before"]
model = MobileNetMultiTask(freeze_before=FREEZE_BEFORE, num_classes=NUM_CLASSES).to(DEVICE)
model.features.load_state_dict(backbone_ckpt["features"])
model.trunk.load_state_dict(backbone_ckpt["trunk"])
model.flow_head.load_state_dict(backbone_ckpt["flow_head"])
# cls_head stays freshly initialized -- phase semantics may differ across datasets
print(f"[seed={SEED}] initialized backbone+trunk+flow_head from Cholec80 checkpoint (freeze_before={FREEZE_BEFORE}); cls_head is fresh")

opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)

print(f"=== [{DATASET} seed={SEED}] fine-tuning from Cholec80 init, Train:{len(train_ids)}, epochs={EPOCHS} ===")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    total_cls, total_flow, n_frames, n_correct = 0.0, 0.0, 0, 0
    for vid in train_ids:
        frame_lab = all_frame_labels[vid]
        valid_idx = np.where(frame_lab >= 0)[0]
        valid_idx = valid_idx[valid_idx > 0]
        if len(valid_idx) == 0: continue
        paths_all = frame_paths(vid, len(frame_lab))
        flow_z = all_flow_z[vid]
        for i in range(0, len(valid_idx), BATCH):
            batch_idx = valid_idx[i:i+BATCH]
            batch_paths = [paths_all[j] for j in batch_idx]
            batch_cls_target = torch.from_numpy(frame_lab[batch_idx]).long().to(DEVICE)
            batch_flow_target = torch.from_numpy(flow_z[batch_idx]).float().to(DEVICE)
            frames = load_frames_batch(batch_paths).to(DEVICE)
            opt.zero_grad()
            cls_logits, flow_pred = model(frames)
            cls_loss = F.cross_entropy(cls_logits, batch_cls_target, weight=class_weights)
            flow_loss = F.mse_loss(flow_pred, batch_flow_target)
            loss = cls_loss + flow_loss
            loss.backward()
            opt.step()
            total_cls += cls_loss.item() * len(batch_idx)
            total_flow += flow_loss.item() * len(batch_idx)
            n_correct += (cls_logits.argmax(dim=1) == batch_cls_target).sum().item()
            n_frames += len(batch_idx)
    print(f"  [seed={SEED}] epoch {epoch}/{EPOCHS}  cls_loss={total_cls/n_frames:.4f}  flow_mse={total_flow/n_frames:.4f}  train_acc={n_correct/n_frames:.4f}  ({time.time()-t0:.0f}s)", flush=True)

model.eval()
for p in model.parameters(): p.requires_grad = False

print(f"\n=== [{DATASET} seed={SEED}] extracting scores ===")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for vid in all_ids:
        T = len(all_frame_labels[vid])
        paths = frame_paths(vid, T)
        scores = []
        for i in range(0, len(paths), BATCH):
            frames = load_frames_batch(paths[i:i+BATCH]).to(DEVICE)
            cls_logits, _ = model(frames)
            conf = F.softmax(cls_logits, dim=1).max(dim=1).values.cpu().numpy()
            scores.append(conf)
        scores = np.concatenate(scores)
        out_dir = OUT_ROOT / vid; out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / "score.npy"), scores.astype(np.float32))
print(f"[{DATASET} seed={SEED}] saved fine-tuned scores for {len(all_ids)} videos to {OUT_ROOT}")
