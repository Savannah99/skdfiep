"""Multi-task MobileNet-TS: does combining the original flow-regression
objective WITH the phase-label supervision (instead of replacing one with
the other) produce a better sampling signal than either alone? Same
MobileNetV3-Small backbone, freeze_before=3 (the best single-task
MobileNet-TS config), two heads sharing the backbone: one classification
head (phase label, cross-entropy) and one regression head (flow_mag_z, MSE).
usage: python pretrain_mobilenet_multitask.py <freeze_before> <epochs> <flow_loss_weight>
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

FRAMES_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/cholec80/frames_1fps")
FEAT_ROOT   = Path("/home/luning/CodeRepo/SurgKeyMAE/features/vit_small")  # for labels only
FLOW_ROOT   = Path("/home/luning/CodeRepo/SurgKeyMAE/features/flow_mag")
SPLIT_DIR   = Path("/home/luning/CodeRepo/SurgKeyMAE/cholec80/splits")
DEVICE = "cuda"
FREEZE_BEFORE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FLOW_WEIGHT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
OUT_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/mobilenet_multitask_score")
BATCH = 64
NUM_CLASSES, WINDOW, STRIDE = 7, 16, 16

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_labels(vid): return np.load(str(FEAT_ROOT / vid / "labels.npy")).astype(np.int64)
def load_flow_mag(vid): return np.load(str(FLOW_ROOT / vid / "flow.npy")).astype(np.float32)[:, 0]
def frame_paths(vid, T): return [FRAMES_ROOT / vid / f"{i:05d}.jpg" for i in range(T)]

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

set_seed = lambda s: (torch.manual_seed(s), torch.cuda.manual_seed_all(s))
set_seed(0)

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

model = MobileNetMultiTask(freeze_before=FREEZE_BEFORE, num_classes=NUM_CLASSES).to(DEVICE)
opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)

print(f"=== multi-task MobileNet (phase classification + flow regression, flow_weight={FLOW_WEIGHT}) ===")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    total_cls_loss, total_flow_loss, n_frames, n_correct = 0.0, 0.0, 0, 0
    for vid in train_ids:
        frame_lab = all_frame_labels[vid]
        valid_idx = np.where(frame_lab >= 0)[0]
        valid_idx = valid_idx[valid_idx > 0]  # skip frame 0 (no flow defined)
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
            loss = cls_loss + FLOW_WEIGHT * flow_loss
            loss.backward()
            opt.step()
            total_cls_loss += cls_loss.item() * len(batch_idx)
            total_flow_loss += flow_loss.item() * len(batch_idx)
            n_correct += (cls_logits.argmax(dim=1) == batch_cls_target).sum().item()
            n_frames += len(batch_idx)
    print(f"  epoch {epoch}/{EPOCHS}  cls_loss={total_cls_loss/n_frames:.4f}  flow_mse={total_flow_loss/n_frames:.4f}  train_acc={n_correct/n_frames:.4f}  ({time.time()-t0:.0f}s)", flush=True)

torch.save({"features": model.features.state_dict(),
            "trunk": model.trunk.state_dict(),
            "flow_head": model.flow_head.state_dict(),
            "freeze_before": FREEZE_BEFORE},
           "/tmp/mobilenet_ts_plus_cholec80_backbone.pt")
print("saved Cholec80-trained backbone+trunk+flow_head checkpoint for cross-dataset fine-tuning")

model.eval()
for p in model.parameters(): p.requires_grad = False

print("\n=== extracting frozen max-softmax-confidence scores (from cls head) for all videos ===")
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
        out_dir = OUT_ROOT / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / "score.npy"), scores.astype(np.float32))

print(f"saved multi-task MobileNet confidence scores for {len(all_ids)} videos to {OUT_ROOT}")
