"""Test whether MobileNet-TS+'s argmax selection systematically avoids
frames near phase-transition boundaries (ambiguous/mixed-content frames)
more than fixed-position does -- a concrete, checkable form of
"content-aware" selection, as opposed to asserting interpretability
without evidence.
"""
from pathlib import Path
import numpy as np

FEAT_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/vit_small")
SCORE_ROOT = Path("/home/luning/CodeRepo/SurgKeyMAE/features/mobilenet_multitask_score")
SPLIT_DIR = Path("/home/luning/CodeRepo/SurgKeyMAE/cholec80/splits")
WINDOW, STRIDE, TOP_K = 16, 16, 8

def load_labels(vid): return np.load(str(FEAT_ROOT / vid / "labels.npy")).astype(np.int64)
def load_score(vid): return np.load(str(SCORE_ROOT / vid / "score.npy")).astype(np.float32)

test_ids = (SPLIT_DIR / "test.txt").read_text().strip().splitlines()
train_ids = (SPLIT_DIR / "train.txt").read_text().strip().splitlines()

all_score_raw = {v: load_score(v) for v in train_ids + test_ids}
train_concat = np.concatenate([all_score_raw[v] for v in train_ids])
mu, sd = train_concat.mean(), train_concat.std() + 1e-6

def nearest_transition_distance(labels, idx):
    """Distance (in frames) from idx to the nearest phase-transition boundary
    in the FULL video label sequence (a boundary is between i-1 and i where
    label changes)."""
    boundaries = np.where(np.diff(labels) != 0)[0]  # boundary is between b and b+1
    if len(boundaries) == 0:
        return len(labels)  # no transitions in this video
    # distance from idx to nearest boundary point (treating boundary as being at b+0.5)
    dists = np.abs(boundaries + 0.5 - idx)
    return dists.min()

fixed0_dists, argmax_dists = [], []
per_video_summary = []

for vid in test_ids:
    labels = load_labels(vid)
    score_raw = all_score_raw[vid]
    score_z = (score_raw - mu) / sd
    T = len(labels)
    vid_fixed, vid_argmax = [], []
    for s in range(0, T - WINDOW + 1, STRIDE):
        for k in range(TOP_K):
            lo = k * WINDOW // TOP_K
            hi = (k + 1) * WINDOW // TOP_K if k < TOP_K - 1 else WINDOW
            bin_size = hi - lo
            # fixed-position: offset 0 (first frame of bin)
            idx_fixed = s + lo + 0
            # MobileNet-TS+ argmax
            local = score_z[s+lo:s+hi]
            idx_argmax = s + lo + int(np.argmax(local))

            d_fixed = nearest_transition_distance(labels, idx_fixed)
            d_argmax = nearest_transition_distance(labels, idx_argmax)
            fixed0_dists.append(d_fixed)
            argmax_dists.append(d_argmax)
            vid_fixed.append(d_fixed); vid_argmax.append(d_argmax)
    per_video_summary.append((vid, np.mean(vid_fixed), np.mean(vid_argmax)))

fixed0_dists = np.array(fixed0_dists)
argmax_dists = np.array(argmax_dists)

print(f"Total bins analyzed: {len(fixed0_dists)}")
print(f"Fixed-position: mean distance to nearest transition = {fixed0_dists.mean():.3f} frames (median={np.median(fixed0_dists):.1f})")
print(f"MobileNet-TS+ argmax: mean distance to nearest transition = {argmax_dists.mean():.3f} frames (median={np.median(argmax_dists):.1f})")

diff = argmax_dists - fixed0_dists
from scipy import stats
t, p = stats.ttest_1samp(diff, 0)
print(f"\nPaired diff (argmax - fixed0): mean={diff.mean():+.4f}  t={t:.3f}  p={p:.6f}")
print(f"Fraction of bins where TS+ is farther from a transition: {np.mean(diff > 0):.3f}")

# fraction of selections landing within 1 frame of a transition boundary (a "bad" pick)
close_fixed = np.mean(fixed0_dists <= 1.0)
close_argmax = np.mean(argmax_dists <= 1.0)
print(f"\nFraction of picks within 1 frame of a transition boundary:")
print(f"  fixed-position: {close_fixed:.3f}")
print(f"  MobileNet-TS+ argmax: {close_argmax:.3f}")
