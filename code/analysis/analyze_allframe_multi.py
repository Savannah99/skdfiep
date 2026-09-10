import re
from collections import defaultdict
import numpy as np
from scipy import stats

pat = re.compile(
    r"RESULT (\w+?)_allframe_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"random=([\d.]+) fixed0=([\d.]+) fixed1=([\d.]+) TSplus=([\d.]+) MGSampler=([\d.]+) "
    r"SCSampler_argmax=([\d.]+) SCSampler_stoch=([\d.]+)"
)

data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

files = [f"/tmp/allframe_multi_shard{i}.log" for i in range(4)]
for fn in files:
    for line in open(fn):
        m = pat.search(line)
        if not m: continue
        arch, ds, w, k, seed, rnd, f0, f1, ts, mg, sca, scs = m.groups()
        w, k = int(w), int(k)
        data[arch][ds][(w, k)].append((float(rnd), float(f0), float(f1), float(ts), float(mg), float(sca), float(scs)))

archs = ["bigru", "tecno", "transsvnet"]
datasets = ["cholec80", "autolaparo", "heichole"]

# avoid re-introducing test-set-selection leakage: compare TS+ against fixed1
# (mid-bin), the established best non-learned convention, NOT against
# max(random,fixed0,fixed1) on the same test set.
rows = []
for arch in archs:
    for ds in datasets:
        for (w, k), vals in sorted(data[arch][ds].items()):
            if w == k: continue
            arr = np.array(vals)
            rnd, f0, f1, ts, mg, sca, scs = [arr[:, i] for i in range(7)]
            diff = ts - f1
            bin_size = w // k
            rows.append((arch, ds, w, k, bin_size, diff, rnd.mean(), f0.mean(), f1.mean(), ts.mean(), mg.mean(), sca.mean(), scs.mean()))

print(f"{'arch':<12}{'dataset':<12}{'w':>4}{'k':>4}{'bin':>5}{'n':>4}{'random':>8}{'fixed0':>8}{'fixed1':>8}{'TS+':>8}{'MG':>8}{'SCa':>8}{'SCs':>8}{'diff(TS+-f1)':>13}{'p':>8}  sign")
for arch, ds, w, k, bs, diff, rm, f0m, f1m, tsm, mgm, scam, scsm in rows:
    if len(diff) >= 2:
        t, p = stats.ttest_1samp(diff, 0)
        pstr = f"{p:.4f}"
    else:
        pstr = "n/a"
    print(f"{arch:<12}{ds:<12}{w:>4}{k:>4}{bs:>5}{len(diff):>4}{rm:>8.4f}{f0m:>8.4f}{f1m:>8.4f}{tsm:>8.4f}{mgm:>8.4f}{scam:>8.4f}{scsm:>8.4f}{diff.mean():>+13.4f}{pstr:>8}  {sum(diff>0)}/{len(diff)}")

print("\n=== Pooled by architecture (all datasets, all bin_sizes), TS+ vs fixed1 ===")
for arch in archs:
    all_diffs = np.concatenate([d for a, ds, w, k, bs, d, *_ in rows if a == arch])
    t, p = stats.ttest_1samp(all_diffs, 0)
    print(f"{arch}: n={len(all_diffs)} mean={all_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(all_diffs>0)}/{len(all_diffs)}")

print("\n=== Pooled by dataset (all architectures, all bin_sizes), TS+ vs fixed1 ===")
for ds in datasets:
    all_diffs = np.concatenate([d for a, dss, w, k, bs, d, *_ in rows if dss == ds])
    t, p = stats.ttest_1samp(all_diffs, 0)
    print(f"{ds}: n={len(all_diffs)} mean={all_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(all_diffs>0)}/{len(all_diffs)}")

grand = np.concatenate([d for a, ds, w, k, bs, d, *_ in rows])
t, p = stats.ttest_1samp(grand, 0)
print(f"\n=== GRAND TOTAL (TS+ vs fixed1) ===")
print(f"n={len(grand)} mean={grand.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(grand>0)}/{len(grand)}")

# also report where MGSampler / SCSampler stand overall, for completeness
print("\n=== Method means, grand pooled (for reference, not a paired test) ===")
all_rnd = np.concatenate([np.array(data[a][ds][(w,k)])[:,0] for a in archs for ds in datasets for (w,k) in data[a][ds]])
all_f0  = np.concatenate([np.array(data[a][ds][(w,k)])[:,1] for a in archs for ds in datasets for (w,k) in data[a][ds]])
all_f1  = np.concatenate([np.array(data[a][ds][(w,k)])[:,2] for a in archs for ds in datasets for (w,k) in data[a][ds]])
all_ts  = np.concatenate([np.array(data[a][ds][(w,k)])[:,3] for a in archs for ds in datasets for (w,k) in data[a][ds]])
all_mg  = np.concatenate([np.array(data[a][ds][(w,k)])[:,4] for a in archs for ds in datasets for (w,k) in data[a][ds]])
all_sca = np.concatenate([np.array(data[a][ds][(w,k)])[:,5] for a in archs for ds in datasets for (w,k) in data[a][ds]])
all_scs = np.concatenate([np.array(data[a][ds][(w,k)])[:,6] for a in archs for ds in datasets for (w,k) in data[a][ds]])
print(f"random={all_rnd.mean():.4f} fixed0={all_f0.mean():.4f} fixed1={all_f1.mean():.4f} TS+={all_ts.mean():.4f} MGSampler={all_mg.mean():.4f} SCSampler_argmax={all_sca.mean():.4f} SCSampler_stoch={all_scs.mean():.4f}")

print("\n=== Grand total by bin_size ===")
by_bin = defaultdict(list)
for a, ds, w, k, bs, d, *_ in rows:
    by_bin[bs].append(d)
for bs in sorted(by_bin):
    all_diffs = np.concatenate(by_bin[bs])
    t, p = stats.ttest_1samp(all_diffs, 0)
    print(f"bin_size={bs}: n={len(all_diffs)} mean={all_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(all_diffs>0)}/{len(all_diffs)}")
