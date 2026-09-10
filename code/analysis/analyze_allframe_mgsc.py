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
for i in range(4):
    for line in open(f"/tmp/allframe_multi_shard{i}.log"):
        m = pat.search(line)
        if not m: continue
        arch, ds, w, k, seed, rnd, f0, f1, ts, mg, sca, scs = m.groups()
        w, k = int(w), int(k)
        data[arch][ds][(w, k)].append((float(rnd), float(f0), float(f1), float(ts), float(mg), float(sca), float(scs)))

archs = ["bigru", "tecno", "transsvnet"]
datasets = ["cholec80", "autolaparo", "heichole"]
col = {"TS+": 3, "MGSampler": 4, "SCSampler_argmax": 5}

print("Under ALL-FRAME training, each method (argmax rule) vs fixed1, full 324-cell grid:")
for name, ci in col.items():
    print(f"\n--- {name} vs fixed1 ---")
    all_diffs = []
    for arch in archs:
        arch_diffs = []
        for ds in datasets:
            for (w, k), vals in data[arch][ds].items():
                if w == k: continue
                arr = np.array(vals)
                diff = arr[:, ci] - arr[:, 2]  # method - fixed1
                arch_diffs.extend(diff); all_diffs.extend(diff)
        arch_diffs = np.array(arch_diffs)
        t, p = stats.ttest_1samp(arch_diffs, 0)
        print(f"  {arch}: n={len(arch_diffs)} mean={arch_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(arch_diffs>0)}/{len(arch_diffs)}")
    all_diffs = np.array(all_diffs)
    t, p = stats.ttest_1samp(all_diffs, 0)
    print(f"  GRAND TOTAL: n={len(all_diffs)} mean={all_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(all_diffs>0)}/{len(all_diffs)}")

print("\n\nMethod means, grand pooled (descriptive):")
for arch in archs:
    for ds in datasets:
        for (w,k), vals in data[arch][ds].items():
            pass
all_rnd, all_f0, all_f1, all_ts, all_mg, all_sca, all_scs = ([] for _ in range(7))
for arch in archs:
    for ds in datasets:
        for (w,k), vals in data[arch][ds].items():
            if w == k: continue
            arr = np.array(vals)
            all_rnd.extend(arr[:,0]); all_f0.extend(arr[:,1]); all_f1.extend(arr[:,2])
            all_ts.extend(arr[:,3]); all_mg.extend(arr[:,4]); all_sca.extend(arr[:,5]); all_scs.extend(arr[:,6])
print(f"random={np.mean(all_rnd):.4f} fixed0={np.mean(all_f0):.4f} fixed1={np.mean(all_f1):.4f} "
      f"TS+={np.mean(all_ts):.4f} MGSampler={np.mean(all_mg):.4f} SCSampler_argmax={np.mean(all_sca):.4f} SCSampler_stoch={np.mean(all_scs):.4f}")
