"""Final significance analysis using a UNIFORM crossover recipe for all three
architectures (BiGRU, TeCNO, Trans-SVNet), replacing the native recipe
previously used for TeCNO/Trans-SVNet -- crossover was found to be
significantly better for TeCNO and statistically tied for Trans-SVNet
(see Section 8 of experiment_log.md), so no architecture benefits from
native training under rigorous full-grid testing.
"""
import re
from collections import defaultdict
import numpy as np
from scipy import stats

archs = ["bigru", "tecno", "transsvnet"]
datasets = ["cholec80", "autolaparo", "heichole"]

pat_orig = re.compile(
    r"RESULT (\w+?)_(?:native_)?valselect(?:_(autolaparo|heichole))? window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_leaky=([\d.]+) test_TSplus=([\d.]+)"
)
pat_new = re.compile(
    r"RESULT (\w+?)_(native|crossover)_valselect_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_TSplus=([\d.]+)"
)

# raw[arch][protocol][(ds,w,k,seed)] = (valsel, tsplus)
raw = defaultdict(lambda: defaultdict(dict))
for fn in [f"/tmp/valselect_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_orig.search(line)
        if not m: continue
        arch, ds, w, k, seed, winner, valsel, leaky, tsplus = m.groups()
        ds = ds if ds else "cholec80"
        protocol = "crossover" if arch == "bigru" else "native"
        raw[arch][protocol][(ds, int(w), int(k), int(seed))] = (float(valsel), float(tsplus))

for fn in [f"/tmp/crossnative_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_new.search(line)
        if not m: continue
        arch, protocol, ds, w, k, seed, winner, valsel, tsplus = m.groups()
        raw[arch][protocol][(ds, int(w), int(k), int(seed))] = (float(valsel), float(tsplus))

# UNIFORM RECIPE: crossover for all three architectures
rows = []
for arch in archs:
    for ds in datasets:
        for w, k in [(8,1),(8,2),(8,4),(16,1),(16,2),(16,4),(16,8),(32,1),(32,2),(32,4),(32,8),(32,16)]:
            per_seed = []
            for seed in [0, 1, 2]:
                key = (ds, w, k, seed)
                if key not in raw[arch]["crossover"]: continue
                valsel, tsplus = raw[arch]["crossover"][key]
                per_seed.append(tsplus - valsel)
            if len(per_seed) == 3:
                rows.append((arch, ds, w, k, w // k, np.array(per_seed)))

print(f"{'arch':<12}{'dataset':<12}{'w':>4}{'k':>4}{'bin':>5}{'n':>4}{'diff':>9}{'p':>8}  sign")
for arch, ds, w, k, bs, diff in rows:
    t, p = stats.ttest_1samp(diff, 0)
    print(f"{arch:<12}{ds:<12}{w:>4}{k:>4}{bs:>5}{len(diff):>4}{diff.mean():>+9.4f}{p:>8.4f}  {sum(diff>0)}/{len(diff)}")

print("\n=== Pooled by architecture (uniform crossover recipe) ===")
for arch in archs:
    d = np.concatenate([r[-1] for r in rows if r[0] == arch])
    t, p = stats.ttest_1samp(d, 0)
    print(f"{arch}: n={len(d)} mean={d.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(d>0)}/{len(d)}")

print("\n=== Pooled by dataset ===")
for ds in datasets:
    d = np.concatenate([r[-1] for r in rows if r[1] == ds])
    t, p = stats.ttest_1samp(d, 0)
    print(f"{ds}: n={len(d)} mean={d.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(d>0)}/{len(d)}")

print("\n=== Pooled by bin_size ===")
by_bin = defaultdict(list)
for r in rows: by_bin[r[4]].append(r[-1])
for bs in sorted(by_bin):
    d = np.concatenate(by_bin[bs])
    t, p = stats.ttest_1samp(d, 0)
    print(f"bin_size={bs}: n={len(d)} mean={d.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(d>0)}/{len(d)}")

grand = np.concatenate([r[-1] for r in rows])
t, p = stats.ttest_1samp(grand, 0)
print(f"\n=== GRAND TOTAL (uniform crossover) ===")
print(f"n={len(grand)} mean={grand.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(grand>0)}/{len(grand)}")

print("\n=== Highlight: main setting (Cholec80, BiGRU, window=16, topk=8) [unchanged] ===")
for arch, ds, w, k, bs, diff in rows:
    if arch=="bigru" and ds=="cholec80" and w==16 and k==8:
        t,p = stats.ttest_1samp(diff, 0)
        print(f"diff={diff.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(diff>0)}/{len(diff)}")
