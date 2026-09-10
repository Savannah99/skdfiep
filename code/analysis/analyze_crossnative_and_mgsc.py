import re
from collections import defaultdict
import numpy as np
from scipy import stats

archs = ["bigru", "tecno", "transsvnet"]
datasets = ["cholec80", "autolaparo", "heichole"]

# ============ Part 1: crossover vs native, MobileNet-TS+ signal ============
# existing (original 324-run): bigru=crossover, tecno/transsvnet=native
pat_orig = re.compile(
    r"RESULT (\w+?)_(?:native_)?valselect(?:_(autolaparo|heichole))? window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_leaky=([\d.]+) test_TSplus=([\d.]+)"
)
# new (crossnative batch): bigru_native_valselect_DS, tecno_crossover_valselect_DS, transsvnet_crossover_valselect_DS
pat_new = re.compile(
    r"RESULT (\w+?)_(native|crossover)_valselect_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_TSplus=([\d.]+)"
)

ts = defaultdict(lambda: defaultdict(dict))  # ts[arch][protocol][(ds,w,k,seed)] = tsplus

for fn in [f"/tmp/valselect_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_orig.search(line)
        if not m: continue
        arch, ds, w, k, seed, winner, valsel, leaky, tsplus = m.groups()
        ds = ds if ds else "cholec80"
        protocol = "crossover" if arch == "bigru" else "native"
        ts[arch][protocol][(ds, int(w), int(k), int(seed))] = float(tsplus)

for fn in [f"/tmp/crossnative_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_new.search(line)
        if not m: continue
        arch, protocol, ds, w, k, seed, winner, valsel, tsplus = m.groups()
        ts[arch][protocol][(ds, int(w), int(k), int(seed))] = float(tsplus)

print("=" * 90)
print("PART 1: Crossover vs Native (MobileNet-TS+ signal) -- 2x3 matrix")
print("=" * 90)
for arch in archs:
    keys = sorted(set(ts[arch]["crossover"]) & set(ts[arch]["native"]))
    cross = np.array([ts[arch]["crossover"][k] for k in keys])
    native = np.array([ts[arch]["native"][k] for k in keys])
    diff = native - cross  # positive = native better
    t, p = stats.ttest_1samp(diff, 0)
    print(f"\n{arch}: n={len(keys)}  crossover_mean={cross.mean():.4f}  native_mean={native.mean():.4f}  "
          f"diff(native-cross)={diff.mean():+.4f}  t={t:.3f} p={p:.4f}  native_wins={sum(diff>0)}/{len(diff)}")
    for ds in datasets:
        sub = [i for i, k in enumerate(keys) if k[0] == ds]
        if not sub: continue
        d = diff[sub]
        t2, p2 = stats.ttest_1samp(d, 0) if len(d) >= 2 else (float('nan'), float('nan'))
        print(f"    {ds}: n={len(d)} diff={d.mean():+.4f} p={p2:.4f} native_wins={sum(d>0)}/{len(d)}")

# ============ Part 2: MGSampler / SCSampler significance vs val-selected baseline ============
print("\n" + "=" * 90)
print("PART 2: MGSampler / SCSampler vs validation-selected baseline, full grid")
print("=" * 90)

pat_bigru_mgsc = re.compile(
    r"RESULT bigru_mgsc_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_MG=([\d.]+) test_SC=([\d.]+)"
)
pat_arch_method = re.compile(
    r"RESULT (tecno|transsvnet)_(mg|sc)native_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_(?:MG|SC)=([\d.]+)"
)

data = defaultdict(lambda: defaultdict(list))  # data[(arch,method)][ds] -> list of (valsel, method_score)

for fn in [f"/tmp/mgsc_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_bigru_mgsc.search(line)
        if m:
            ds, w, k, seed, winner, valsel, mg, sc = m.groups()
            data[("bigru", "mg")][ds].append((float(valsel), float(mg)))
            data[("bigru", "sc")][ds].append((float(valsel), float(sc)))
            continue
        m = pat_arch_method.search(line)
        if m:
            arch, method, ds, w, k, seed, winner, valsel, score = m.groups()
            data[(arch, method)][ds].append((float(valsel), float(score)))

for method, label in [("mg", "MGSampler"), ("sc", "SCSampler-argmax")]:
    print(f"\n--- {label} ---")
    all_diffs = []
    for arch in archs:
        arch_diffs = []
        for ds in datasets:
            vals = data[(arch, method)][ds]
            if not vals: continue
            arr = np.array(vals)
            diff = arr[:, 1] - arr[:, 0]
            arch_diffs.extend(diff)
            all_diffs.extend(diff)
        arch_diffs = np.array(arch_diffs)
        if len(arch_diffs) >= 2:
            t, p = stats.ttest_1samp(arch_diffs, 0)
            print(f"  {arch}: n={len(arch_diffs)} mean={arch_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(arch_diffs>0)}/{len(arch_diffs)}")
    all_diffs = np.array(all_diffs)
    t, p = stats.ttest_1samp(all_diffs, 0)
    print(f"  GRAND TOTAL: n={len(all_diffs)} mean={all_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(all_diffs>0)}/{len(all_diffs)}")
