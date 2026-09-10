import re
from collections import defaultdict
import numpy as np
from scipy import stats

pat = re.compile(
    r"RESULT (\w+?)_valselect(?:_(autolaparo|heichole))? window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_leaky=([\d.]+) test_TSplus=([\d.]+)"
)

# data[arch][dataset][(w,k)] -> list of (valsel, leaky, tsplus)
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

files = [f"/tmp/valselect_shard{i}.log" for i in range(4)]
for fn in files:
    for line in open(fn):
        m = pat.search(line)
        if not m: continue
        arch_raw, ds, w, k, seed, winner, valsel, leaky, tsplus = m.groups()
        arch = arch_raw.replace("_native", "")
        ds = ds if ds else "cholec80"
        w, k = int(w), int(k)
        data[arch][ds][(w, k)].append((float(valsel), float(leaky), float(tsplus)))

archs = ["bigru", "tecno", "transsvnet"]
datasets = ["cholec80", "autolaparo", "heichole"]

rows = []
for arch in archs:
    for ds in datasets:
        for (w, k), vals in sorted(data[arch][ds].items()):
            if w == k: continue
            arr = np.array(vals)
            valsel, leaky, tsplus = arr[:,0], arr[:,1], arr[:,2]
            diff = tsplus - valsel
            bin_size = w // k
            rows.append((arch, ds, w, k, bin_size, diff, valsel.mean(), leaky.mean(), tsplus.mean()))

print(f"{'arch':<12}{'dataset':<12}{'w':>4}{'k':>4}{'bin':>5}{'n':>4}{'valsel':>9}{'leaky':>9}{'TS+':>9}{'diff':>9}{'p':>8}  sign")
for arch, ds, w, k, bs, diff, vm, lm, tm in rows:
    if len(diff) >= 2:
        t, p = stats.ttest_1samp(diff, 0)
        pstr = f"{p:.4f}"
    else:
        pstr = "n/a"
    print(f"{arch:<12}{ds:<12}{w:>4}{k:>4}{bs:>5}{len(diff):>4}{vm:>9.4f}{lm:>9.4f}{tm:>9.4f}{diff.mean():>+9.4f}{pstr:>8}  {sum(diff>0)}/{len(diff)}")

print("\n=== Highlight: main setting (Cholec80, BiGRU, window=16, topk=8) ===")
for arch, ds, w, k, bs, diff, vm, lm, tm in rows:
    if arch=="bigru" and ds=="cholec80" and w==16 and k==8:
        t,p = stats.ttest_1samp(diff, 0)
        print(f"valsel_mean={vm:.4f} leaky_mean={lm:.4f} TS+_mean={tm:.4f} diff={diff.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(diff>0)}/{len(diff)}")
        print(f"per-seed diffs: {diff.round(4)}")

print("\n=== Pooled by architecture (all datasets, all bin_sizes) ===")
for arch in archs:
    all_diffs = np.concatenate([d for a,ds,w,k,bs,d,vm,lm,tm in rows if a==arch])
    t,p = stats.ttest_1samp(all_diffs, 0)
    print(f"{arch}: n={len(all_diffs)} mean={all_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(all_diffs>0)}/{len(all_diffs)}")

print("\n=== Pooled by dataset (all architectures, all bin_sizes) ===")
for ds in datasets:
    all_diffs = np.concatenate([d for a,dss,w,k,bs,d,vm,lm,tm in rows if dss==ds])
    t,p = stats.ttest_1samp(all_diffs, 0)
    print(f"{ds}: n={len(all_diffs)} mean={all_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(all_diffs>0)}/{len(all_diffs)}")

print("\n=== Pooled by bin_size (all architectures, all datasets) ===")
by_bin = defaultdict(list)
for a,ds,w,k,bs,d,vm,lm,tm in rows:
    by_bin[bs].append(d)
for bs in sorted(by_bin):
    all_diffs = np.concatenate(by_bin[bs])
    t,p = stats.ttest_1samp(all_diffs, 0)
    print(f"bin_size={bs}: n={len(all_diffs)} mean={all_diffs.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(all_diffs>0)}/{len(all_diffs)}")

grand = np.concatenate([d for a,ds,w,k,bs,d,vm,lm,tm in rows])
t,p = stats.ttest_1samp(grand, 0)
print(f"\n=== GRAND TOTAL ===")
print(f"n={len(grand)} mean={grand.mean():+.4f} t={t:.3f} p={p:.4f} sign={sum(grand>0)}/{len(grand)}")

# Also compare against the OLD leaky methodology on the same data, for direct comparison
grand_leaky_style = np.concatenate([tsplus_arr - leaky_arr for a,ds,w,k,bs,d,vm,lm,tm,leaky_arr,tsplus_arr in
                                     [(a,ds,w,k,bs,d,vm,lm,tm, np.array([x[1] for x in data[a][ds][(w,k)]]), np.array([x[2] for x in data[a][ds][(w,k)]])) for a,ds,w,k,bs,d,vm,lm,tm in rows]])
t2,p2 = stats.ttest_1samp(grand_leaky_style, 0)
print(f"\n(for comparison, same data but using the OLD leaky test-set-selected baseline instead:)")
print(f"n={len(grand_leaky_style)} mean={grand_leaky_style.mean():+.4f} t={t2:.3f} p={p2:.4f} sign={sum(grand_leaky_style>0)}/{len(grand_leaky_style)}")
