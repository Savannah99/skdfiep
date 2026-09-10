"""Directly compute SCSampler-argmax MINUS MobileNet-TS+ (paired by config),
the true grid-wide analog of the single-setting '0.22 points' gap in
Table tab:main -- rather than comparing two differently-defined quantities
(SC-vs-baseline and TS+-vs-baseline) as if they were the same thing.
"""
import re
from collections import defaultdict
import numpy as np
from scipy import stats

archs = ["bigru", "tecno", "transsvnet"]
datasets = ["cholec80", "autolaparo", "heichole"]

# TS+ under the FINAL uniform-crossover recipe (same source as analyze_final_v2)
pat_orig = re.compile(
    r"RESULT (\w+?)_(?:native_)?valselect(?:_(autolaparo|heichole))? window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_leaky=([\d.]+) test_TSplus=([\d.]+)"
)
pat_new = re.compile(
    r"RESULT (\w+?)_(native|crossover)_valselect_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_TSplus=([\d.]+)"
)
raw = defaultdict(lambda: defaultdict(dict))
for fn in [f"/tmp/valselect_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_orig.search(line)
        if not m: continue
        arch, ds, w, k, seed, winner, valsel, leaky, tsplus = m.groups()
        ds = ds if ds else "cholec80"
        protocol = "crossover" if arch == "bigru" else "native"
        raw[arch][protocol][(ds, int(w), int(k), int(seed))] = float(tsplus)
for fn in [f"/tmp/crossnative_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_new.search(line)
        if not m: continue
        arch, protocol, ds, w, k, seed, winner, valsel, tsplus = m.groups()
        raw[arch][protocol][(ds, int(w), int(k), int(seed))] = float(tsplus)

tsplus_final = {}  # (arch,ds,w,k,seed) -> TS+ score, uniform crossover
for arch in archs:
    for key, val in raw[arch]["crossover"].items():
        tsplus_final[(arch,) + key] = val

# SC-argmax, per-arch best-available protocol (bigru=crossover-reuse, tecno/transsvnet=native)
pat_bigru_mgsc = re.compile(
    r"RESULT bigru_mgsc_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_MG=([\d.]+) test_SC=([\d.]+)"
)
pat_arch_method = re.compile(
    r"RESULT (tecno|transsvnet)_(mg|sc)native_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_(?:MG|SC)=([\d.]+)"
)
sc_scores = {}
for fn in [f"/tmp/mgsc_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_bigru_mgsc.search(line)
        if m:
            ds, w, k, seed, winner, valsel, mg, sc = m.groups()
            sc_scores[("bigru", ds, int(w), int(k), int(seed))] = float(sc)
            continue
        m = pat_arch_method.search(line)
        if m:
            arch, method, ds, w, k, seed, winner, valsel, score = m.groups()
            if method == "sc":
                sc_scores[(arch, ds, int(w), int(k), int(seed))] = float(score)

# pair them
diffs = []
rows_by_arch = defaultdict(list)
for key, ts in tsplus_final.items():
    if key in sc_scores:
        d = sc_scores[key] - ts
        diffs.append(d)
        rows_by_arch[key[0]].append(d)

diffs = np.array(diffs)
print(f"Paired cells: {len(diffs)} (expect 324)")
t, p = stats.ttest_1samp(diffs, 0)
print(f"\nSC-argmax minus MobileNet-TS+ (direct, paired by config):")
print(f"grand total: n={len(diffs)} mean={diffs.mean():+.4f} t={t:.3f} p={p:.4g} sign={sum(diffs>0)}/{len(diffs)}")
for arch in archs:
    d = np.array(rows_by_arch[arch])
    t2, p2 = stats.ttest_1samp(d, 0)
    print(f"  {arch}: n={len(d)} mean={d.mean():+.4f} p={p2:.4g} sign={sum(d>0)}/{len(d)}")

print(f"\nFor comparison, main-setting single-cell gap (Table tab:main): SC(0.7565) - TS+(0.7543) = {0.7565-0.7543:+.4f}")

# Also: SC vs baseline at the main setting specifically, for a clean like-for-like check
print("\n--- Sanity: SC vs BASELINE (not TS+) at main setting vs grid-wide, for a clean apples-to-apples check ---")
main_key = ("bigru", "cholec80", 16, 8)
main_sc = [sc_scores[("bigru","cholec80",16,8,s)] for s in range(3)]
print(f"SC-argmax at main setting (3 seeds): {main_sc}, mean={np.mean(main_sc):.4f}")
print("Fixed-position (best offset) at main setting (Table tab:main): 0.7508")
print(f"SC - Fixed(best) at main setting = {np.mean(main_sc)-0.7508:+.4f}")
