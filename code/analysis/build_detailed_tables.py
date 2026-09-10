import re
from collections import defaultdict
import numpy as np

archs = ["bigru", "tecno", "transsvnet"]
arch_label = {"bigru": "BiGRU", "tecno": "TeCNO", "transsvnet": "Trans-SVNet"}
datasets = ["cholec80", "autolaparo", "heichole"]
ds_label = {"cholec80": "Cholec80", "autolaparo": "AutoLaparo", "heichole": "HeiChole"}

def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")

# ---------- A. Main 324-grid (valselect): baseline + TS+ ----------
pat_orig = re.compile(
    r"RESULT (\w+?)_(?:native_)?valselect(?:_(autolaparo|heichole))? window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_leaky=([\d.]+) test_TSplus=([\d.]+)"
)
mainA = defaultdict(lambda: defaultdict(list))  # mainA[arch][ds] -> list of (valsel, leaky, tsplus)
for fn in [f"/tmp/valselect_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_orig.search(line)
        if not m: continue
        arch, ds, w, k, seed, winner, valsel, leaky, tsplus = m.groups()
        ds = ds if ds else "cholec80"
        mainA[arch][ds].append((float(valsel), float(leaky), float(tsplus)))

print("### A. 主实验(324格)：各架构x数据集 平均macro-Jaccard\n")
print("| 架构 | 数据集 | n | 验证集选基线(valsel) | 测试集最优(leaky,仅参考) | MobileNet-TS+ |")
print("|---|---|---|---|---|---|")
for arch in archs:
    for ds in datasets:
        vals = mainA[arch][ds]
        arr = np.array(vals)
        print(f"| {arch_label[arch]} | {ds_label[ds]} | {len(vals)} | {mean(arr[:,0]):.4f} | {mean(arr[:,1]):.4f} | {mean(arr[:,2]):.4f} |")

# ---------- B. MGSampler / SCSampler extension ----------
pat_bigru_mgsc = re.compile(
    r"RESULT bigru_mgsc_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_MG=([\d.]+) test_SC=([\d.]+)"
)
pat_arch_method = re.compile(
    r"RESULT (tecno|transsvnet)_(mg|sc)native_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_(?:MG|SC)=([\d.]+)"
)
mgsc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # mgsc[arch][method][ds] -> (valsel, score)
for fn in [f"/tmp/mgsc_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_bigru_mgsc.search(line)
        if m:
            ds, w, k, seed, winner, valsel, mg, sc = m.groups()
            mgsc["bigru"]["mg"][ds].append((float(valsel), float(mg)))
            mgsc["bigru"]["sc"][ds].append((float(valsel), float(sc)))
            continue
        m = pat_arch_method.search(line)
        if m:
            arch, method, ds, w, k, seed, winner, valsel, score = m.groups()
            mgsc[arch][method][ds].append((float(valsel), float(score)))

print("\n### B. MGSampler / SCSampler-argmax 全网格：各架构x数据集 平均macro-Jaccard\n")
print("| 架构 | 数据集 | n | 验证集选基线 | MGSampler | SCSampler-argmax |")
print("|---|---|---|---|---|---|")
for arch in archs:
    for ds in datasets:
        mg_vals = np.array(mgsc[arch]["mg"][ds])
        sc_vals = np.array(mgsc[arch]["sc"][ds])
        n = len(mg_vals)
        baseline = mean(mg_vals[:,0]) if n else float("nan")
        print(f"| {arch_label[arch]} | {ds_label[ds]} | {n} | {baseline:.4f} | {mean(mg_vals[:,1]):.4f} | {mean(sc_vals[:,1]):.4f} |")

# ---------- C. Crossover vs Native completion matrix ----------
pat_new = re.compile(
    r"RESULT (\w+?)_(native|crossover)_valselect_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_TSplus=([\d.]+)"
)
cn = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # cn[arch][protocol][ds] -> (valsel, tsplus)
# original protocol per arch (bigru=crossover, tecno/transsvnet=native) also goes here for completeness
for arch in archs:
    orig_protocol = "crossover" if arch == "bigru" else "native"
    for ds in datasets:
        cn[arch][orig_protocol][ds] = mainA[arch][ds]  # (valsel, leaky, tsplus) -- reuse, index tsplus at [2]

for fn in [f"/tmp/crossnative_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_new.search(line)
        if not m: continue
        arch, protocol, ds, w, k, seed, winner, valsel, tsplus = m.groups()
        cn[arch][protocol][ds].append((float(valsel), None, float(tsplus)))

print("\n### C. Crossover vs Native 完整矩阵：各架构x数据集 平均macro-Jaccard (MobileNet-TS+信号)\n")
print("| 架构 | 数据集 | crossover: n | crossover TS+ | native: n | native TS+ | 差异(native-crossover) |")
print("|---|---|---|---|---|---|---|")
for arch in archs:
    for ds in datasets:
        cro = np.array(cn[arch]["crossover"][ds])
        nat = np.array(cn[arch]["native"][ds])
        cro_ts = mean(cro[:,2]) if len(cro) else float("nan")
        nat_ts = mean(nat[:,2]) if len(nat) else float("nan")
        diff = nat_ts - cro_ts
        print(f"| {arch_label[arch]} | {ds_label[ds]} | {len(cro)} | {cro_ts:.4f} | {len(nat)} | {nat_ts:.4f} | {diff:+.4f} |")

# ---------- D. All-frame training ablation (full breakdown, has every method) ----------
pat_af = re.compile(
    r"RESULT (\w+?)_allframe_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"random=([\d.]+) fixed0=([\d.]+) fixed1=([\d.]+) TSplus=([\d.]+) MGSampler=([\d.]+) "
    r"SCSampler_argmax=([\d.]+) SCSampler_stoch=([\d.]+)"
)
af = defaultdict(lambda: defaultdict(list))
for i in range(4):
    for line in open(f"/tmp/allframe_multi_shard{i}.log"):
        m = pat_af.search(line)
        if not m: continue
        arch, ds, w, k, seed, rnd, f0, f1, ts, mg, sca, scs = m.groups()
        af[arch][ds].append(tuple(map(float, (rnd, f0, f1, ts, mg, sca, scs))))

print("\n### D. 全帧训练消融：各架构x数据集 平均macro-Jaccard（推理时套用各选帧策略）\n")
print("| 架构 | 数据集 | n | Random | Fixed0 | Fixed1 | MobileNet-TS+ | MGSampler | SCSampler-argmax | SCSampler-stoch |")
print("|---|---|---|---|---|---|---|---|---|---|")
for arch in archs:
    for ds in datasets:
        vals = af[arch][ds]
        arr = np.array(vals)
        n = len(vals)
        if n == 0:
            print(f"| {arch_label[arch]} | {ds_label[ds]} | 0 | - | - | - | - | - | - | - |")
            continue
        print(f"| {arch_label[arch]} | {ds_label[ds]} | {n} | {mean(arr[:,0]):.4f} | {mean(arr[:,1]):.4f} | {mean(arr[:,2]):.4f} | "
              f"{mean(arr[:,3]):.4f} | {mean(arr[:,4]):.4f} | {mean(arr[:,5]):.4f} | {mean(arr[:,6]):.4f} |")

# ---------- Grand pooled summary row for each table ----------
print("\n### E. 全部四张表的Grand Pooled(跨架构x数据集)均值，便于横向比较\n")
print("| 数据来源 | Random | Fixed(0/1) | MobileNet-TS+ | MGSampler | SCSampler-argmax |")
print("|---|---|---|---|---|---|")

allA = np.concatenate([np.array(mainA[a][d]) for a in archs for d in datasets if mainA[a][d]])
print(f"| A.主实验(324格，valsel基线) | - | {mean(allA[:,0]):.4f}(valsel) | {mean(allA[:,2]):.4f} | - | - |")

allB_mg = np.concatenate([np.array(mgsc[a]["mg"][d]) for a in archs for d in datasets if mgsc[a]["mg"][d]])
allB_sc = np.concatenate([np.array(mgsc[a]["sc"][d]) for a in archs for d in datasets if mgsc[a]["sc"][d]])
print(f"| B.MG/SC扩展(324格，valsel基线) | - | {mean(allB_mg[:,0]):.4f}(valsel) | - | {mean(allB_mg[:,1]):.4f} | {mean(allB_sc[:,1]):.4f} |")

allD = np.concatenate([np.array(af[a][d]) for a in archs for d in datasets if af[a][d]])
print(f"| D.全帧训练消融(324格) | {mean(allD[:,0]):.4f} | {mean(allD[:,1]):.4f}/{mean(allD[:,2]):.4f} | {mean(allD[:,3]):.4f} | {mean(allD[:,4]):.4f} | {mean(allD[:,5]):.4f} |")
