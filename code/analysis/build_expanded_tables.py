import re
from collections import defaultdict
import numpy as np

archs = ["bigru", "tecno", "transsvnet"]
arch_label = {"bigru": "BiGRU", "tecno": "TeCNO", "transsvnet": "Trans-SVNet"}
datasets = ["cholec80", "autolaparo", "heichole"]
ds_label = {"cholec80": "Cholec80", "autolaparo": "AutoLaparo", "heichole": "HeiChole"}
configs = [(8,1),(8,2),(8,4),(16,1),(16,2),(16,4),(16,8),(32,1),(32,2),(32,4),(32,8),(32,16)]

def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")

# ---------- A. Main 324-grid ----------
pat_orig = re.compile(
    r"RESULT (\w+?)_(?:native_)?valselect(?:_(autolaparo|heichole))? window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_leaky=([\d.]+) test_TSplus=([\d.]+)"
)
mainA = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for fn in [f"/tmp/valselect_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_orig.search(line)
        if not m: continue
        arch, ds, w, k, seed, winner, valsel, leaky, tsplus = m.groups()
        ds = ds if ds else "cholec80"
        mainA[arch][ds][(int(w), int(k))].append((float(valsel), float(leaky), float(tsplus)))

print("### A(展开). 主实验：每个(window,top_k)配置 平均macro-Jaccard(3种子)\n")
for arch in archs:
    print(f"\n**{arch_label[arch]}**\n")
    print("| 数据集 | window | top_k | bin_size | 验证集选基线 | MobileNet-TS+ | 差异 |")
    print("|---|---|---|---|---|---|---|")
    for ds in datasets:
        for (w, k) in configs:
            vals = mainA[arch][ds].get((w, k), [])
            if not vals: continue
            arr = np.array(vals)
            valsel, ts = mean(arr[:,0]), mean(arr[:,2])
            print(f"| {ds_label[ds]} | {w} | {k} | {w//k} | {valsel:.4f} | {ts:.4f} | {ts-valsel:+.4f} |")

# ---------- B. MGSampler / SCSampler ----------
pat_bigru_mgsc = re.compile(
    r"RESULT bigru_mgsc_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_MG=([\d.]+) test_SC=([\d.]+)"
)
pat_arch_method = re.compile(
    r"RESULT (tecno|transsvnet)_(mg|sc)native_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"winner=(\S+) test_valsel=([\d.]+) test_(?:MG|SC)=([\d.]+)"
)
mgsc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
for fn in [f"/tmp/mgsc_shard{i}.log" for i in range(4)]:
    for line in open(fn):
        m = pat_bigru_mgsc.search(line)
        if m:
            ds, w, k, seed, winner, valsel, mg, sc = m.groups()
            mgsc["bigru"]["mg"][ds][(int(w), int(k))].append((float(valsel), float(mg)))
            mgsc["bigru"]["sc"][ds][(int(w), int(k))].append((float(valsel), float(sc)))
            continue
        m = pat_arch_method.search(line)
        if m:
            arch, method, ds, w, k, seed, winner, valsel, score = m.groups()
            mgsc[arch][method][ds][(int(w), int(k))].append((float(valsel), float(score)))

print("\n\n### B(展开). MGSampler / SCSampler-argmax：每个(window,top_k)配置 平均macro-Jaccard(3种子)\n")
for arch in archs:
    print(f"\n**{arch_label[arch]}**\n")
    print("| 数据集 | window | top_k | bin_size | 验证集选基线 | MGSampler | SCSampler-argmax |")
    print("|---|---|---|---|---|---|---|")
    for ds in datasets:
        for (w, k) in configs:
            mg_vals = mgsc[arch]["mg"][ds].get((w, k), [])
            sc_vals = mgsc[arch]["sc"][ds].get((w, k), [])
            if not mg_vals: continue
            mg_arr, sc_arr = np.array(mg_vals), np.array(sc_vals)
            print(f"| {ds_label[ds]} | {w} | {k} | {w//k} | {mean(mg_arr[:,0]):.4f} | {mean(mg_arr[:,1]):.4f} | {mean(sc_arr[:,1]):.4f} |")

# ---------- D. All-frame training ----------
pat_af = re.compile(
    r"RESULT (\w+?)_allframe_(cholec80|autolaparo|heichole) window=(\d+) topk=(\d+) seed=(\d+)\s+"
    r"random=([\d.]+) fixed0=([\d.]+) fixed1=([\d.]+) TSplus=([\d.]+) MGSampler=([\d.]+) "
    r"SCSampler_argmax=([\d.]+) SCSampler_stoch=([\d.]+)"
)
af = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for i in range(4):
    for line in open(f"/tmp/allframe_multi_shard{i}.log"):
        m = pat_af.search(line)
        if not m: continue
        arch, ds, w, k, seed, rnd, f0, f1, ts, mg, sca, scs = m.groups()
        af[arch][ds][(int(w), int(k))].append(tuple(map(float, (rnd, f0, f1, ts, mg, sca, scs))))

print("\n\n### D(展开). 全帧训练消融：每个(window,top_k)配置 平均macro-Jaccard(3种子)\n")
for arch in archs:
    print(f"\n**{arch_label[arch]}**\n")
    print("| 数据集 | window | top_k | bin_size | Random | Fixed0 | Fixed1 | TS+ | MGSampler | SCSampler-argmax | SCSampler-stoch |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for (w, k) in configs:
            vals = af[arch][ds].get((w, k), [])
            if not vals: continue
            arr = np.array(vals)
            print(f"| {ds_label[ds]} | {w} | {k} | {w//k} | {mean(arr[:,0]):.4f} | {mean(arr[:,1]):.4f} | {mean(arr[:,2]):.4f} | "
                  f"{mean(arr[:,3]):.4f} | {mean(arr[:,4]):.4f} | {mean(arr[:,5]):.4f} | {mean(arr[:,6]):.4f} |")
