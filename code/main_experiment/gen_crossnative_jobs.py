configs = [(8,1),(8,2),(8,4),(16,1),(16,2),(16,4),(16,8),(32,1),(32,2),(32,4),(32,8),(32,16)]
seeds = [0, 1, 2]
datasets = ["cholec80", "autolaparo", "heichole"]
PY = "/home/luning/miniconda3/envs/medai/bin/python"

jobs = []
for script in ["grid_bigru_native_valselect.py", "grid_tecno_crossover_valselect.py", "grid_transsvnet_crossover_valselect.py"]:
    for ds in datasets:
        for (w,k) in configs:
            for s in seeds:
                jobs.append(f"{PY} {script} {ds} {w} {k} {s}")

print(f"Total jobs: {len(jobs)}")
N_SHARDS = 4
shards = [[] for _ in range(N_SHARDS)]
for i, job in enumerate(jobs):
    shards[i % N_SHARDS].append(job)

for i, shard in enumerate(shards):
    lines = ["#!/bin/bash", "cd /tmp", f'LOG=/tmp/crossnative_shard{i}.log', ': > "$LOG"']
    for cmd in shard:
        lines.append(f'echo "=== {cmd} ===" >> "$LOG"')
        lines.append(f'{cmd} >> "$LOG" 2>&1')
    lines.append(f'echo "SHARD{i} DONE" >> "$LOG"')
    with open(f"/tmp/crossnative_shard{i}.sh", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"shard {i}: {len(shard)} jobs -> /tmp/crossnative_shard{i}.sh")
