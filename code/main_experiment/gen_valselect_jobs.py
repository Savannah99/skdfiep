configs = [(8,1),(8,2),(8,4),(16,1),(16,2),(16,4),(16,8),(32,1),(32,2),(32,4),(32,8),(32,16)]
seeds = [0, 1, 2]
PY = "/home/luning/miniconda3/envs/medai/bin/python"

jobs = []
# Cholec80 (no dataset arg)
for script in ["grid_bigru_valselect.py", "grid_tecno_native_valselect.py", "grid_transsvnet_native_valselect.py"]:
    for (w,k) in configs:
        for s in seeds:
            jobs.append(f"{PY} {script} {w} {k} {s}")
# AutoLaparo / HeiChole (dataset arg)
for script in ["grid_bigru_valselect_multi.py", "grid_tecno_native_valselect_multi.py", "grid_transsvnet_native_valselect_multi.py"]:
    for ds in ["autolaparo", "heichole"]:
        for (w,k) in configs:
            for s in seeds:
                jobs.append(f"{PY} {script} {ds} {w} {k} {s}")

print(f"Total jobs: {len(jobs)}")

N_SHARDS = 4
shards = [[] for _ in range(N_SHARDS)]
for i, job in enumerate(jobs):
    shards[i % N_SHARDS].append(job)

for i, shard in enumerate(shards):
    lines = ["#!/bin/bash", "cd /tmp", f'LOG=/tmp/valselect_shard{i}.log', ': > "$LOG"']
    for cmd in shard:
        lines.append(f'echo "=== {cmd} ===" >> "$LOG"')
        lines.append(f'{cmd} >> "$LOG" 2>&1')
    lines.append(f'echo "SHARD{i} DONE" >> "$LOG"')
    with open(f"/tmp/valselect_shard{i}.sh", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"shard {i}: {len(shard)} jobs -> /tmp/valselect_shard{i}.sh")
