# Code and paper source for "Efficient Frame Sampling for Surgical Phase Recognition: A Compute-Aware Hybrid Sampler Informed by Systematic Ablation"

This repository accompanies the CMPB submission. It contains the paper source and the scripts that produced every number and figure reported in the paper.

## Contents

```
paper/                       LaTeX source, bibliography, and generated figures
code/
  models/                    MobileNet-TS+, TeCNO, and Trans-SVNet model definitions
  main_experiment/           The 324-configuration validation-set-selected-baseline grid
                             (Table 2/3 in the paper: main results + cross-architecture/
                             cross-dataset/crossover-vs-native validation)
  all_frame_ablation/        The "train on all frames" ablation (Section 4.3 robustness check)
  analysis/                  Scripts that parse raw experiment logs into the paper's tables
                             and significance tests
  figures/                   Scripts that generate the accuracy/compute frontier plot and
                             the graphical abstract from the paper's own numbers
```

## Mapping to the paper

| Paper section | Code |
|---|---|
| §3.3 MobileNet-TS+ architecture | `code/models/pretrain_mobilenet_multitask.py` (Cholec80), `code/models/finetune_mobilenet_multitask_from_cholec80*.py` (AutoLaparo/HeiChole transfer) |
| §3.3 crossover-vs-native comparison | `code/main_experiment/grid_bigru_native_valselect.py`, `grid_tecno_crossover_valselect.py`, `grid_transsvnet_crossover_valselect.py` (paired against the native/crossover runs below) |
| §4.2 Main Results (Table 2) | `code/main_experiment/grid_bigru_valselect.py` |
| §4.3 Cross-Architecture/Cross-Dataset Validation (Table 3) | `code/main_experiment/grid_{bigru,tecno,transsvnet}_*valselect*.py`, `code/main_experiment/eval_bigru_mg_sc.py`, `grid_{tecno,transsvnet}_native_mgsc.py` |
| §4.3 all-frame training robustness check | `code/all_frame_ablation/*.py` |
| Statistics reported throughout §4.3/Discussion | `code/analysis/analyze_valselect_final.py` (main significance tests), `analyze_crossnative_and_mgsc.py` (crossover-vs-native + MGSampler/SCSampler grid), `analyze_final_v2_crossover_uniform.py` (final uniform-crossover recipe numbers), `verify_sc_vs_tsplus_direct.py` (SCSampler-vs-MobileNet-TS+ direct comparison), `build_detailed_tables.py` / `build_expanded_tables.py` (per-configuration data tables) |
| Discussion, interpretability check | `code/analysis/analyze_transition_distance.py` |
| Figure: accuracy/compute frontier | `code/figures/make_frontier_plot.py` |
| Graphical abstract | `code/figures/make_graphical_abstract.py` |

## Data

Experiments use three publicly available datasets, none of which are redistributed in this repository:

- **Cholec80** -- Twinanda et al., 2017 (see paper references)
- **AutoLaparo** -- Wang et al., 2022
- **HeiChole** -- Wagner et al., 2023

Frame features (frozen ViT-Small embeddings), optical-flow magnitudes, and MobileNet-TS+ scores are precomputed and cached to disk by earlier preprocessing steps (not included here); the scripts in `code/` consume those cached `.npy` files. Absolute paths in these scripts (e.g. `/home/.../features/vit_small`) reflect the original compute environment and will need to be adapted to reproduce results elsewhere.

## Reproducing the main result

```bash
python code/main_experiment/grid_bigru_valselect.py <window> <top_k> <seed>
# e.g. python code/main_experiment/grid_bigru_valselect.py 16 8 0
```

This trains one BiGRU model under the crossover recipe and evaluates MobileNet-TS+ against the validation-selected non-learned baseline (random/fixed-position), matching one row of the 324-configuration grid reported in the paper. The full grid sweeps `window in {8,16,32}`, `top_k` dividing `window`, and `seed in {0,1,2}`; job-generation helpers for running the full sweep are in `code/main_experiment/gen_*.py`.

## Citation

If you use this code, please cite the paper (citation details to be added upon publication).
