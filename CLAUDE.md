# CLAUDE.md

> 🇰🇷 한국어 버전: [CLAUDE_kor.md](CLAUDE_kor.md) — this English file is the source of truth; both must stay in sync.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Public supplementary material for the manuscript *"A Two-Layer Safe Reinforcement Learning Framework for Multi-Constraint Reservoir Operation: Integrating Action Masking and Lagrangian Dual Method"*. It contains only:

- `DRL_training_config/*_config.py` — 50 DI-engine style training configs (10 experiment variants × 5 seeds).
- Two `.xlsx` appendix tables (Appendix E stress-test log with two sheets, `Three Gorges` daily log and `Summary`; Appendix F cross-reservoir decision logs).
- `README.md` / `README_kor.md` — the authoritative mapping from paper experiments to config files. Read it before editing configs.
- `DRL_training_config/description.md` / `description_kor.md` — per-variant explanation of the config files and why they cannot run standalone.
- `tutorial_nbs/` — five Jupyter notebooks plus `toy_reservoir.py`, a small stand-in environment that reproduces the paper's mechanisms (constraint types, action masks, non-linear mapping, constraint cost). Its README explains each notebook; `operation_guide.md` is the procedure for applying the framework to a real dam.
- `hourly_operation_nbs/` — hourly decision-support extension (`hourly_reservoir.py`, `ppo_lagrangian.py`, five notebooks) built to an agency's requirements: flood > drought priority, no hydropower constraint, monthly guide band, operator target position, 72 h hourly + daily forecasts. Notebook 03 writes `models/policy.pt`; 04 and 05 need it.

There is no build, lint, or test tooling for the configs themselves. The configs are **not runnable from this repo**: they `sys.path`-hack to `../../../` and `../../../../` expecting to live at `DI-engine/agent/reservoir_single_agent/<folder>/`, and they import `agent.tool_function.ppo_lagrangian`, `ding`, and the private reservoir environment. Inflow data (`ResInflowEnhan.xlsx`) and trained weights are also not included. Do not try to "fix" the imports or paths to make them run standalone; they are released as a faithful record of what was trained.

If the private codebase is available, the intended launch is:

```
python A3_Lagrangian_R1_config.py --seed 0
```

## Tutorial environment and notebooks

```bash
bash tutorial_nbs/setup_env.sh                 # conda env "ppo_rl" (Python 3.10) + Jupyter kernel "Python (ppo_rl)"
conda activate ppo_rl
jupyter lab tutorial_nbs
# re-execute one notebook headlessly (04 trains for a few minutes)
cd tutorial_nbs && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 01_config_anatomy.ipynb
```

- Python 3.10 is deliberate: DI-engine 0.5.3 supports 3.7–3.10 and pins `gym==0.25.1`, `numpy<2`, `setuptools<=66.1.1`. Do not upgrade to gymnasium or numpy 2.
- PyTorch comes from the CUDA 12.8 wheel index (`--index-url https://download.pytorch.org/whl/cu128`) because the dev machine has an RTX 5090.
- Notebook 01 loads the real config files by injecting stub modules for `agent`, `agent.tool_function` and `agent.tool_function.ppo_lagrangian` into `sys.modules` and running the file with `runpy.run_path(..., run_name="__tutorial__")` so the `__main__` training launch is skipped. Reuse that pattern rather than editing the configs.
- Notebooks 02–04 import `toy_reservoir.py` from the notebook's own directory (`sys.path.insert(0, ".")`), so run them with `tutorial_nbs` as the working directory. They are stored with outputs; re-execute after changing `toy_reservoir.py`.
- Notebook markdown is in Korean; code, comments and column names are English.
- `hourly_operation_nbs/` uses the same env. Its mask is interval intersection with nearest-boundary relaxation (never drop a layer); `groupby(...).level` collides with pandas' `GroupBy.level`, use `g["level"]`.

## How the configs are structured

Every config is the same ~300-line file. The only lines that differ across all 50 files are:

1. `filename_without_ext` (line ~20) — the experiment run name with its original timestamp, e.g. `A3_Lagrangian_R1_202511180027`. This is the *only* difference between `R1`–`R5` of a variant; the seed is supplied via `--seed`, not embedded in the file.
2. A handful of module-level switches near lines 60–96.

The variants form a chain of single-switch changes starting from `A3_Lagrangian` (the paper's full method baseline for groups B and C):

| Variant | Change relative to previous |
|---|---|
| `A3_Lagrangian` | `USE_PPO_LAGRANGIAN=True`, `CONSTRAINT_TYPES=3`, no mask, no nonlinear mapping |
| `A2_Penalty` | from A3: `USE_PPO_LAGRANGIAN=False` (fixed-penalty PPO) |
| `A1_Correction` | from A3: `USE_PPO_LAGRANGIAN=False`, `CONSTRAINT_TYPES=2` (correct without penalising) |
| `B2_WaterMask` | from A3: `ENABLE_ACTION_MASK=True` (`MASK_RULE_TYPE='basic'`) |
| `B3_FullMask` | from B2: `MASK_RULE_TYPE='enhanced'`. Also serves as **C1 (linear mapping)** |
| `C2_Square` | from B3: `ENABLE_NONLINEAR_MAPPING=True`, `NONLINEAR_MAPPING_TYPE='square'` |
| `C3_Sqrt` / `C4_Cubic` / `C5_Exp` / `C6_Log` | from C2: `NONLINEAR_MAPPING_TYPE` only |

Note that B and C variants keep `USE_PPO_LAGRANGIAN=True` — they are layered on top of the Lagrangian method, not on plain PPO.

Other points that matter when reading or explaining a config:

- `CONSTRAINT_TYPES`: 1 = terminate episode on violation; 2 = correct but don't penalise; 3 = correct and penalise.
- `MAX_EPISODE=60` is **per collector env**; with `CO_NUM=5` this is the "300 episodes" the paper reports.
- `POLICY_TYPE` resolves to `'ppo_lagrangian'` or `'ppo'` from `USE_PPO_LAGRANGIAN`; the `policy.lagrangian` block is always present but gated by `enable`.
- The adversarial (`ADVERSARIAL_*`), curriculum, and `OBS_NOISE_ENABLE` blocks are all **off** in every released config; they are leftover capabilities of the private codebase, not paper experiments.
- Output directory is derived by replacing `test` with `result` in the config's parent folder name.
- Inline comments are in Chinese; keep that convention when editing rather than mixing languages within a file.

## Editing conventions

- Because the 50 files are near-identical, any change to shared content must be applied to all of them (e.g. with `sed` across `DRL_training_config/*.py`), then verified with `diff` between a pair of files so that only `filename_without_ext` and the intended switches differ.
- Keep `README.md`, `README_kor.md`, and the switch table above (in both `CLAUDE.md` and `CLAUDE_kor.md`) in sync if a variant's switches change.
- Docs are maintained as English/Korean pairs (`README.md` ↔ `README_kor.md`, `CLAUDE.md` ↔ `CLAUDE_kor.md`, `description.md` ↔ `description_kor.md`, `tutorial_nbs/README.md` ↔ `tutorial_nbs/README_kor.md`, `tutorial_nbs/operation_guide.md` ↔ `operation_guide_kor.md`, `hourly_operation_nbs/README.md` ↔ `README_kor.md`). When editing one, apply the same change to the other.
