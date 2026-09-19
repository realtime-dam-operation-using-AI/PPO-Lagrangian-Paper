# Tutorial notebooks

> 🇰🇷 한국어 버전: [README_kor.md](README_kor.md)

Hands-on notebooks for understanding the paper *"A Two-Layer Safe Reinforcement Learning Framework for Multi-Constraint Reservoir Operation"* and the released training configs. The paper's environment and policy code are private, so the notebooks use a small, transparent **toy reservoir environment** (`toy_reservoir.py`) that reproduces the same mechanisms: multi-constraint reward, `CONSTRAINT_TYPES`, action masking, non-linear action mapping and the normalised constraint cost used by PPO-Lagrangian.

## Setup

```bash
bash tutorial_nbs/setup_env.sh      # creates conda env "ppo_rl" (Python 3.10) and registers a Jupyter kernel
conda activate ppo_rl
jupyter lab tutorial_nbs            # select the "Python (ppo_rl)" kernel
```

The script installs everything `DRL_training_config/*_config.py` imports (`easydict`, `pytz`, `torch`, `gym==0.25.1`, `DI-engine==0.5.3`) plus `pandas`, `openpyxl`, `matplotlib`, `jupyter`. PyTorch is installed from the CUDA 12.8 wheel index so it works on RTX 50-series GPUs; it falls back to CPU automatically. Python 3.10 is used because DI-engine 0.5.3 supports 3.7–3.10 and pins `gym==0.25.1` and `numpy<2`.

To re-execute every notebook from the command line:

```bash
conda activate ppo_rl && cd tutorial_nbs
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$nb"; done
```

## Notebooks

| # | Notebook | What you learn |
|---|---|---|
| 01 | `01_config_anatomy.ipynb` | Load the real `*_config.py` files with the private `agent` package stubbed out, tabulate the hyper-parameters, recompute "300 episodes", show the single-switch chain between the 10 variants, compare with DI-engine `PPOPolicy` defaults. |
| 02 | `02_reservoir_environment.ipynb` | Water balance, synthetic inflow, the 7 reward terms, the 6-d observation (type 5), and what `CONSTRAINT_TYPES` 1/2/3 do. |
| 03 | `03_action_mask_and_mapping.ipynb` | `basic` vs `enhanced` action masks (Experiment B) and linear vs square/sqrt/cubic/exp/log action grids (Experiment C). |
| 04 | `04_ppo_vs_ppo_lagrangian.ipynb` | A compact PyTorch PPO and PPO-Lagrangian (primal–dual) implementation using the config's `policy.lagrangian` parameters; trains A1, A2, A3, B3, C2 analogues on the toy environment and compares return, constraint cost and the multiplier λ. |
| 05 | `05_appendix_data.ipynb` | Reads the released Appendix E (30-year stress test of `C2_Square_R1`) and Appendix F (Xiluodu / Xiangjiaba) logs with English column names and plots the learned operating pattern. |

Notebook 04 takes a few minutes with the default `N_ITERS = 40`; raise it for better-converged curves.

## Applying the framework to a real dam

`operation_guide.md` is a step-by-step procedure (purpose, required data and conditions, environment calibration, mask design, training, evaluation protocol, decision-support deployment, monitoring) with runnable analysis snippets.

## Files

- `toy_reservoir.py` — the toy environment and helpers shared by notebooks 02–04. Not the paper's environment.
- `setup_env.sh`, `requirements.txt` — environment definition.
- `operation_guide.md` / `operation_guide_kor.md` — real-dam application procedure.

## Can I run the real configs here?

No. See `../DRL_training_config/description.md`. With the `ppo_rl` env active, `python A3_Lagrangian_R1_config.py --seed 0` now gets past the third-party imports and fails at `ModuleNotFoundError: No module named 'agent'`, which is the private codebase.
