

# DRL training configuration files (for public release)

> 🇰🇷 한국어 버전: [README_kor.md](README_kor.md)

This folder contains the **training configuration files** used in the experiments of our manuscript:

> **A Two-Layer Safe Reinforcement Learning Framework for Multi-Constraint Reservoir Operation: Integrating Action Masking and Lagrangian Dual Method**

We release these configs to improve **implementation transparency and reproducibility** (Reviewer comment on implementation details).

## What is included

```
AppE_ExtremeStressEval_C2Square_R1_Episode2.xlsx
AppF_CrossReservoir_DecisionLogs_Public_Xiluodu_Xiangjiaba.xlsx
DRL_training_config/
	A1_Correction_R*_config.py
	A2_Penalty_R*_config.py
	A3_Lagrangian_R*_config.py
	B2_WaterMask_R*_config.py
	B3_FullMask_R*_config.py
	C2_Square_R*_config.py
	C3_Sqrt_R*_config.py
	C4_Cubic_R*_config.py
	C5_Exp_R*_config.py
	C6_Log_R*_config.py
```

Each `*_config.py` is a self-contained DI-engine style configuration (Python + `EasyDict`). Most configs also include an optional `__main__` entry so they can be launched directly.

The `AppE_ExtremeStressEval_C2Square_R1_Episode2.xlsx` is the supplementary stress-test evaluation table reported in Appendix E (single run example). It contains two sheets:

- Sheet 1, `Three Gorges`: daily scheduling-process log (time series records);
- Sheet 2, `Summary`: key aggregate indicators used in the appendix summary table.

The `AppF_CrossReservoir_DecisionLogs_Public_Xiluodu_Xiangjiaba.xlsx` contains the cross-reservoir decision logs reported in Appendix F.

## How to interpret filenames

Filename pattern:

```
<Group>_<Variant>_R<k>_config.py
```

- `Group` corresponds to the **paper experiment group**:
	- **A***: constraint-handling comparison
	- **B***: action masking comparison
	- **C***: non-linear action discretization/mapping comparison
- `R1`–`R5` denote the **five independent runs** reported in the paper (five random seeds).

### Mapping to paper settings

| Paper item | This folder | Key switches in config |
|---|---|---|
| A1 (Correction) | `A1_Correction_R*_config.py` | `USE_PPO_LAGRANGIAN=False` (or policy type = PPO) + environment correction enabled |
| A2 (Penalty) | `A2_Penalty_R*_config.py` | fixed penalty / reward shaping baseline |
| A3 (PPO-Lagrangian) | `A3_Lagrangian_R*_config.py` | `USE_PPO_LAGRANGIAN=True` |
| B2 (BaseMask) | `B2_WaterMask_R*_config.py` | `ENABLE_ACTION_MASK=True`, `MASK_RULE_TYPE='basic'` |
| B3 (FullMask) | `B3_FullMask_R*_config.py` | `ENABLE_ACTION_MASK=True`, `MASK_RULE_TYPE='enhanced'` |
| C1 (Linear mapping) | **use `B3_FullMask_R*_config.py`** | `ENABLE_NONLINEAR_MAPPING=False` (linear baseline) |
| C2–C5 (Non-linear mappings) | `C2_*`–`C6_*` configs | `ENABLE_NONLINEAR_MAPPING=True`, `NONLINEAR_MAPPING_TYPE=...` |

Notes:
- In our codebase, the linear baseline for Experiment C is implemented by **disabling** non-linear mapping (`ENABLE_NONLINEAR_MAPPING=False`), which is why we reuse the `B3_FullMask_*` configs.
- The `C2...C6` file indices are internal naming; the mapping type is unambiguous from `NONLINEAR_MAPPING_TYPE`.

## Key implementation details captured by configs

These configs explicitly record (non-exhaustive):

### Neural network architecture

In `policy.model` (e.g., `A3_Lagrangian_R1_config.py`):

- Actor–Critic with a shared encoder (DI-engine VAC-style)
- `encoder_hidden_size_list = [512, 512, 256]`
- `actor_head_layer_num = 1`, `critic_head_layer_num = 1`
- `actor_head_hidden_size = 256`, `critic_head_hidden_size = 256`
- Discrete action space with `action_shape = 100`

### PPO training pipeline

Examples of recorded settings:

- `collector_env_num = 5`
- `episode_num = 60` per collector environment
- `max_episode_step = 365*4` (4-year horizon with daily steps)
- `epoch_per_collect = 10`
- `batch_size = 365*4`
- `n_sample = int(max_episode_step/2 * collector_env_num)`
- `learning_rate = 0.001`, `clip_ratio = 0.2`, `discount_factor = 0.99`, `gae_lambda = 0.95`
- `entropy_weight = 0.016`, `value_weight = 0.6`
- `adv_norm = True`, `value_norm = True`

**Important clarification (episodes):**

The paper reports **total training episodes per run**. In these configs, `episode_num` is **per collector environment**. With `collector_env_num=5` and `episode_num=60`, the total number of episodes is:

$$
N_{\text{episodes,total}} = N_{\text{collector env}} \times N_{\text{episode per env}} = 5 \times 60 = 300.
$$

This is why the manuscript may refer to “300 episodes” while the config shows `MAX_EPISODE = 60`.

### Lagrangian (primal–dual) settings

The PPO-Lagrangian mechanism is controlled by the `policy.lagrangian` block, e.g.:

- `constraint_key = 'constraint_cost_normalized'`
- `constraint_threshold = 0.05`
- `learning_rate = 0.01`
- `update_interval = 4`
- `initial_penalty = 1.0`

## How to run (if you have the codebase)

These config files are designed to be used with our DI-engine based training pipeline (not included in this paper-writing folder).

If you have the corresponding codebase available, you can typically launch training with:

- Run the config directly and pass a seed:
	- `python A3_Lagrangian_R1_config.py --seed 0`

`R1`–`R5` correspond to the five independent runs; you can use any five seeds (we recommend `0,1,2,3,4` unless you maintain a different seed list).

## Tutorial

`tutorial_nbs/` contains five Jupyter notebooks (config anatomy, toy reservoir environment, action masking / non-linear mapping, PPO vs PPO-Lagrangian, appendix data) and a `setup_env.sh` that builds the `ppo_rl` conda environment with every dependency the configs import. See `tutorial_nbs/README.md`. `hourly_operation_nbs/` extends this to hourly decision support with flood/drought priority, a monthly guide band and operator preferences (see its README).

## Data and model weights

- **Historical inflow data** are subject to institutional access restrictions and are **not** included here.
- The configs reference an inflow file name (e.g., `ResInflowEnhan.xlsx`) expected by the training environment.
- The supplementary appendix-level output tables are released as:
	- `AppE_ExtremeStressEval_C2Square_R1_Episode2.xlsx`
	- `AppF_CrossReservoir_DecisionLogs_Public_Xiluodu_Xiangjiaba.xlsx`
- **Trained model checkpoints/weights** and full experiment outputs can be obtained from the corresponding author upon reasonable request (see the manuscript Data availability statement).

## GitHub release

These configuration files and supplementary data are published at:

- https://github.com/lovemevol/PPO-Lagrangian-Paper

## Contact

Please contact the corresponding author listed in the manuscript for restricted data access and trained model parameters.

