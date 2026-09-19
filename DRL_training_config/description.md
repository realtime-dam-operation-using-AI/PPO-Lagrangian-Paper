# DRL_training_config — file descriptions

> 🇰🇷 한국어 버전: [description_kor.md](description_kor.md)

This folder holds the 50 training configuration files used for the paper's experiments: 10 experiment variants × 5 independent runs (`R1`–`R5`). Every file is a DI-engine style config (Python + `EasyDict`) that builds two objects, `main_config` and `create_config`, and hands them to the training pipeline when run as `__main__`.

## Common structure

All 50 files share the same body. Reading from top to bottom:

| Section | What it defines |
|---|---|
| Imports and `sys.path` setup | Adds `../../../` and `../../../../` to the path so the file can import the private DI-engine codebase (`agent.*`, `ding`). Registers the custom policy `agent.tool_function.ppo_lagrangian` and the Gym environment `ResevoirAgent-v0`. |
| `filename_without_ext` | The run name with its original training timestamp. This is the only line that differs between `R1`–`R5` of the same variant. |
| Training scale constants | `CO_NUM=5` collector envs, `MAX_EPISODE=60` per env (300 total), `MAX_EPISODE_STEP=365*4` daily steps, `N_SAMPLE`, `BATCH_SIZE`, `UPDATE_NUM=10`, `LR=0.001`. |
| Reward coefficients | `POWER_`, `LEVEL_`, `FLOOD_`, `ECOLOGICAL_`, `POWERLIMIT_`, `NAVIGATION_`, `ACTION_REWARD_COEFF`, all `1.0`. |
| PPO-Lagrangian block | `USE_PPO_LAGRANGIAN`, threshold `0.05`, multiplier LR `0.01`, initial penalty `1.0`, update every 4 minibatches, ReLU on positive violations only. |
| Experiment switches | `CONSTRAINT_TYPES`, `OBSERVATION_TYPES=5`, `ACTION_SPACE_TYPE='discrete'` with `ACTION_SHAPE=100`, `ENABLE_ACTION_MASK`, `MASK_RULE_TYPE`, `ENABLE_NONLINEAR_MAPPING`, `NONLINEAR_MAPPING_TYPE`. |
| Disabled extras | `ADVERSARIAL_*`, `CURRICULUM_LEARNING`, `OBS_NOISE_ENABLE` are all `False`. They exist in the private codebase but were not used in the paper. |
| `main_config` | `env` (episode counts, reward coefficients, inflow file `ResInflowEnhan.xlsx`, `experimental_setup`, data export, TensorBoard) and `policy` (model sizes, PPO learn/collect/eval settings, `lagrangian` block). |
| `create_config` | Environment type `resevoiragent-v2`, `base` env manager, and policy type `ppo_lagrangian` or `ppo`. |
| `__main__` | Parses `--seed`, picks the policy class, and calls `resrevoiragent_pipeline_onpolicy([main_config, create_config], seed=...)`. |

Model: shared-encoder Actor–Critic, encoder `[512, 512, 256]`, one hidden layer of 256 in each head, 6-dimensional observation, 100 discrete actions.

## The 10 variants

The variants form a chain: each one changes a single switch relative to the one before it. `A3_Lagrangian` is the base for groups B and C.

### Group A — constraint handling (paper Experiment A)

| File prefix | `USE_PPO_LAGRANGIAN` | `CONSTRAINT_TYPES` | Meaning |
|---|---|---|---|
| `A1_Correction` | `False` | `2` | Plain PPO. When an action would violate a constraint the environment corrects it but applies no penalty. |
| `A2_Penalty` | `False` | `3` | Plain PPO. The environment corrects the action and adds a fixed penalty to the reward (reward-shaping baseline). |
| `A3_Lagrangian` | `True` | `3` | PPO-Lagrangian. Same correction and penalty signal, but the penalty weight is a Lagrange multiplier updated by the dual method to keep the normalised constraint cost below `0.05`. |

`CONSTRAINT_TYPES` values: 1 = terminate the episode on violation (unused), 2 = correct without penalty, 3 = correct and penalise.

### Group B — action masking (paper Experiment B)

Both build on `A3_Lagrangian` (`USE_PPO_LAGRANGIAN=True`, `CONSTRAINT_TYPES=3`).

| File prefix | `ENABLE_ACTION_MASK` | `MASK_RULE_TYPE` | Meaning |
|---|---|---|---|
| `B2_WaterMask` | `True` | `'basic'` | Masks out actions that violate basic water-level / flow limits before sampling. Paper "BaseMask". |
| `B3_FullMask` | `True` | `'enhanced'` | Adds the Three Gorges operating-rule constraints to the mask. Paper "FullMask". Also used as the **C1 linear-mapping** baseline. |

The B1 (no mask) setting of the paper is `A3_Lagrangian`.

### Group C — non-linear action mapping (paper Experiment C)

All build on `B3_FullMask` and set `ENABLE_NONLINEAR_MAPPING=True`. The 100 discrete actions are mapped onto the continuous release range through the chosen function instead of linearly.

| File prefix | `NONLINEAR_MAPPING_TYPE` |
|---|---|
| `C2_Square` | `'square'` |
| `C3_Sqrt` | `'sqrt'` |
| `C4_Cubic` | `'cubic'` |
| `C5_Exp` | `'exp'` |
| `C6_Log` | `'log'` |

The linear baseline (paper C1) is `B3_FullMask` with `ENABLE_NONLINEAR_MAPPING=False`; there is no separate `C1_*` file.

### Runs `R1`–`R5`

The five files of a variant are identical except for `filename_without_ext`. The random seed is not stored in the file; it is passed on the command line (`--seed 0` … `--seed 4`).

## Can these files be run on their own?

**No, not from this repository.** Each file is an entry point into a private DI-engine based training pipeline, and it needs all of the following, none of which are included here:

1. Python packages `easydict`, `pytz`, `torch`, `gym`, and DI-engine (`ding`).
2. The private package `agent.*` located two directories above this folder:
   - `agent.tool_function.ppo_lagrangian` — the PPO-Lagrangian policy class
   - `agent.reservoir_single_agent.envs.reservoir_env_gym` and `reservoir_env_v2` — the reservoir simulation environment (water balance, hydropower, constraints, masking, non-linear mapping)
   - `agent.reservoir_single_agent.entry.resevoiragent_entry_onpolicy` — the training loop
3. The historical inflow file `ResInflowEnhan.xlsx`, which is under institutional access restriction.

Running a file here stops at the first line with `ModuleNotFoundError: No module named 'easydict'`, and after installing that it would fail on `from agent.tool_function import ppo_lagrangian`. To actually train, place these files inside the original codebase at `DI-engine/agent/reservoir_single_agent/<folder>/`, obtain the inflow data, and run `python <config>.py --seed <k>`.
