# Hourly dam-operation notebooks

> 🇰🇷 한국어 버전: [README_kor.md](README_kor.md)

An hourly extension of the paper's two-layer safe-RL framework (action masking + PPO-Lagrangian) for **operational decision support**, built to an operating agency's requirements:

| # | Requirement | Where it is implemented |
|---|---|---|
| 1 | Constraint priority: flood > drought > everything else, above hydropower | Priority-ordered mask layers in `HourlyReservoirEnv.mask_intervals()`; two Lagrange multipliers (flood, drought) with different thresholds in `ppo_lagrangian.train()` |
| 2 | Hydropower is not a constraint | Energy only appears in the reward with weight `w_energy=0.05`; no cost or mask layer |
| 3 | Decision inputs: dam level, downstream control-point flow, forecast inflow from weather | 19-d observation (`HourlyReservoirEnv.OBS_NAMES`) |
| 4 | Upper / lower guide levels defined on the 1st of each month; operate inside the band | `UPPER_GUIDE`, `LOWER_GUIDE`, interpolated by `guide_levels()`; band costs |
| 5 | Operator chooses near-upper / middle / near-lower operation | `target_position ∈ [0, 1]` is a policy input; one trained model serves all preferences |
| 6 | 72-hour hourly flood forecast, daily forecasts beyond; annual stability review | `make_forecasts()` (72 h hourly + 7 daily), notebooks 04 and 05 |

Everything is synthetic. `hourly_reservoir.py` documents which tables and generators to replace with real data.

## Setup

Same conda environment as `tutorial_nbs/` (`bash tutorial_nbs/setup_env.sh`, kernel "Python (ppo_rl)"). Run the notebooks with this folder as the working directory, in order; notebook 03 writes `models/policy.pt`, which 04 and 05 load.

```bash
conda activate ppo_rl && cd hourly_operation_nbs
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"; done
```

## Notebooks

| # | Notebook | Content |
|---|---|---|
| 01 | `01_problem_setup_and_data.ipynb` | Dam characteristics, monthly guide band and operator target, synthetic hourly inflow, the two forecast products and their skill, observation vector, table of real data to substitute |
| 02 | `02_hourly_environment_and_rule_baseline.ipynb` | Priority-ordered mask with nearest-boundary relaxation (worked flood example), cost structure, rule-based target-tracking baseline over a year for three operator preferences, flood-event zoom |
| 03 | `03_train_ppo_lagrangian.ipynb` | Behaviour-cloning warm start from the rule, PPO with two Lagrange multipliers, learning curves, annual comparison with the rule |
| 04 | `04_decision_support_72h.ipynb` | Morning situation → 72 h release plan, 30-member forecast ensemble, risk probabilities, decision table, comparison across operator preferences and against the rule, deployment procedure |
| 05 | `05_annual_stability_review.ipynb` | Dry / normal / wet years × three preferences × (policy, rule); monthly stability table; forecast-error sensitivity; how to write the review |

## Modules

- `hourly_reservoir.py` — environment, forecasts, mask, rule baseline, `summarize()` (KPIs in priority order).
- `ppo_lagrangian.py` — batched PPO-Lagrangian with two multipliers, running return normalisation, BC pretraining, `load_policy()`, `PolicyAgent`.

## Key design decisions

- **Mask relaxation instead of dropping**: layers are intersected as intervals in priority order; a lower-priority layer that cannot be satisfied is relaxed to its nearest boundary, so dam safety is always honoured and the downstream / ramp limits are violated as little as possible. `mask_relaxed` in the log records every such hour.
- **Square action mapping with 60 actions**: dry-season inflows are 20–60 m³/s, so the grid needs fine resolution near the minimum release.
- **Guide band is the lowest-priority hard mask layer** (`band_hard=True`, priority 5) so requirement 4 is enforced mechanically whenever it does not conflict with safety, downstream, minimum-flow or ramp limits; it is also a soft cost. Set `band_hard=False` to keep it soft only.
- **Warm start and out-of-band starts**: a random policy empties the reservoir in a day, so PPO starts from a behaviour-cloned copy of the rule; training episodes start anywhere from 1 m below the lower guide to 2.5 m above the upper guide so the policy learns recovery after floods and droughts.
