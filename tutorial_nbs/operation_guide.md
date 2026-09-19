# Operation Guide: applying the framework to a real dam

> 🇰🇷 한국어 버전: [operation_guide_kor.md](operation_guide_kor.md)

This document is a step-by-step procedure for **agencies and researchers who want to apply** the framework of the paper *"A Two-Layer Safe Reinforcement Learning Framework for Multi-Constraint Reservoir Operation"* (action masking + PPO-Lagrangian) to a real reservoir. It explains in what order, and with what inputs, to use the released configs (`DRL_training_config/`), the tutorial notebooks (`tutorial_nbs/`) and the toy environment (`toy_reservoir.py`).

> **Premise:** the paper's environment and policy code are private. This guide therefore assumes you will (1) use the released config files as the training recipe and (2) **re-implement the environment for your own dam** following the structure of the toy environment. If you obtain the original codebase from the corresponding author, the config files can be reused almost unchanged from step 5 (training) onward.

---

## 0. Purpose and scope

### 0.1 What the framework does

| Item | Content |
|---|---|
| Decision | One **daily release** (or turbine discharge), chosen from 100 discrete actions |
| Objective | Maximise hydropower minus penalties for level, flood, ecological flow, guaranteed output, navigation and action change |
| Constraint handling | Layer 1: an **action mask** removes rule-violating actions before sampling (hard constraints). Layer 2: **Lagrangian dual optimisation** keeps the expected remaining violation below a threshold (soft constraints) |
| Inputs | Current level, today's inflow, forecast inflow, today's upper-limit level, date (sin/cos) |
| Outputs | Recommended release plus the allowed release range for the day (the mask) |

### 0.2 Recommended modes of use

1. **Decision support** — the operator sees the recommended release and the allowed range every morning and makes the final call. **The most realistic and recommended mode.**
2. **Rule-curve improvement** — analyse a 30-year operating log produced by the trained policy (Appendix E format) to find improvements to the existing rule curve.
3. **Scenario stress testing** — check policy behaviour under extreme drought / flood scenarios in advance.
4. **Closed-loop automatic control** — **not recommended** for regulatory and safety reasons. At minimum keep an operator-approval step and a rule-curve fallback.

### 0.3 Suitable conditions

- A single reservoir (or one whose upstream control is independent) whose main purpose is hydropower with multi-purpose constraints.
- At least 20 years of daily inflow records and codified operating rules (flood-season limit level, minimum flows, guaranteed output, ...).
- Cascade operation, hourly peaking and demand-driven water supply require extending the environment and action space and are beyond this guide.

---

## 1. Procedure overview

```
[1] Problem definition ─► [2] Data collection & QC ─► [3] Environment build & calibration ─► [4] Safety layer (mask)
                                                                                                        │
[8] Operation & retraining ◄─ [7] Decision-support deployment ◄─ [6] Evaluation & stress tests ◄─ [5] Training (PPO-Lagrangian)
```

| Step | Deliverable | Repository material |
|---|---|---|
| 1 Problem definition | Objective/constraint spec, action & observation definition | `description.md`, notebook 01 |
| 2 Data | Validated inflow file, characteristic curves, rule table | this document §3 |
| 3 Environment | `<dam>_env.py`, calibration report | `toy_reservoir.py`, notebook 02 |
| 4 Mask | Mask rule function, empty-mask tests | notebook 03 |
| 5 Training | Config files (A/B/C variants), checkpoints, TensorBoard logs | `DRL_training_config/`, notebook 04 |
| 6 Evaluation | Metric table, stress-test log (Appendix E format) | notebook 05, `AppE_*.xlsx` |
| 7 Deployment | Daily inference script, decision log | this document §8 |
| 8 Operation | Drift monitoring, retraining criteria | this document §9 |

---

## 2. Step 1 — Problem definition

Settle the table below **together with the dam operations department** before training. It becomes the environment code and the config switches.

| Decision | Paper setting | What to decide for your dam |
|---|---|---|
| Time step | 1 day | 1 day recommended. Hourly for peaking (requires redesigning state and action) |
| Episode length | 4 years (1460 days) | 2–5 hydrological years |
| Action | 100 discrete release levels | Release range `[Q_MIN, Q_MAX]`, number of steps, mapping function (§5) |
| Observation | 6-d (type 5) | Level, inflow, forecast inflow, upper-limit level, date. Add downstream level / supply demand if needed |
| Reward terms | 7 coefficients (all 1.0) | Add water supply, environmental flow etc. **Start with all 1.0** |
| Hard constraints (mask) | Level range, regulatory flows | Only items that can never be violated: dead level, flood-limit level, minimum flows, daily ramp limit |
| Soft constraints (Lagrangian) | `constraint_cost_normalized ≤ 0.05` | Items whose violation is undesirable but not catastrophic. Threshold: §6.3 |
| Constraint type | 3 (correct + penalise) | Fix at 3 for real operation |

**Deliverable:** `problem_spec.md` (the table plus the regulation clause numbers behind each row).

---

## 3. Step 2 — Required data and conditions

### 3.1 Mandatory data

| Data | Resolution / span | Example format | Use | Notes |
|---|---|---|---|---|
| **Observed inflow** | Daily, ≥20 years (30 recommended) | `date, inflow_m3s` | Training / test scenarios | ≤5% missing. Use net inflow if upstream releases interfere |
| **Inflow forecast** (or forecast-error statistics) | Daily, 1–7 day lead | `date, lead, inflow_fc` | `future_inflow` observation | Without a forecast, use a moving average of observations plus noise (cf. the configs' `obs_noise_*` options) |
| **Level–storage curve** | 0.1–0.5 m steps | `level_m, storage_1e8m3` | Water balance | Latest survey (sedimentation) |
| **Outflow–tailwater curve** | Per flow band | `outflow_m3s, tailwater_m` | Head computation | Two-variable if backwater from a downstream dam |
| **Plant characteristics** | Constants | Installed capacity, max turbine flow, minimum head, efficiency (or curve) | Energy | The tutorial's `EFFICIENCY_K=8.5` is a crude approximation |
| **Operating rules** | By date | Flood season and limit level, normal pool, dead level, minimum flows (ecological / navigation / supply), guaranteed output, daily release change limit | Mask and penalties | Build a 1:1 table to regulation clauses |
| **Historical operation** | Daily, ≥5 years | `date, level, inflow, outflow, energy` | Environment calibration, baseline | See the Appendix F format |

### 3.2 Useful additional data

- Precipitation / temperature (forecast model inputs), downstream demand and abstraction, grid price or demand (to weight energy value), evaporation and seepage losses.
- Extreme-event records (largest flood, longest drought) for stress scenarios.

### 3.3 File formats

The paper's code reads a single `ResInflowEnhan.xlsx`. Since that code is private, standardise on **three CSV files** as below.

```
data/<dam>/inflow_daily.csv         # date, inflow_m3s [, inflow_fc_1d ... inflow_fc_7d]
data/<dam>/curves.csv               # kind(level_storage|tail_outflow), x, y
data/<dam>/rules.yaml               # rule parameters (example below)
data/<dam>/historical_operation.csv # date, level_m, inflow_m3s, outflow_m3s, energy_1e8kWh
```

Example `rules.yaml`:

```yaml
dam: ExampleDam
levels: {dead: 145.0, normal: 175.0, flood_limit: 145.0}
flood_season: {start: "06-01", end: "09-30"}
release: {q_min: 4000, q_max: 40000, turbine_max: 30000, max_daily_change_ratio: 0.3}
min_flows: {ecological: 6000, navigation: 5600, water_supply: 0}
power: {capacity_mw: 22500, guaranteed_mw: 4990, efficiency_k: 8.5}
```

### 3.4 Data quality check

```python
import pandas as pd, numpy as np

def check_inflow(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    report = {
        "years": (df.index.max() - df.index.min()).days / 365.25,
        "missing_days": len(full.difference(df.index)),
        "nan_ratio": df.inflow_m3s.isna().mean(),
        "negative": int((df.inflow_m3s < 0).sum()),
        "p99_over_p50": df.inflow_m3s.quantile(.99) / df.inflow_m3s.quantile(.5),
        "monthly_mean": df.inflow_m3s.groupby(df.index.month).mean().round(0).to_dict(),
    }
    assert report["years"] >= 20, "need at least 20 years"
    assert report["nan_ratio"] < 0.05 and report["negative"] == 0
    return report
```

### 3.5 Data split

- **Training:** 70% by hydrological year (e.g. 1990–2014). Sample episode start years at random so each episode is 4 consecutive years.
- **Validation:** 15% — hyper-parameter and threshold selection.
- **Test:** 15% and **the extreme-event years must be in the test set** (they must not be in training, otherwise generalisation cannot be checked).
- **Stress:** 30-year synthetic scenarios built by reshuffling and scaling observed years (inflow ×0.6, ×1.4, ...), as in Appendix E.

### 3.6 Non-data prerequisites

| Condition | Minimum |
|---|---|
| Compute | One GPU (paper setting: 300 episodes × 1460 days × 5 envs, a few hours per variant). CPU works but is several times slower |
| Software | The `tutorial_nbs/setup_env.sh` environment (Python 3.10, DI-engine 0.5.3, torch) |
| People | One hydrology / dam-operation expert (rules, calibration), one RL engineer (environment, training), an operator representative (acceptance) |
| Institutional | Internal approval process for using recommendations, retention policy for decision logs |

---

## 4. Step 3 — Building and calibrating the environment

### 4.1 Implementation

Copy `ToyReservoirEnv` from `toy_reservoir.py` to `<dam>_env.py` and replace the following.

| Toy component | Replace with |
|---|---|
| `storage_from_level`, `level_from_storage` (linear) | Interpolation of the surveyed level–storage curve (`np.interp`) |
| `tailwater_level(q)` (linear) | Surveyed outflow–tailwater curve |
| `EFFICIENCY_K * Q * H` | Efficiency curve or turbine characteristics; no generation below minimum head |
| `synthetic_inflow` | Slices of `inflow_daily.csv` sampled by episode start date |
| `upper_limit_level(doy)` | The regulation's limit-level table by date |
| `cost_parts` entries | Your constraint list. **Normalise each to 0–1** (the paper's `cost_normalizer`) |
| `future` observation | Real forecast or a forecast-error model |

**Keep unchanged:** `info["constraint_cost_normalized"]` (Lagrangian input), `info["action_mask"]`, the meaning of `CONSTRAINT_TYPES`, observation normalisation.

### 4.2 Calibration

Feed the historical releases into the environment and check that **levels and energy are reproduced**. This is the credibility test of the environment.

```python
def replay_historical(env_cls, hist: pd.DataFrame, rules):
    """hist: date, level_m, inflow_m3s, outflow_m3s, energy_1e8kWh (actual operation records)"""
    env = env_cls(constraint_types=2)                      # no penalties needed for calibration
    env.reset(init_level=hist.level_m.iloc[0], inflow_series=hist.inflow_m3s.values)
    sim = []
    for q in hist.outflow_m3s.values:
        a = int(np.argmin(np.abs(env.grid - q)))            # closest action to the actual release
        env.step(a)
        sim.append(dict(level=env.history[-1]["level"], energy=env.history[-1]["energy"]))
    sim = pd.DataFrame(sim, index=hist.index)
    err = dict(level_rmse_m=np.sqrt(((sim.level - hist.level_m) ** 2).mean()),
               energy_bias_pct=100 * (sim.energy.sum() / hist.energy_1e8kWh.sum() - 1))
    return sim, err
```

**Suggested acceptance:** level RMSE < 0.5 m, annual energy bias < 5%. If exceeded, fix the curves, efficiency and loss terms first.

### 4.3 Unit tests

```python
def test_mass_balance(env):
    obs, info = env.reset(seed=0)
    v0 = storage_from_level(env.level)
    tot_in = tot_out = 0
    for _ in range(365):
        _, _, te, tu, info = env.step(50)
        h = env.history[-1]; tot_in += h["inflow"]; tot_out += h["release"]
        if te or tu: break
    v1 = storage_from_level(env.level)
    assert abs((v1 - v0) - (tot_in - tot_out) * 86400 / 1e8) < 1e-6
```

Also test: the level never drops below dead level, the mask is never empty, observations stay within [−1, 1].

---

## 5. Step 4 — Designing the safety layer (action mask)

### 5.1 Rule tiers

| Tier | Paper name | Rules included | Basis |
|---|---|---|---|
| basic | `MASK_RULE_TYPE='basic'` (B2) | Tomorrow's level ∈ [dead level, today's upper limit] | Physical / legal absolute limits |
| enhanced | `MASK_RULE_TYPE='enhanced'` (B3) | basic + minimum flows (ecological / navigation / supply) + daily ramp limit + flood-season pre-release rules | Operating rulebook |

The paper's results and tutorial notebooks 03 and 04 show that **enhanced mask + Lagrangian** is the most stable combination. In practice, every item written in the regulations should go into the mask.

### 5.2 Design rules

1. **Never empty:** when rules conflict, relax them by priority (safety > statutory minimum flow > ramp limit); the final fallback is "release that holds the current level". See the fallback branch in `toy_reservoir.action_mask()`.
2. **Forecast uncertainty:** compute tomorrow's level with an upper forecast bound (e.g. forecast × 1.2) so the mask is conservative.
3. **Log mask statistics:** record the number of allowed actions each day. Appendix E averaged 31/100 over 30 years. If it stays below 10 the rules are too conservative.

### 5.3 Choosing the action mapping

Use the resolution plot in notebook 03 to pick a mapping that places enough actions in **your dam's everyday release band**. The paper adopted `square` (fine resolution at low releases). If flood control dominates, also compare `sqrt` / `log`. This choice is the single config switch `NONLINEAR_MAPPING_TYPE`.

---

## 6. Step 5 — Training

### 6.1 Preparing the config file

Copy `DRL_training_config/A3_Lagrangian_R1_config.py` and change only the following.

| Item | Change |
|---|---|
| `filename_without_ext` | `<dam>_A3_R1_<timestamp>` |
| `MAX_EPISODE_STEP` | Episode length (e.g. 365×4) |
| `res_inflow_filename` | Your inflow file |
| `ACTION_SHAPE`, `Q` range | Values from §2 |
| `reward_coeffs` | All 1.0 at first |
| `lagrangian.constraint_threshold` | §6.3 |
| `ENABLE_ACTION_MASK=True`, `MASK_RULE_TYPE='enhanced'`, `NONLINEAR_MAPPING_TYPE` | Values from §5 |

Repeat the paper's experimental design (A1/A2/A3 → B2/B3 → C2–C6) **on your own dam**. The improvement of each step over the previous one is your justification for deployment. Five seeds (`--seed 0..4`) are mandatory.

### 6.2 Running and monitoring

```bash
conda activate ppo_rl
for s in 0 1 2 3 4; do python <dam>_B3_FullMask_config.py --seed $s; done
tensorboard --logdir result_<folder>/
```

Curves to watch in TensorBoard:

| Curve | Healthy | Warning sign |
|---|---|---|
| episode return | Rises then plateaus | Keeps falling → penalties too large, re-tune coefficients |
| `lagrangian/lambda` | Rises then oscillates near a level | Unbounded growth → threshold unattainable (mask / rule conflict) |
| `constraint_cost_normalized` | Converges below the threshold | Stuck above → strengthen the mask or raise the threshold |
| entropy | Decreases slowly | Collapses → increase `entropy_weight` |

### 6.3 Choosing the threshold and multiplier parameters

- `constraint_threshold` (paper 0.05): first compute the **normalised cost of rule-curve operation** on validation data and start at 50–100% of that. Zero is unattainable and makes λ diverge.
- `learning_rate` of the multiplier (paper 0.01): appropriate if λ responds within 20–50 iterations; too large causes oscillation.
- `initial_penalty` (paper 1.0): start at 2–5 if the rule-curve cost is far above the threshold.
- `dual_clip`: in real applications set an upper bound (e.g. 20) to prevent divergence.

### 6.4 Training time

The paper setting (5 envs × 60 episodes × 1460 days = 438,000 steps, 3,000 gradient steps) takes a few hours per variant and seed on one GPU. All 10 variants × 5 seeds take days, so prioritise A3 → B3 → C2.

---

## 7. Step 6 — Evaluation and stress tests

### 7.1 Mandatory metrics (same as the Appendix E summary sheet)

```python
def evaluate_log(log: pd.DataFrame, rules) -> dict:
    """log: daily operating log with columns level, inflow, release, energy, power_mw, cost_*"""
    days = len(log)
    return {
        "days": days,
        "total_energy_1e8kWh": log.energy.sum(),
        "mean_daily_energy": log.energy.mean(),
        "min_daily_energy": log.energy.min(),
        "mean_level_m": log.level.mean(),
        "total_spill_m3s": log.get("spill", pd.Series(0, index=log.index)).sum(),
        "navigation_guarantee_pct": 100 * (log.release >= rules["min_flows"]["navigation"]).mean(),
        "ecological_guarantee_pct": 100 * (log.release >= rules["min_flows"]["ecological"]).mean(),
        "power_guarantee_pct": 100 * (log.power_mw >= rules["power"]["guaranteed_mw"]).mean(),
        "flood_limit_violation_days": int((log.cost_flood > 0).sum()),
        "action_correction_pct": 100 * (log.agent_release != log.release).mean(),
        "mean_constraint_cost": log.constraint_cost.mean(),
    }
```

### 7.2 Baselines

Compare with at least two:
1. **The current rule curve** — rule-based releases on the same scenarios.
2. **Historical operation** — actual records for the test period (Appendix F format).
If possible add 3. **A perfect-information optimum** (DP / LP with known future inflow, an upper bound) and report the policy as a percentage of it.

### 7.3 Test protocol

| Test | Method | Example acceptance |
|---|---|---|
| Test years | Held-out observed years × 5 seeds | Guarantee rates ≥ rule curve, energy ≥ rule curve |
| Extreme stress | 30-year synthetic (Appendix E style), inflow ×0.6 / ×1.4 | 0 days of dead-level / limit-level violation, action violation rate < 0.1% |
| Forecast error | ±20–30% noise on the forecast (`obs_noise_*`) | Metric degradation < 5% |
| Initial conditions | 5 start levels × 4 start months | Zero violations for every combination |
| Generalisation | Another dam's data (Appendix F style) | Zero violations without retraining (optional) |
| Seed variance | Mean ± std over 5 seeds | Std < 5% of the mean |

### 7.4 Decision-log format

Adopting the 30 columns of Appendix E as they are (day, level, tailwater, storage, inflow, outflow, turbine flow, spill, energy, guaranteed-output flag, water balance, agent action, actual action, mask min/max, action violation, reward and six penalties, six observations) makes your results directly comparable with the paper. `COLS_E` in notebook 05 gives the English column names.

---

## 8. Step 7 — Decision-support deployment

### 8.1 Daily pipeline

```
06:00  Collect observations (level, yesterday's inflow) and the 1–7 day inflow forecast
06:10  Build the observation vector → compute the mask → run the policy (5-seed ensemble)
06:15  Produce the recommendation: release, allowed range, expected level and energy, violation risks
06:30  Operator review → approve / modify → actual release decision
next day  Ingest observations, store the decision log, record recommendation vs actual
```

### 8.2 Inference code

```python
import torch, numpy as np

class Recommender:
    def __init__(self, env, models, mapping="square"):
        self.env, self.models = env, models          # models: trained ActorCritic for 5 seeds

    @torch.no_grad()
    def recommend(self, level, inflow_today, inflow_fc, date):
        self.env.set_state(level=level, inflow=inflow_today, forecast=inflow_fc, date=date)
        obs, mask = self.env.observe(), self.env.action_mask()
        probs = np.mean([m(torch.as_tensor(obs), torch.as_tensor(mask))[0].probs.numpy() for m in self.models], axis=0)
        a = int(probs.argmax())
        q_rec = float(self.env.grid[a])
        q_lo, q_hi = self.env.grid[mask].min(), self.env.grid[mask].max()
        z_next = self.env.preview_level(q_rec)          # tomorrow's level from the water balance
        return dict(release_m3s=q_rec, allowed_range=(q_lo, q_hi), expected_level=z_next,
                    confidence=float(probs[a]), n_allowed=int(mask.sum()))
```

(`set_state`, `observe` and `preview_level` are methods to add to `<dam>_env.py`; reuse `_obs`, `action_mask` and `feasible_release_range` from `ToyReservoirEnv`.)

### 8.3 Operational safeguards

| Safeguard | Content |
|---|---|
| Human approval | A recommendation is executed only after operator approval. The system never controls the outlet works directly |
| Fallback | Missing observation, empty mask, model error, or recommendation outside the allowed range → present the rule-curve release |
| Ensemble disagreement | Flag "uncertain" when the std of the 5 seeds' recommended releases exceeds 10% |
| Flood-alert mode | When forecast inflow exceeds a set fraction of the design flood, disable policy recommendations and apply flood-control rules only |
| Log retention | Store inputs, mask, recommendation, actual release and reasons (Appendix E format plus approver) |

---

## 9. Step 8 — Monitoring and retraining in operation

| Monitored item | Trigger | Action |
|---|---|---|
| Recommendation vs actual release | 30-day mean gap > 15% | Interview operators → revisit reward coefficients and mask |
| Environment prediction error | Expected − observed level > 0.3 m persistently | Recalibrate curves (sedimentation etc.) |
| Inflow distribution shift | Recent 5-year monthly means outside ±2σ of the training distribution | Retrain with recent data |
| Regulation change | Limit level or minimum flow changed | Update the mask immediately (no training needed) → re-evaluate → retrain if needed |
| Scheduled retraining | Once a year | Add the latest 5 years, 5 seeds, re-pass the §7 protocol |

Keep the **config files exactly as trained** (the approach of this repository) so that data, seeds and rules used for every model remain traceable.

---

## 10. Where the analysis code lives

| Analysis | Code |
|---|---|
| Load and compare config files | notebook 01 `load_config()`, switch table |
| Water balance and constraint cost | `toy_reservoir.py`, notebook 02 |
| Mask and mapping visualisation | notebook 03 |
| PPO / PPO-Lagrangian training loop (reference) | notebook 04 `train()` |
| Reading appendix logs and metrics | notebook 05 `COLS_E`, `LABELS` |
| Data quality check | this document §3.4 |
| Environment calibration and unit tests | this document §4.2–4.3 |
| Evaluation metrics | this document §7.1 |
| Daily inference | this document §8.2 |

---

## 11. Limitations and cautions

- **The toy environment is not a real dam.** Tutorial numbers illustrate mechanisms and do not guarantee performance on a real reservoir.
- **The paper's code is private:** the environment and policy must be re-implemented. Use the config files only as the record of hyper-parameters.
- **Out-of-distribution events:** policy behaviour under extremes absent from training is not guaranteed. The mask and flood-alert mode mitigate this, but regulatory responsibility stays with the operator.
- **Single-objective bias:** with all reward coefficients at 1.0, energy may dominate. Run a coefficient sensitivity analysis if water supply or environmental objectives matter.
- **Legal status:** in most jurisdictions an AI recommendation is advisory; the legal decision-maker for releases does not change.

---

## 12. Deployment checklist

- [ ] §2 problem-definition table settled, regulation clause mapping written
- [ ] Inflow ≥20 years, <5% missing, latest characteristic curves, rules parameterised (`rules.yaml`)
- [ ] Environment calibrated: level RMSE <0.5 m, energy bias <5%
- [ ] Unit tests pass: mass balance, non-empty mask, observation range
- [ ] Trained in the order A3 → B3 → C2, 5 seeds, healthy TensorBoard curves
- [ ] Test-year, stress, forecast-error and initial-condition protocols passed; improvement over the rule curve confirmed
- [ ] Decision-support pipeline, fallback, logging and approval process in place
- [ ] Monitoring triggers and retraining schedule documented
