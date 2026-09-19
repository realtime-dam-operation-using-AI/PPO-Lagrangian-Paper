"""
Toy single-reservoir environment for the tutorial notebooks.

This is NOT the environment used in the paper (that code is private). It is a small,
transparent re-implementation of the same *ideas* so the config switches can be
demonstrated end to end:

  * daily water-balance simulation over a 1-year horizon (365 steps)
  * 100 discrete actions mapped to a release in [Q_MIN, Q_MAX] (linear or non-linear mapping)
  * multi-constraint reward: hydropower reward minus level / flood / ecological /
    power-guarantee / navigation / action-change penalties (same 7 coefficients as the configs)
  * CONSTRAINT_TYPES 1/2/3 semantics (terminate / correct / correct + penalise)
  * a 'basic' or 'enhanced' action mask exposed in info["action_mask"]
  * a normalised constraint cost in info["constraint_cost_normalized"] for PPO-Lagrangian
  * observation type 5: [level, inflow, future inflow, upper-limit level, sin(day), cos(day)]

The numbers are loosely inspired by the Three Gorges reservoir but are simplified.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- physical constants
Z_DEAD, Z_NORMAL = 145.0, 175.0          # dead level / normal pool level (m)
Z_FLOOD_LIMIT = 155.0                    # flood-season upper limit (m), softer than the real 145 m
FLOOD_SEASON = (152, 273)                # day-of-year window (June 1 .. Sept 30)
V_DEAD, V_NORMAL = 171.5, 393.0          # storage at Z_DEAD / Z_NORMAL (1e8 m^3)
STORAGE_SLOPE = (V_NORMAL - V_DEAD) / (Z_NORMAL - Z_DEAD)  # 1e8 m^3 per metre

Q_MIN, Q_MAX = 4000.0, 40000.0           # release range (m^3/s)
Q_TURBINE_MAX = 30000.0                  # max flow through turbines (m^3/s)
P_CAP_MW = 22500.0                       # installed capacity (MW)
P_GUARANTEE_MW = 4990.0                  # guaranteed output (MW)
Q_ECOLOGICAL = 6000.0                    # minimum ecological flow (m^3/s)
Q_NAVIGATION = 5600.0                    # minimum navigation flow (m^3/s)
EFFICIENCY_K = 8.5                       # P[kW] = K * Q[m^3/s] * H[m]
MAX_DAILY_CHANGE = 0.30 * (Q_MAX - Q_MIN)  # enhanced-mask limit on |Q_t - Q_{t-1}|

N_ACTIONS = 100
HORIZON = 365


def level_from_storage(v: np.ndarray | float) -> np.ndarray | float:
    return Z_DEAD + (v - V_DEAD) / STORAGE_SLOPE


def storage_from_level(z: np.ndarray | float) -> np.ndarray | float:
    return V_DEAD + (z - Z_DEAD) * STORAGE_SLOPE


def tailwater_level(q: float) -> float:
    return 62.0 + 0.0004 * q  # m


def upper_limit_level(day_of_year: int) -> float:
    lo, hi = FLOOD_SEASON
    return Z_FLOOD_LIMIT if lo <= day_of_year <= hi else Z_NORMAL


# ----------------------------------------------------------------------------- inflow generator
def synthetic_inflow(n_days: int = HORIZON, seed: int | None = None, start_doy: int = 1) -> np.ndarray:
    """Seasonal sinusoid peaking in mid July + log-normal noise + occasional flood pulses."""
    rng = np.random.default_rng(seed)
    doy = (np.arange(n_days) + start_doy - 1) % 365 + 1
    seasonal = 14000 + 12000 * np.sin(2 * np.pi * (doy - 105) / 365)   # min ~2000 in Jan, max ~26000 in Jul
    noise = rng.lognormal(mean=0.0, sigma=0.15, size=n_days)
    pulses = np.zeros(n_days)
    for t in rng.choice(n_days, size=max(1, n_days // 120), replace=False):
        if 150 <= doy[t] <= 280:
            width = rng.integers(3, 8)
            pulses[t:t + width] += rng.uniform(15000, 35000)
    return np.clip(seasonal * noise + pulses, 1500, None)


# ----------------------------------------------------------------------------- action mapping
def map_action(a: np.ndarray | int, n_actions: int = N_ACTIONS, mapping: str = "linear") -> np.ndarray | float:
    """Map a discrete action index to a release fraction u in [0, 1], then to a flow in [Q_MIN, Q_MAX].

    The paper compares linear vs square / sqrt / cubic / exp / log mappings (config switch
    NONLINEAR_MAPPING_TYPE). Non-linear mappings change the *resolution* of the discrete grid:
    e.g. 'square' concentrates actions at low releases, 'sqrt' at high releases.
    """
    u = np.asarray(a, dtype=float) / (n_actions - 1)
    if mapping == "linear":
        f = u
    elif mapping == "square":
        f = u ** 2
    elif mapping == "sqrt":
        f = np.sqrt(u)
    elif mapping == "cubic":
        f = u ** 3
    elif mapping == "exp":
        f = (np.exp(3 * u) - 1) / (np.exp(3) - 1)
    elif mapping == "log":
        f = np.log1p(19 * u) / np.log(20)
    else:
        raise ValueError(f"unknown mapping {mapping!r}")
    return Q_MIN + f * (Q_MAX - Q_MIN)


def release_grid(mapping: str = "linear", n_actions: int = N_ACTIONS) -> np.ndarray:
    return map_action(np.arange(n_actions), n_actions, mapping)


# ----------------------------------------------------------------------------- environment
class ToyReservoirEnv:
    """Minimal gym-like API (reset / step) so the notebooks do not depend on a gym version."""

    def __init__(
        self,
        constraint_types: int = 3,
        enable_action_mask: bool = False,
        mask_rule_type: str = "basic",          # 'basic' | 'enhanced'
        mapping: str = "linear",                # 'linear' | 'square' | 'sqrt' | 'cubic' | 'exp' | 'log'
        reward_coeffs: dict | None = None,
        horizon: int = HORIZON,
        seed: int | None = None,
    ):
        self.constraint_types = constraint_types
        self.enable_action_mask = enable_action_mask
        self.mask_rule_type = mask_rule_type
        self.mapping = mapping
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.grid = release_grid(mapping)
        c = dict(power=1.0, level=1.0, flood=1.0, ecological=1.0, powerlimit=1.0, navigation=1.0, action=1.0)
        if reward_coeffs:
            c.update(reward_coeffs)
        self.coeffs = c
        self.obs_dim = 6
        self.n_actions = N_ACTIONS

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        doy = self._doy()
        future = self.inflow[self.t + 1: self.t + 8].mean() if self.t + 1 < self.horizon else self.inflow[-1]
        return np.array([
            (self.level - Z_DEAD) / (Z_NORMAL - Z_DEAD),
            self.inflow[self.t] / Q_MAX,
            future / Q_MAX,
            (upper_limit_level(doy) - Z_DEAD) / (Z_NORMAL - Z_DEAD),
            np.sin(2 * np.pi * doy / 365),
            np.cos(2 * np.pi * doy / 365),
        ], dtype=np.float32)

    def _doy(self) -> int:
        return (self.t + self.start_doy - 1) % 365 + 1

    def feasible_release_range(self) -> tuple[float, float]:
        """Release range that keeps tomorrow's level inside [Z_DEAD, upper_limit]. Used by the mask."""
        inflow = self.inflow[self.t]
        v_now = storage_from_level(self.level)
        z_hi = upper_limit_level(self._doy())
        sec = 86400 / 1e8
        q_lo = inflow - (storage_from_level(z_hi) - v_now) / sec      # release at least this to stay below z_hi
        q_hi = inflow + (v_now - storage_from_level(Z_DEAD)) / sec    # release at most this to stay above dead level
        return q_lo, q_hi

    def action_mask(self) -> np.ndarray:
        """Boolean mask over the 100 actions. 'basic' = level feasibility only;
        'enhanced' = also minimum flows and a maximum daily change (an operating-rule proxy)."""
        q_lo, q_hi = self.feasible_release_range()
        mask = (self.grid >= q_lo) & (self.grid <= q_hi)
        if self.mask_rule_type == "enhanced":
            mask &= self.grid >= max(Q_ECOLOGICAL, Q_NAVIGATION)
            if self.prev_release is not None:
                mask &= np.abs(self.grid - self.prev_release) <= MAX_DAILY_CHANGE
        if not mask.any():  # never return an empty mask: fall back to the closest feasible action
            target = np.clip(0.5 * (q_lo + q_hi), Q_MIN, Q_MAX)
            mask[np.argmin(np.abs(self.grid - target))] = True
        return mask

    # ------------------------------------------------------------------ gym-like API
    def reset(self, seed: int | None = None, start_doy: int = 1, init_level: float | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.start_doy = start_doy
        self.inflow = synthetic_inflow(self.horizon + 8, seed=int(self.rng.integers(1 << 31)), start_doy=start_doy)
        self.level = float(init_level) if init_level is not None else float(self.rng.uniform(150, 172))
        self.prev_release = None
        self.history: list[dict] = []
        return self._obs(), {"action_mask": self.action_mask()}

    def step(self, action: int):
        doy = self._doy()
        inflow = float(self.inflow[self.t])
        agent_release = float(self.grid[int(action)])
        release = agent_release
        v_now = storage_from_level(self.level)
        sec = 86400 / 1e8
        z_hi = upper_limit_level(doy)
        terminated = False
        cost_parts = {}

        # ---- level constraint handling (config CONSTRAINT_TYPES)
        v_next = v_now + (inflow - release) * sec
        z_next = level_from_storage(v_next)
        level_violation = max(0.0, Z_DEAD - z_next) + max(0.0, z_next - Z_NORMAL)
        flood_violation = max(0.0, z_next - z_hi) if z_hi < Z_NORMAL else 0.0
        if self.constraint_types == 1 and level_violation > 0:
            terminated = True
        if self.constraint_types in (2, 3):
            # correct the release so that the level stays within [Z_DEAD, Z_NORMAL]
            q_lo = inflow - (storage_from_level(Z_NORMAL) - v_now) / sec
            q_hi = inflow + (v_now - storage_from_level(Z_DEAD)) / sec
            release = float(np.clip(release, max(q_lo, 0.0), q_hi))
            v_next = v_now + (inflow - release) * sec
            z_next = level_from_storage(v_next)
        action_correction = abs(release - agent_release) / (Q_MAX - Q_MIN)

        # ---- hydropower
        head = self.level - tailwater_level(release)
        q_turbine = min(release, Q_TURBINE_MAX)
        power_mw = min(EFFICIENCY_K * q_turbine * head / 1000.0, P_CAP_MW)
        energy = power_mw * 24 / 1e5  # 1e8 kWh per day

        # ---- constraint costs, each normalised to [0, 1]
        cost_parts["level"] = min(level_violation / 5.0, 1.0)
        cost_parts["flood"] = min(flood_violation / 5.0, 1.0)
        cost_parts["ecological"] = max(0.0, Q_ECOLOGICAL - release) / Q_ECOLOGICAL
        cost_parts["powerlimit"] = max(0.0, P_GUARANTEE_MW - power_mw) / P_GUARANTEE_MW
        cost_parts["navigation"] = max(0.0, Q_NAVIGATION - release) / Q_NAVIGATION
        cost_parts["action"] = action_correction
        if self.prev_release is not None:
            cost_parts["action"] += 0.5 * abs(release - self.prev_release) / (Q_MAX - Q_MIN)

        c = self.coeffs
        power_reward = c["power"] * energy / (P_CAP_MW * 24 / 1e5)  # 1.0 == running at full capacity
        penalty = (c["level"] * cost_parts["level"] + c["flood"] * cost_parts["flood"]
                   + c["ecological"] * cost_parts["ecological"] + c["powerlimit"] * cost_parts["powerlimit"]
                   + c["navigation"] * cost_parts["navigation"] + c["action"] * cost_parts["action"])
        if self.constraint_types == 2:
            penalty = 0.0  # "correct but do not penalise"
        reward = power_reward - penalty
        constraint_cost = float(np.mean([cost_parts[k] for k in ("level", "flood", "ecological", "powerlimit", "navigation")]))

        # ---- advance
        self.history.append(dict(t=self.t, doy=doy, level=self.level, inflow=inflow, agent_release=agent_release,
                                 release=release, power_mw=power_mw, energy=energy, reward=reward,
                                 constraint_cost=constraint_cost, **{f"cost_{k}": v for k, v in cost_parts.items()}))
        self.level = float(np.clip(z_next, Z_DEAD - 5, Z_NORMAL + 5))
        self.prev_release = release
        self.t += 1
        truncated = self.t >= self.horizon
        info = {
            "constraint_cost_normalized": constraint_cost,
            "cost_parts": cost_parts,
            "power_mw": power_mw,
            "release": release,
            "action_mask": self.action_mask() if not (terminated or truncated) else np.ones(N_ACTIONS, bool),
        }
        return self._obs(), float(reward), terminated, truncated, info


def run_episode(env: ToyReservoirEnv, policy, seed: int = 0, **reset_kw):
    """policy(obs, mask) -> action. Returns a pandas DataFrame of the daily log."""
    import pandas as pd
    obs, info = env.reset(seed=seed, **reset_kw)
    done = False
    while not done:
        mask = info["action_mask"] if env.enable_action_mask else np.ones(N_ACTIONS, bool)
        a = policy(obs, mask)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    return pd.DataFrame(env.history)
