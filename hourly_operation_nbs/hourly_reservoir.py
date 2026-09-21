"""
Hourly multipurpose-dam operation environment for decision support.

Design goals (from the operating agency's requirements):
  * hourly time step; flood forecasting on a 72-hour hourly horizon, daily forecasts beyond that
  * constraint priority: dam safety / flood > drought > everything else. Hydropower is NOT a constraint
    (it only enters as a small optional reward term)
  * decision inputs: dam level, flow at a downstream control point, weather-based inflow forecast
  * monthly operating band: upper / lower guide levels defined on the 1st of every month and
    linearly interpolated in between; operation must stay inside the band
  * operator preference: a target position inside the band (0 = lower guide, 0.5 = middle,
    1 = upper guide) that conditions the policy

All numbers are synthetic and only loosely representative of a mid-size Korean multipurpose dam.
Replace the tables / generators with real data as described in the notebooks.
"""
from __future__ import annotations

import numpy as np

H = 3600.0                       # seconds per hour
HOURS_PER_DAY = 24
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_START_DOY = np.concatenate([[0], np.cumsum(DAYS)[:-1]])  # 0-based day-of-year of the 1st of each month

# --------------------------------------------------------------------------- dam characteristics
Z_DEAD = 110.0            # dead (minimum) level, m
Z_DESIGN_FLOOD = 123.0    # design flood level, m  (absolute hard limit)
Z_CREST = 125.0
LEVEL_TABLE = np.array([105.0, 110.0, 113.0, 116.0, 119.0, 121.0, 123.0, 125.0])
STORAGE_TABLE = np.array([40.0, 150.0, 260.0, 410.0, 600.0, 760.0, 950.0, 1150.0])  # 1e6 m^3

Q_MIN, Q_MAX = 5.0, 3000.0        # total release range, m^3/s
Q_TURBINE_MAX = 300.0             # m^3/s
Q_ENV_MIN = 15.0                  # minimum environmental flow, m^3/s
Q_DS_SAFE = 2500.0                # downstream control-point channel capacity, m^3/s
Q_DS_WARN = 2000.0                # downstream warning flow, m^3/s
RAMP_MAX = 300.0                  # max change of release per hour, m^3/s
LATERAL_RATIO = 0.35              # local inflow between dam and control point, as a ratio of dam inflow
ROUTING_LAG_H = 3                 # travel time dam -> control point, hours
ROUTING_K = 0.65                  # simple linear-reservoir attenuation factor
TAILWATER = 62.0                  # m
EFFICIENCY = 0.85

# monthly guide levels on the 1st of each month (m).  Flood season (Jun-Sep) has a lowered upper guide.
UPPER_GUIDE = np.array([118.0, 118.0, 118.0, 117.5, 117.0, 115.0, 114.0, 114.0, 116.0, 118.0, 118.5, 118.0])
LOWER_GUIDE = np.array([112.0, 111.5, 111.0, 111.0, 111.0, 111.0, 111.0, 111.0, 111.5, 112.0, 112.5, 112.5])

N_ACTIONS = 60
DEFAULT_MAPPING = "square"        # fine resolution at low releases (dry-season flows of 20-60 m^3/s)
FC_HOURS = 72                     # hourly forecast horizon
FC_DAYS = 7                       # daily forecast horizon beyond the hourly one (days 4..10)

# --------------------------------------------------------------------------- static helpers
def storage_from_level(z):
    return np.interp(z, LEVEL_TABLE, STORAGE_TABLE)


def level_from_storage(v):
    return np.interp(v, STORAGE_TABLE, LEVEL_TABLE)


def guide_levels(doy: float) -> tuple[float, float]:
    """Upper / lower guide level for a (fractional, 0-based) day of year, linearly interpolated
    between the values defined on the 1st of each month."""
    x = np.concatenate([MONTH_START_DOY, [365.0]])
    up = np.concatenate([UPPER_GUIDE, [UPPER_GUIDE[0]]])
    lo = np.concatenate([LOWER_GUIDE, [LOWER_GUIDE[0]]])
    d = doy % 365
    return float(np.interp(d, x, up)), float(np.interp(d, x, lo))


def map_action(a, n_actions=N_ACTIONS, mapping=None):
    mapping = mapping or DEFAULT_MAPPING
    u = np.asarray(a, dtype=float) / (n_actions - 1)
    f = {"linear": u, "square": u ** 2, "sqrt": np.sqrt(u), "cubic": u ** 3}[mapping]
    return Q_MIN + f * (Q_MAX - Q_MIN)


def release_grid(mapping=None, n_actions=N_ACTIONS):
    return map_action(np.arange(n_actions), n_actions, mapping)


def hydropower_mw(release, level):
    q = min(release, Q_TURBINE_MAX)
    head = max(level - TAILWATER, 0.0)
    return 9.81 * EFFICIENCY * q * head / 1000.0


# --------------------------------------------------------------------------- synthetic hydrology
def synthetic_hourly_inflow(n_hours: int, seed=None, start_doy: int = 0, wetness: float = 1.0):
    """Hourly inflow = seasonal baseflow + storm hydrographs generated from synthetic rainfall events.
    Storms are concentrated in the monsoon season (June-September). `wetness` scales the whole series."""
    rng = np.random.default_rng(seed)
    t_h = np.arange(n_hours)
    doy = (t_h / 24.0 + start_doy) % 365
    base = 45 + 90 * np.exp(-((doy - 215) / 55.0) ** 2) + 12 * np.sin(2 * np.pi * (doy - 100) / 365)
    base = base * np.exp(rng.normal(0, 0.05, n_hours).cumsum() * 0.02)      # slow random walk
    storm = np.zeros(n_hours)
    # storm frequency depends on season
    n_days = n_hours // 24 + 1
    for d in range(n_days):
        dd = (d + start_doy) % 365
        p = 0.02 + 0.12 * np.exp(-((dd - 205) / 40.0) ** 2) + 0.06 * np.exp(-((dd - 255) / 20.0) ** 2)   # monsoon + autumn typhoons
        if rng.random() < p:
            t0 = d * 24 + rng.integers(0, 24)
            peak = rng.lognormal(np.log(600), 0.8) * (1 + 3 * np.exp(-((dd - 205) / 40.0) ** 2) + 2 * np.exp(-((dd - 255) / 20.0) ** 2))
            rise, fall = rng.integers(4, 10), rng.integers(12, 36)
            shape = np.concatenate([np.linspace(0, 1, rise, endpoint=False), np.exp(-np.arange(fall) / (fall / 3))])
            seg = storm[t0:t0 + len(shape)]
            seg += peak * shape[: len(seg)]
    return np.clip((base + storm) * wetness, 3.0, None)


def make_forecasts(inflow: np.ndarray, seed=None, hourly_sigma=(0.08, 0.35), daily_sigma=0.45):
    """Return (hourly_fc, daily_fc) forecast arrays issued at every hour t.
    hourly_fc[t, k] = forecast for hour t+1+k (k < 72), multiplicative log-normal error growing with lead,
    correlated across leads within one issue time.
    daily_fc[t, j]  = forecast daily-mean inflow for day (t//24) + 4 + j, j < 7."""
    rng = np.random.default_rng(seed)
    T = len(inflow)
    pad = np.concatenate([inflow, np.full(FC_HOURS + 24 * (FC_DAYS + 4), inflow[-24:].mean())])
    leads = np.arange(1, FC_HOURS + 1)
    sig = hourly_sigma[0] + (hourly_sigma[1] - hourly_sigma[0]) * (leads / FC_HOURS)
    eps0 = rng.normal(0, 1, (T, 1))
    eps = eps0 * np.sqrt(0.6) + rng.normal(0, 1, (T, FC_HOURS)) * np.sqrt(0.4)   # shared + lead-specific
    idx = np.arange(T)[:, None] + leads[None, :]
    hourly_fc = pad[idx] * np.exp(sig[None, :] * eps - 0.5 * sig[None, :] ** 2)
    daily_true = pad[: (len(pad) // 24) * 24].reshape(-1, 24).mean(1)
    n_days = T // 24 + 1
    didx = np.arange(n_days)[:, None] + 4 + np.arange(FC_DAYS)[None, :]
    didx = np.clip(didx, 0, len(daily_true) - 1)
    daily_fc = daily_true[didx] * np.exp(rng.normal(0, daily_sigma, didx.shape) - 0.5 * daily_sigma ** 2)
    return hourly_fc, daily_fc


# --------------------------------------------------------------------------- environment
class HourlyReservoirEnv:
    """Gym-like hourly environment.

    reset(seed, start_doy, horizon_hours, init_level, target_position, inflow=None) -> obs, info
    step(action) -> obs, reward, terminated, truncated, info
    info carries: 'action_mask', 'flood_cost', 'drought_cost', 'cost_parts', 'downstream_flow', 'release'.
    """

    OBS_NAMES = ["level_in_band", "level_abs", "band_upper", "band_lower", "target_position", "level_minus_target",
                 "inflow_now", "fc_max_6h", "fc_mean_24h", "fc_max_72h", "fc_vol_72h_vs_freeboard", "daily_fc_mean_7d",
                 "ds_flow_now", "lateral_now", "prev_release", "doy_sin", "doy_cos", "hour_sin", "hour_cos"]

    def __init__(self, mapping=None, mask_tier="band", band_hard=True, w_target=1.0, w_ramp=0.2, w_energy=0.05,
                 w_flood=10.0, w_drought=5.0, penalise_in_reward=True):
        self.grid = release_grid(mapping)
        self.n_actions = len(self.grid)
        self.mask_tier = mask_tier              # 'safety' | 'band'
        self.band_hard = band_hard
        self.w = dict(target=w_target, ramp=w_ramp, energy=w_energy, flood=w_flood, drought=w_drought)
        self.penalise_in_reward = penalise_in_reward
        self.obs_dim = len(self.OBS_NAMES)

    # ------------------------------------------------------------------ state helpers
    def _doy(self, t=None):
        t = self.t if t is None else t
        return self.start_doy + t / 24.0

    def _band(self, t=None):
        return guide_levels(self._doy(t))

    def target_level(self, t=None):
        up, lo = self._band(t)
        return lo + self.target_position * (up - lo)

    def _lateral(self, t):
        return LATERAL_RATIO * self.inflow[max(t - 2, 0)]

    def _route(self, release):
        """Simple lag + linear-reservoir routing of the dam release to the control point."""
        self.release_buf.append(release)
        lagged = self.release_buf[-ROUTING_LAG_H - 1] if len(self.release_buf) > ROUTING_LAG_H else self.release_buf[0]
        self.routed = ROUTING_K * self.routed + (1 - ROUTING_K) * lagged
        return self.routed

    def predicted_level(self, release, hours, t=None):
        """Level after `hours` hours if `release` is held constant and the hourly forecast is exact."""
        t = self.t if t is None else t
        fc = self.hourly_fc[t, :hours]
        v = storage_from_level(self.level) + (fc.sum() - release * hours) * H / 1e6
        return float(level_from_storage(v))

    # ------------------------------------------------------------------ mask (priority: dam safety > downstream flood > env flow > ramp > band)
    def mask_intervals(self):
        """Feasible release intervals of every constraint layer, in priority order."""
        up, lo = self._band()
        v_now = storage_from_level(self.level)
        fc6 = self.hourly_fc[self.t, :6]
        fc24 = self.hourly_fc[self.t, :24]
        # 1. dam safety: level after 6 h must stay below the design flood level  -> release >= q_safe_min
        q_safe_min = (fc6.sum() - (storage_from_level(Z_DESIGN_FLOOD) - v_now) * 1e6 / H) / 6.0
        # 2. dead storage: level after 6 h must stay above the dead level          -> release <= q_dead_max
        q_dead_max = (fc6.sum() + (v_now - storage_from_level(Z_DEAD)) * 1e6 / H) / 6.0
        # 3. downstream flood: routed release + lateral inflow must not exceed the channel capacity
        q_ds_max = Q_DS_SAFE - LATERAL_RATIO * fc6.max()
        # 4. band: keep tomorrow's level inside the guide band (optional hard layer, soft cost otherwise)
        q_band_min = (fc24.sum() - (storage_from_level(up) - v_now) * 1e6 / H) / 24.0
        q_band_max = (fc24.sum() + (v_now - storage_from_level(lo)) * 1e6 / H) / 24.0
        layers = [("dam_safety", q_safe_min, max(q_safe_min, q_dead_max)),
                  ("downstream_capacity", -np.inf, q_ds_max),
                  ("env_min_flow", Q_ENV_MIN, np.inf),
                  ("ramp", self.prev_release - RAMP_MAX if self.prev_release is not None else -np.inf,
                           self.prev_release + RAMP_MAX if self.prev_release is not None else np.inf)]
        if self.mask_tier == "band" and self.band_hard:
            layers.append(("band", q_band_min, q_band_max))
        return layers

    def action_mask(self) -> np.ndarray:
        """Intersect the layer intervals in priority order. When a lower-priority layer does not intersect
        the current interval it is *relaxed* to the nearest boundary instead of being dropped, so higher
        priority constraints are always honoured and the lower one is violated as little as possible."""
        lo, hi = Q_MIN, Q_MAX
        applied = {}
        for name, a, b in self.mask_intervals():
            if max(lo, a) <= min(hi, b):
                lo, hi = max(lo, a), min(hi, b); applied[name] = "ok"
            elif a > hi:
                lo = hi; applied[name] = "relaxed(upper)"
            else:
                hi = lo; applied[name] = "relaxed(lower)"
        g = self.grid
        mask = (g >= lo - 1e-9) & (g <= hi + 1e-9)
        if not mask.any():                                     # no grid point inside: take the nearest one(s)
            dist = np.maximum(np.maximum(lo - g, g - hi), 0.0)
            mask = dist <= dist.min() + 1e-9
        self.last_mask_info = dict(interval=(lo, hi), applied=applied)
        return mask

    # ------------------------------------------------------------------ observation
    def _obs(self):
        t = self.t
        up, lo = self._band()
        width = max(up - lo, 0.5)
        fc = self.hourly_fc[t]
        free = max(storage_from_level(Z_DESIGN_FLOOD) - storage_from_level(self.level), 1.0)
        dfc = self.daily_fc[min(t // 24, len(self.daily_fc) - 1)]
        hour = t % 24
        o = np.array([
            (self.level - lo) / width,
            (self.level - Z_DEAD) / (Z_DESIGN_FLOOD - Z_DEAD),
            (up - Z_DEAD) / (Z_DESIGN_FLOOD - Z_DEAD),
            (lo - Z_DEAD) / (Z_DESIGN_FLOOD - Z_DEAD),
            self.target_position,
            (self.level - self.target_level()) / width,
            self.inflow[t] / Q_MAX,
            fc[:6].max() / Q_MAX,
            fc[:24].mean() / Q_MAX,
            fc.max() / Q_MAX,
            min(fc.sum() * H / 1e6 / free, 3.0),
            dfc.mean() / Q_MAX,
            self.ds_flow / Q_DS_SAFE,
            self._lateral(t) / Q_MAX,
            (self.prev_release if self.prev_release is not None else self.inflow[t]) / Q_MAX,
            np.sin(2 * np.pi * self._doy() / 365), np.cos(2 * np.pi * self._doy() / 365),
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        ], dtype=np.float32)
        return o

    # ------------------------------------------------------------------ API
    def reset(self, seed=None, start_doy=None, horizon_hours=24 * 30, init_level=None, target_position=0.5,
              inflow=None, wetness=1.0, forecast_seed=None, init_spread=(0.0, 0.0)):
        """init_spread=(a, b): sample the initial level in [lower_guide + a, upper_guide + b] (default: inside the band).
        Training uses e.g. (-1.0, 2.5) so the policy also learns to recover from outside the band."""
        self.rng = np.random.default_rng(seed)
        self.start_doy = int(self.rng.integers(0, 365)) if start_doy is None else int(start_doy)
        self.horizon = int(horizon_hours)
        self.target_position = float(target_position)
        n = self.horizon + FC_HOURS + 24 * (FC_DAYS + 4)
        self.inflow = np.asarray(inflow, float) if inflow is not None else synthetic_hourly_inflow(
            n, seed=int(self.rng.integers(1 << 31)), start_doy=self.start_doy, wetness=wetness)
        self.hourly_fc, self.daily_fc = make_forecasts(self.inflow, seed=forecast_seed if forecast_seed is not None else int(self.rng.integers(1 << 31)))
        self.t = 0
        up, lo = self._band()
        self.level = float(init_level) if init_level is not None else float(np.clip(self.rng.uniform(lo + init_spread[0], up + init_spread[1]), Z_DEAD + 0.3, Z_DESIGN_FLOOD - 0.5))
        self.prev_release = None
        self.release_buf = [self.inflow[0]]
        self.routed = self.inflow[0]
        self.ds_flow = self.inflow[0] + self._lateral(0)
        self.history = []
        return self._obs(), {"action_mask": self.action_mask()}

    def step(self, action):
        t = self.t
        up, lo = self._band()
        width = max(up - lo, 0.5)
        inflow = float(self.inflow[t])
        release = float(self.grid[int(action)])
        relaxed = [k for k, v in getattr(self, 'last_mask_info', {'applied': {}})['applied'].items() if v != 'ok']
        v_next = storage_from_level(self.level) + (inflow - release) * H / 1e6
        # physical clipping: cannot release water that is not there
        if v_next < storage_from_level(Z_DEAD):
            release = max(inflow + (storage_from_level(self.level) - storage_from_level(Z_DEAD)) * 1e6 / H, 0.0)
            v_next = storage_from_level(Z_DEAD)
        z_next = float(level_from_storage(v_next))
        ds = self._route(release) + self._lateral(t + 1)
        self.ds_flow = ds

        # ---- costs (all normalised to [0, 1]); flood and drought are the constraint groups
        c = {}
        c["flood_level"] = min(max(z_next - up, 0.0) / max(Z_DESIGN_FLOOD - up, 0.5), 1.0)
        c["flood_design"] = min(max(z_next - Z_DESIGN_FLOOD, 0.0) / (Z_CREST - Z_DESIGN_FLOOD), 1.0)
        c["flood_downstream"] = min(max(ds - Q_DS_SAFE, 0.0) / Q_DS_SAFE, 1.0)
        c["drought_level"] = min(max(lo - z_next, 0.0) / max(lo - Z_DEAD, 0.5), 1.0)
        c["drought_flow"] = min(max(Q_ENV_MIN - release, 0.0) / Q_ENV_MIN, 1.0)
        flood_cost = max(c["flood_level"], c["flood_downstream"], 3 * c["flood_design"])
        drought_cost = max(c["drought_level"], c["drought_flow"])
        target_dev = abs(z_next - self.target_level(t + 1)) / width
        ramp = abs(release - self.prev_release) / Q_MAX if self.prev_release is not None else 0.0
        power = hydropower_mw(release, self.level)
        energy_norm = power / hydropower_mw(Q_TURBINE_MAX, Z_DESIGN_FLOOD)

        w = self.w
        reward = -w["target"] * target_dev - w["ramp"] * ramp + w["energy"] * energy_norm
        if self.penalise_in_reward:
            reward -= w["flood"] * flood_cost + w["drought"] * drought_cost
        terminated = z_next > Z_CREST

        self.history.append(dict(t=t, doy=self._doy(), level=self.level, level_next=z_next, upper=up, lower=lo,
                                 target=self.target_level(), inflow=inflow, release=release, ds_flow=ds,
                                 lateral=self._lateral(t + 1), power_mw=power, reward=reward,
                                 flood_cost=flood_cost, drought_cost=drought_cost, target_dev=target_dev,
                                 mask_relaxed=";".join(relaxed), **c))
        self.level = min(z_next, Z_CREST)
        self.prev_release = release
        self.t += 1
        truncated = self.t >= self.horizon
        done = terminated or truncated
        info = dict(action_mask=self.action_mask() if not done else np.ones(self.n_actions, bool),
                    flood_cost=flood_cost, drought_cost=drought_cost, cost_parts=c, downstream_flow=ds, release=release)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ utilities for decision support
    def snapshot(self):
        return dict(t=self.t, level=self.level, prev_release=self.prev_release, release_buf=list(self.release_buf),
                    routed=self.routed, ds_flow=self.ds_flow, history=list(self.history))

    def restore(self, snap):
        self.t, self.level, self.prev_release = snap["t"], snap["level"], snap["prev_release"]
        self.release_buf, self.routed, self.ds_flow = list(snap["release_buf"]), snap["routed"], snap["ds_flow"]
        self.history = list(snap["history"])


# --------------------------------------------------------------------------- rule-based baseline
class TargetTrackingRule:
    """Operator-style rule: release so that the level reaches the target in `tau_h` hours using the 24 h
    forecast mean, then clip to the action mask (which encodes all the hard constraints)."""

    def __init__(self, env: HourlyReservoirEnv, tau_h: int = 72):
        self.env, self.tau = env, tau_h

    def __call__(self, obs, mask):
        env = self.env
        v_now, v_tgt = storage_from_level(env.level), storage_from_level(env.target_level())
        q = env.hourly_fc[env.t, :24].mean() + (v_now - v_tgt) * 1e6 / H / self.tau
        allowed = np.flatnonzero(mask)
        return int(allowed[np.argmin(np.abs(env.grid[allowed] - q))])


def run_episode(env: HourlyReservoirEnv, policy, **reset_kw):
    import pandas as pd
    obs, info = env.reset(**reset_kw)
    done = False
    while not done:
        a = policy(obs, info["action_mask"])
        obs, r, te, tr, info = env.step(a)
        done = te or tr
    return pd.DataFrame(env.history)


def summarize(log, name=""):
    """Operating KPIs with the agency's priority order: flood, drought, band compliance, then energy."""
    import pandas as pd
    hours = len(log)
    return pd.Series({
        "hours": hours,
        "flood_hours_above_upper": int((log.level_next > log.upper).sum()),
        "hours_above_design_flood": int((log.level_next > Z_DESIGN_FLOOD).sum()),
        "downstream_over_capacity_hours": int((log.ds_flow > Q_DS_SAFE).sum()),
        "downstream_max_flow": log.ds_flow.max(),
        "drought_hours_below_lower": int((log.level_next < log.lower).sum()),
        "env_flow_violation_hours": int((log.release < Q_ENV_MIN).sum()),
        "band_compliance_pct": 100 * ((log.level_next <= log.upper) & (log.level_next >= log.lower)).mean(),
        "mean_abs_target_dev_m": (log.level_next - log.target).abs().mean(),
        "max_hourly_ramp": log.release.diff().abs().max(),
        "ramp_over_limit_hours": int((log.release.diff().abs() > RAMP_MAX + 1e-6).sum()),
        "mask_relaxed_hours": int((log.mask_relaxed != "").sum()),
        "energy_GWh": log.power_mw.sum() / 1000.0,
        "mean_flood_cost": log.flood_cost.mean(),
        "mean_drought_cost": log.drought_cost.mean(),
        "mean_reward": log.reward.mean(),
    }, name=name)
