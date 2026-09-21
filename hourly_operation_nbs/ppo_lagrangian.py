"""
Compact PPO-Lagrangian for the hourly reservoir environment.

* batched rollout over N parallel environments (one forward pass per hour for all envs)
* masked categorical policy (hard constraints come from env.action_mask())
* TWO Lagrange multipliers, one per constraint group (flood, drought), each with its own threshold,
  so the priority "flood > drought > everything else" is expressed as thresholds + initial multipliers
* the policy is conditioned on the operator's target position (sampled per episode during training)
* hydropower is not a constraint; it only appears in the reward with a small weight (env.w['energy'])
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hourly_reservoir import HourlyReservoirEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256, 128)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.encoder = nn.Sequential(*layers)
        self.actor = nn.Linear(d, n_actions)
        self.v_r = nn.Linear(d, 1)                  # reward value
        self.v_c = nn.Linear(d, 2)                  # cost values: [flood, drought]

    def forward(self, obs, mask=None):
        z = self.encoder(obs)
        logits = self.actor(z)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits), self.v_r(z).squeeze(-1), self.v_c(z)


def gae(rew, val, done, gamma, lam):
    T = len(rew)
    adv = np.zeros_like(rew)
    g = np.zeros(rew.shape[1:], dtype=rew.dtype)
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv * (1 - done[t]) - val[t]
        g = delta + gamma * lam * (1 - done[t]) * g
        adv[t] = g
    return adv, adv + val


class RunningNorm:
    """Running mean / std used to normalise value-function targets (a light PopArt)."""
    def __init__(self, shape=()):
        self.mean, self.var, self.n = np.zeros(shape, np.float64), np.ones(shape, np.float64), 1e-4
    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, *self.mean.shape)
        m, v, n = x.mean(0), x.var(0), x.shape[0]
        d = m - self.mean; tot = self.n + n
        self.mean = self.mean + d * n / tot
        self.var = (self.var * self.n + v * n + d ** 2 * self.n * n / tot) / tot
        self.n = tot
    @property
    def std(self):
        return np.sqrt(self.var) + 1e-6


def pretrain_bc(model, envs, rng, n_episodes=16, horizon=24 * 30, epochs=30, lr=1e-3, target_choices=(0.2, 0.5, 0.8), wetness_range=(0.7, 1.5), init_spread=(0.0, 0.0), verbose=True):
    """Behaviour cloning from TargetTrackingRule so that PPO starts from a sensible operator-like policy
    instead of a random one (a random policy empties the reservoir within a day)."""
    from hourly_reservoir import TargetTrackingRule
    obs_l, mask_l, act_l = [], [], []
    for k in range(n_episodes):
        e = envs[k % len(envs)]
        rule = TargetTrackingRule(e)
        o, info = e.reset(seed=int(rng.integers(1 << 31)), horizon_hours=horizon, target_position=float(rng.choice(target_choices)), wetness=float(rng.uniform(*wetness_range)), init_spread=init_spread)
        done = False
        while not done:
            a = rule(o, info["action_mask"]); obs_l.append(o); mask_l.append(info["action_mask"]); act_l.append(a)
            o, r, te, tr, info = e.step(a); done = te or tr
    obs = torch.as_tensor(np.array(obs_l), device=DEVICE); mask = torch.as_tensor(np.array(mask_l), device=DEVICE); act = torch.as_tensor(np.array(act_l), device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        tot = 0.0
        for idx in torch.randperm(len(act), device=DEVICE).split(1024):
            dist, _, _ = model(obs[idx], mask[idx])
            loss = -dist.log_prob(act[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  BC epoch {ep:3d} | nll {tot/len(act):.3f} | samples {len(act)}")
    return model


@torch.no_grad()
def collect(model, envs, horizon, rng, target_choices, wetness_range, gamma, lam, norm_r, norm_c, init_spread=(0.0, 0.0)):
    n = len(envs)
    obs_l, mask_l, act_l, logp_l, rew_l, cost_l, vr_l, vc_l, done_l = ([] for _ in range(9))
    obs = np.zeros((n, envs[0].obs_dim), np.float32)
    mask = np.zeros((n, envs[0].n_actions), bool)
    for i, e in enumerate(envs):
        o, info = e.reset(seed=int(rng.integers(1 << 31)), horizon_hours=horizon, init_spread=init_spread,
                          target_position=float(rng.choice(target_choices)), wetness=float(rng.uniform(*wetness_range)))
        obs[i], mask[i] = o, info["action_mask"]
    ep_stats = []
    for t in range(horizon):
        ot = torch.as_tensor(obs, device=DEVICE)
        mt = torch.as_tensor(mask, device=DEVICE)
        dist, vr, vc = model(ot, mt)
        a = dist.sample()
        logp = dist.log_prob(a)
        obs_l.append(obs.copy()); mask_l.append(mask.copy()); act_l.append(a.cpu().numpy()); logp_l.append(logp.cpu().numpy())
        vr_l.append(vr.cpu().numpy() * norm_r.std + norm_r.mean); vc_l.append(vc.cpu().numpy() * norm_c.std + norm_c.mean)
        rew = np.zeros(n, np.float32); cost = np.zeros((n, 2), np.float32); done = np.zeros(n, np.float32)
        for i, e in enumerate(envs):
            o, r, te, tr, info = e.step(int(a[i]))
            rew[i] = r; cost[i] = (info["flood_cost"], info["drought_cost"]); done[i] = float(te or tr)
            obs[i], mask[i] = o, info["action_mask"]
            if te or tr:
                h = pd.DataFrame(e.history)
                ep_stats.append(dict(ret=h.reward.sum(), flood=h.flood_cost.mean(), drought=h.drought_cost.mean(),
                                     band=100 * ((h.level_next <= h.upper) & (h.level_next >= h.lower)).mean()))
                o, info = e.reset(seed=int(rng.integers(1 << 31)), horizon_hours=horizon, init_spread=init_spread,
                                  target_position=float(rng.choice(target_choices)), wetness=float(rng.uniform(*wetness_range)))
                obs[i], mask[i] = o, info["action_mask"]
        rew_l.append(rew); cost_l.append(cost); done_l.append(done)
    b = dict(obs=np.array(obs_l), mask=np.array(mask_l), act=np.array(act_l), logp=np.array(logp_l),
             rew=np.array(rew_l), cost=np.array(cost_l), vr=np.array(vr_l), vc=np.array(vc_l), done=np.array(done_l))
    b["adv_r"], b["ret_r"] = gae(b["rew"], b["vr"], b["done"], gamma, lam)
    adv_c, ret_c = zip(*[gae(b["cost"][..., k], b["vc"][..., k], b["done"], gamma, lam) for k in range(2)])
    b["adv_c"], b["ret_c"] = np.stack(adv_c, -1), np.stack(ret_c, -1)
    return {k: v.reshape(-1, *v.shape[2:]) for k, v in b.items()}, pd.DataFrame(ep_stats)


def train(env_kwargs=None, n_envs=8, horizon=24 * 30, n_iters=60, epoch_per_collect=8, batch_size=1024, lr=3e-4,
          clip_ratio=0.2, entropy_weight=0.005, value_weight=0.5, gamma=0.995, lam=0.95,
          lagrangian=True, thresholds=(0.005, 0.02), lag_lr=(2.0, 1.0), initial_penalty=(2.0, 1.0), dual_clip=10.0, bc_episodes=16, bc_epochs=30, init_spread=(-1.0, 2.5),
          target_choices=(0.2, 0.35, 0.5, 0.65, 0.8), wetness_range=(0.7, 1.5), seed=0, log_every=5, save_path=None, verbose=True):
    """thresholds / lag_lr / initial_penalty are (flood, drought). Flood is stricter (smaller threshold, larger
    initial multiplier) which is how the operating priority 'flood > drought' is encoded."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env_kwargs = env_kwargs or {}
    envs = [HourlyReservoirEnv(**env_kwargs) for _ in range(n_envs)]
    model = ActorCritic(envs[0].obs_dim, envs[0].n_actions).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lam_v = np.array(initial_penalty, np.float32) if lagrangian else np.zeros(2, np.float32)
    norm_r, norm_c = RunningNorm(()), RunningNorm((2,))
    if bc_episodes > 0:
        pretrain_bc(model, envs, rng, n_episodes=bc_episodes, horizon=horizon, epochs=bc_epochs, target_choices=target_choices, wetness_range=wetness_range, init_spread=init_spread, verbose=verbose)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, t0 = [], time.time()
    for it in range(n_iters):
        b, ep = collect(model, envs, horizon, rng, target_choices, wetness_range, gamma, lam, norm_r, norm_c, init_spread)
        T = len(b["rew"])
        norm_r.update(b["ret_r"]); norm_c.update(b["ret_c"])
        b["ret_r"] = (b["ret_r"] - norm_r.mean) / norm_r.std; b["ret_c"] = (b["ret_c"] - norm_c.mean) / norm_c.std
        tens = {k: torch.as_tensor(v, device=DEVICE) for k, v in b.items() if k in ("obs", "mask", "act", "logp", "adv_r", "ret_r", "adv_c", "ret_c")}
        tens["adv_r"] = (tens["adv_r"] - tens["adv_r"].mean()) / (tens["adv_r"].std() + 1e-8)
        if lagrangian:
            tens["adv_c"] = (tens["adv_c"] - tens["adv_c"].mean(0)) / (tens["adv_c"].std(0) + 1e-8)
        lam_t = torch.as_tensor(lam_v, device=DEVICE)
        for _ in range(epoch_per_collect):
            for idx in torch.randperm(T, device=DEVICE).split(batch_size):
                dist, vr, vc = model(tens["obs"][idx], tens["mask"][idx])
                logp = dist.log_prob(tens["act"][idx]); ratio = torch.exp(logp - tens["logp"][idx])
                adv = tens["adv_r"][idx]
                if lagrangian:
                    adv = (adv - (tens["adv_c"][idx] * lam_t).sum(-1)) / (1 + lam_t.sum())
                pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv).mean()
                vloss = ((vr - tens["ret_r"][idx]) ** 2).mean() + (((vc - tens["ret_c"][idx]) ** 2).mean() if lagrangian else 0.0)
                loss = pg + value_weight * vloss - entropy_weight * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        # ---- dual update once per collect, using the mean episode cost of this batch
        mean_cost = np.array([b["cost"][:, 0].mean(), b["cost"][:, 1].mean()])
        if lagrangian:
            lam_v = np.clip(lam_v + np.array(lag_lr) * (mean_cost - np.array(thresholds)), 0.0, dual_clip).astype(np.float32)
        row = dict(iter=it, ret=ep.ret.mean() if len(ep) else np.nan, flood_cost=mean_cost[0], drought_cost=mean_cost[1],
                   band_pct=ep.band.mean() if len(ep) else np.nan, lam_flood=lam_v[0], lam_drought=lam_v[1], sec=time.time() - t0)
        hist.append(row)
        if verbose and (it % log_every == 0 or it == n_iters - 1):
            print(f"iter {it:3d} | ret {row['ret']:8.2f} | flood {row['flood_cost']:.4f} | drought {row['drought_cost']:.4f} | "
                  f"band {row['band_pct']:5.1f}% | lam {lam_v[0]:.2f}/{lam_v[1]:.2f} | {row['sec']:.0f}s")
    if save_path:
        torch.save(dict(state_dict=model.state_dict(), obs_dim=envs[0].obs_dim, n_actions=envs[0].n_actions, env_kwargs=env_kwargs), save_path)
    return model, pd.DataFrame(hist)


def load_policy(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ActorCritic(ck["obs_dim"], ck["n_actions"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck["env_kwargs"]


class PolicyAgent:
    """Callable (obs, mask) -> action for run_episode / decision support. greedy=True picks the mode."""

    def __init__(self, model, greedy=True):
        self.model, self.greedy = model, greedy

    @torch.no_grad()
    def __call__(self, obs, mask):
        dist, _, _ = self.model(torch.as_tensor(obs, device=DEVICE)[None], torch.as_tensor(mask, device=DEVICE)[None])
        return int(dist.probs.argmax()) if self.greedy else int(dist.sample())

    @torch.no_grad()
    def probs(self, obs, mask):
        dist, _, _ = self.model(torch.as_tensor(obs, device=DEVICE)[None], torch.as_tensor(mask, device=DEVICE)[None])
        return dist.probs[0].cpu().numpy()
