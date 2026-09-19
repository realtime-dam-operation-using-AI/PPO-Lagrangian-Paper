# 실제 댐 운영 적용 가이드 (Operation Guide)

> 🇬🇧 English version: [operation_guide.md](operation_guide.md)

이 문서는 논문 *"A Two-Layer Safe Reinforcement Learning Framework for Multi-Constraint Reservoir Operation"* 의 프레임워크(액션 마스킹 + PPO-Lagrangian)를 **실제 댐 운영에 적용하려는 기관·연구자**를 위한 절차서입니다. 저장소의 설정 파일(`DRL_training_config/`), 튜토리얼 노트북(`tutorial_nbs/`), toy 환경(`toy_reservoir.py`)을 어떤 순서로, 무엇을 준비해서 활용하는지 단계별로 설명합니다.

> **전제:** 논문의 실제 환경·정책 코드는 비공개입니다. 따라서 이 가이드는 (1) 공개된 설정 파일이 기록한 학습 레시피, (2) 튜토리얼의 toy 환경 구조를 **자기 댐에 맞게 다시 구현**하는 것을 전제로 합니다. 실제 코드베이스를 교신저자에게 제공받으면 4단계(학습)부터는 설정 파일을 거의 그대로 재사용할 수 있습니다.

---

## 0. 목적과 활용 범위

### 0.1 이 프레임워크가 하는 일

| 항목 | 내용 |
|---|---|
| 결정 대상 | **일 단위 방류량**(또는 발전 방류량) 하나. 이산 액션 100개 중 하나를 선택 |
| 목적함수 | 발전량 최대화 − (수위·홍수·생태·보증출력·항운·액션변화) 페널티 |
| 제약 처리 | 1층: **액션 마스크**가 규정 위반 액션을 사전에 제거(하드 제약) / 2층: **Lagrangian 쌍대 최적화**가 남은 위반의 기대값을 임계값 이하로 유지(소프트 제약) |
| 입력 | 현재 수위, 당일 유입량, 미래(예보) 유입량, 당일 상한수위, 날짜(sin/cos) |
| 출력 | 권고 방류량 + 그날 허용된 방류량 범위(마스크) |

### 0.2 권장 활용 형태

1. **의사결정 지원(Decision Support)** — 운영자가 매일 아침 권고 방류량과 허용 범위를 참고하여 최종 결정. **가장 현실적이고 권장되는 형태.**
2. **운영 규칙 개선 도구** — 학습된 정책이 만든 30년 운영 로그(부록 E 형식)를 분석해 기존 운영 곡선(rule curve)의 개선점 도출.
3. **시나리오 스트레스 테스트** — 극한 가뭄·홍수 시나리오에서 정책의 거동을 사전에 검증.
4. **자동 폐루프 제어** — 규정상·안전상 이유로 **권장하지 않음**. 최소한 운영자 승인 단계와 rule-curve 폴백이 있어야 합니다.

### 0.3 적용에 적합한 조건

- 단일 저수지(또는 상류 제어가 독립적인 저수지)이고 주 목적이 발전 + 다목적 제약 준수인 경우.
- 20년 이상의 일 단위 유입량 기록과 명문화된 운영 규정(홍수기 제한수위, 최소 방류량, 보증출력 등)이 있는 경우.
- 다중 저수지 연계 운영, 시간 단위 첨두 운영, 수요 대응형 급수는 환경과 액션 공간을 확장해야 하며 이 가이드의 범위를 넘습니다.

---

## 1. 전체 절차 요약

```
[1] 문제 정의 ─► [2] 데이터 수집·검증 ─► [3] 환경 구축·보정 ─► [4] 안전층(마스크) 설계
                                                                        │
[8] 운영·재학습 ◄─ [7] 의사결정 지원 배치 ◄─ [6] 평가·스트레스 테스트 ◄─ [5] 학습(PPO-Lagrangian)
```

| 단계 | 산출물 | 관련 저장소 자료 |
|---|---|---|
| 1 문제 정의 | 목적·제약 명세서, 액션/관측 정의 | `description_kor.md`, 노트북 01 |
| 2 데이터 | 검증된 유입량 파일, 특성곡선, 규정표 | 이 문서 §3 |
| 3 환경 | `<dam>_env.py`, 보정 리포트 | `toy_reservoir.py`, 노트북 02 |
| 4 마스크 | 마스크 규칙 함수, 공집합 테스트 | 노트북 03 |
| 5 학습 | 설정 파일(A/B/C 변형), 체크포인트, TensorBoard 로그 | `DRL_training_config/`, 노트북 04 |
| 6 평가 | 지표표, 스트레스 테스트 로그(부록 E 형식) | 노트북 05, `AppE_*.xlsx` |
| 7 배치 | 일일 추론 스크립트, 의사결정 로그 | 이 문서 §8 |
| 8 운영 | 드리프트 모니터링, 재학습 기준 | 이 문서 §9 |

---

## 2. 1단계 — 문제 정의

학습 전에 아래 표를 **댐 운영 부서와 함께** 확정합니다. 이 표가 곧 환경 코드와 설정 파일의 스위치가 됩니다.

| 결정 항목 | 논문 설정 | 자기 댐에서 정할 것 |
|---|---|---|
| 시간 간격 | 1일 | 1일 권장. 첨두 발전이면 1시간(액션·상태 재설계 필요) |
| 에피소드 길이 | 4년(1460일) | 수문 연도 단위로 2–5년 |
| 액션 | 방류량 100단계 이산 | 방류량 범위 `[Q_MIN, Q_MAX]`, 단계 수, 매핑 함수(§5) |
| 관측 | 6차원(타입 5) | 수위, 유입량, 예보 유입량, 상한수위, 날짜. 필요 시 하류 수위·급수 수요 추가 |
| 보상 항 | 7개 계수(모두 1.0) | 발전 외에 급수·환경유량 등 항 추가 가능. **먼저 모두 1.0으로 시작** |
| 하드 제약(마스크) | 수위 범위, 규정 유량 | 절대 위반 불가 항목만. 예: 사수위, 홍수기 제한수위, 최소 방류량, 일 변화율 |
| 소프트 제약(Lagrangian) | `constraint_cost_normalized ≤ 0.05` | 위반해도 치명적이지 않은 항목. 임계값은 §6.3 |
| 제약 처리 타입 | 3 (보정+페널티) | 실제 운영에서는 3 고정 |

**산출물:** `problem_spec.md` (표 + 근거 규정 조항 번호).

---

## 3. 2단계 — 필요한 데이터와 조건

### 3.1 필수 데이터

| 데이터 | 해상도·기간 | 형식 예시 | 용도 | 비고 |
|---|---|---|---|---|
| **유입량 실측** | 일, ≥20년 (30년 권장) | `date, inflow_m3s` | 학습·검증 시나리오 | 결측 5% 이하. 상류 댐 방류 영향이 있으면 순유입량 사용 |
| **유입량 예보** (또는 예보 오차 통계) | 일, 1–7일 선행 | `date, lead, inflow_fc` | 관측 `future_inflow` | 예보가 없으면 실측 이동평균 + 잡음으로 대체(논문 `obs_noise_*` 옵션 참고) |
| **수위–저수량 곡선** | 0.1–0.5 m 간격 | `level_m, storage_1e8m3` | 물수지 | 준설·퇴사 반영 최신본 |
| **방류량–미수위(tailwater) 곡선** | 유량 구간별 | `outflow_m3s, tailwater_m` | 낙차 계산 | 하류 댐 배수위 영향 시 2변수 |
| **발전설비 제원** | 상수 | 설비용량, 최대 발전유량, 최소 낙차, 효율(또는 효율곡선) | 발전량 | 튜토리얼 `EFFICIENCY_K=8.5` 는 단순 근사 |
| **운영 규정** | 날짜별 | 홍수기 기간·제한수위, 상시만수위, 사수위, 최소 방류량(생태·항운·급수), 보증출력, 일 방류 변화 한도 | 마스크·페널티 | 규정 문서 조항과 1:1 대응표 작성 |
| **과거 실제 운영 기록** | 일, ≥5년 | `date, level, inflow, outflow, energy` | 환경 보정, 베이스라인 | 부록 F 형식 참고 |

### 3.2 있으면 좋은 데이터

- 강수·기온(예보 모델 입력), 하류 수요·취수량, 전력 계통 가격/수요(발전 가치 가중), 증발·침투 손실.
- 극한 사상 기록(역대 최대 홍수·최장 가뭄): 스트레스 테스트 시나리오 구성.

### 3.3 데이터 파일 형식

논문 코드는 `ResInflowEnhan.xlsx` 하나를 읽습니다. 공개 코드가 없으므로 아래처럼 **CSV 3개**로 표준화하는 것을 권장합니다.

```
data/<dam>/inflow_daily.csv         # date, inflow_m3s [, inflow_fc_1d ... inflow_fc_7d]
data/<dam>/curves.csv               # kind(level_storage|tail_outflow), x, y
data/<dam>/rules.yaml               # 규정 파라미터 (아래 예시)
data/<dam>/historical_operation.csv # date, level_m, inflow_m3s, outflow_m3s, energy_1e8kWh
```

`rules.yaml` 예시:

```yaml
dam: ExampleDam
levels: {dead: 145.0, normal: 175.0, flood_limit: 145.0}
flood_season: {start: "06-01", end: "09-30"}
release: {q_min: 4000, q_max: 40000, turbine_max: 30000, max_daily_change_ratio: 0.3}
min_flows: {ecological: 6000, navigation: 5600, water_supply: 0}
power: {capacity_mw: 22500, guaranteed_mw: 4990, efficiency_k: 8.5}
```

### 3.4 데이터 품질 검사 코드

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
    assert report["years"] >= 20, "20년 이상 필요"
    assert report["nan_ratio"] < 0.05 and report["negative"] == 0
    return report
```

### 3.5 데이터 분할

- **학습:** 수문 연도 기준 70% (예: 1990–2014). 4년 단위 에피소드가 되도록 시작 연도를 무작위 추출.
- **검증:** 15% — 하이퍼파라미터·임계값 선택.
- **시험:** 15% + **극한 사상 연도**는 반드시 시험에 포함(학습 데이터에 없어야 일반화 확인 가능).
- **스트레스:** 실측 연도를 재배열·증폭(유입량 × 0.6, × 1.4 등)한 30년 합성 시나리오(부록 E 방식).

### 3.6 비데이터 조건

| 조건 | 최소 요건 |
|---|---|
| 컴퓨팅 | GPU 1장(논문 설정 300 에피소드 × 1460일 × 5 env, 변형당 수 시간). CPU만으로도 가능하나 수 배 느림 |
| 소프트웨어 | `tutorial_nbs/setup_env.sh` 환경(Python 3.10, DI-engine 0.5.3, torch) |
| 인력 | 수문·댐운영 전문가 1명(규정·보정), RL 엔지니어 1명(환경·학습), 운영자 대표(수용성) |
| 제도 | 권고안 사용에 대한 내부 승인 절차, 의사결정 로그 보관 규정 |

---

## 4. 3단계 — 환경 구축과 보정

### 4.1 구현

`toy_reservoir.py` 의 `ToyReservoirEnv` 를 복사해 `<dam>_env.py` 로 만들고 아래를 교체합니다.

| toy 구성요소 | 실제 댐에서 교체할 것 |
|---|---|
| `storage_from_level`, `level_from_storage` (선형) | 실측 수위–저수량 곡선 보간 (`np.interp`) |
| `tailwater_level(q)` (선형) | 실측 방류량–미수위 곡선 |
| `EFFICIENCY_K * Q * H` | 효율곡선 또는 수차 특성. 최소 낙차 미만 발전 정지 |
| `synthetic_inflow` | `inflow_daily.csv` 에서 에피소드 시작일을 샘플링해 잘라내기 |
| `upper_limit_level(doy)` | 규정의 날짜별 제한수위 표 |
| `cost_parts` 항목 | 규정의 제약 목록. 각 항목은 **0–1 로 정규화** (논문 `cost_normalizer`) |
| 관측 `future` | 실제 예보 또는 예보 오차 모델 |

**반드시 유지할 것:** `info["constraint_cost_normalized"]`(Lagrangian 입력), `info["action_mask"]`, `CONSTRAINT_TYPES` 의미, 관측 정규화.

### 4.2 보정(calibration)

과거 실제 방류량을 그대로 환경에 넣어 **수위와 발전량이 재현되는지** 확인합니다. 이것이 환경의 신뢰성 검증입니다.

```python
def replay_historical(env_cls, hist: pd.DataFrame, rules):
    """hist: date, level_m, inflow_m3s, outflow_m3s, energy_1e8kWh (실제 운영기록)"""
    env = env_cls(constraint_types=2)                      # 보정 시에는 페널티 불필요
    env.reset(init_level=hist.level_m.iloc[0], inflow_series=hist.inflow_m3s.values)
    sim = []
    for q in hist.outflow_m3s.values:
        a = int(np.argmin(np.abs(env.grid - q)))            # 실제 방류량에 가장 가까운 액션
        env.step(a)
        sim.append(dict(level=env.history[-1]["level"], energy=env.history[-1]["energy"]))
    sim = pd.DataFrame(sim, index=hist.index)
    err = dict(level_rmse_m=np.sqrt(((sim.level - hist.level_m) ** 2).mean()),
               energy_bias_pct=100 * (sim.energy.sum() / hist.energy_1e8kWh.sum() - 1))
    return sim, err
```

**합격 기준(권장):** 수위 RMSE < 0.5 m, 연간 발전량 편차 < 5%. 초과하면 곡선·효율·손실 항을 먼저 수정합니다.

### 4.3 단위 테스트

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

추가로: 사수위 이하로 내려가지 않는지, 마스크가 공집합이 되지 않는지, 관측이 [−1, 1] 범위인지.

---

## 5. 4단계 — 안전층(액션 마스크) 설계

### 5.1 규칙 계층

| 계층 | 논문 명칭 | 포함 규칙 | 근거 |
|---|---|---|---|
| basic | `MASK_RULE_TYPE='basic'` (B2) | 내일 수위 ∈ [사수위, 당일 상한수위] | 물리적·법적 절대 한계 |
| enhanced | `MASK_RULE_TYPE='enhanced'` (B3) | basic + 최소 방류량(생태·항운·급수) + 일 변화율 한도 + 홍수기 예비방류 규정 | 운영 규정집 |

논문 결과와 튜토리얼 노트북 03·04 는 **enhanced 마스크 + Lagrangian** 조합이 가장 안정적임을 보여줍니다. 실제 적용에서도 규정에 명시된 항목은 모두 마스크로 올리는 것이 원칙입니다.

### 5.2 설계 규칙

1. **공집합 금지:** 규칙이 충돌하면 우선순위(안전 > 법정 최소유량 > 변화율)로 완화하고, 최종 폴백은 "현재 수위 유지 방류". `toy_reservoir.action_mask()` 의 fallback 분기를 참고.
2. **예보 불확실성 반영:** 유입량 예보 상한(예: 예보 × 1.2)으로 내일 수위를 계산해 보수적으로 마스킹.
3. **마스크 통계 기록:** 매일 허용 액션 수를 로그. 부록 E 에서는 30년 평균 31/100 이었습니다. 10 미만이 지속되면 규칙이 과도하게 보수적입니다.

### 5.3 액션 매핑 선택

노트북 03 의 해상도 그래프로 **자기 댐의 일상 방류량 구간**에 액션이 충분히 배치되는 매핑을 고릅니다. 논문에서는 `square` 가 채택되었습니다(평수기 저방류 정밀도). 홍수 조절 비중이 크면 `sqrt`/`log` 도 비교하십시오. 이 선택은 설정 파일 `NONLINEAR_MAPPING_TYPE` 하나로 바뀝니다.

---

## 6. 5단계 — 학습

### 6.1 설정 파일 준비

`DRL_training_config/A3_Lagrangian_R1_config.py` 를 복사해 아래만 수정합니다.

| 항목 | 수정 |
|---|---|
| `filename_without_ext` | `<dam>_A3_R1_<timestamp>` |
| `MAX_EPISODE_STEP` | 에피소드 길이(예: 365×4) |
| `res_inflow_filename` | 자기 유입량 파일 |
| `ACTION_SHAPE`, `Q` 범위 | §2 결정값 |
| `reward_coeffs` | 처음엔 모두 1.0 |
| `lagrangian.constraint_threshold` | §6.3 |
| `ENABLE_ACTION_MASK=True`, `MASK_RULE_TYPE='enhanced'`, `NONLINEAR_MAPPING_TYPE` | §5 결정값 |

논문과 같은 실험 설계(A1/A2/A3 → B2/B3 → C2–C6)를 **자기 댐에서도 반복**하는 것을 권장합니다. 각 단계가 전 단계 대비 얼마나 개선되는지가 곧 적용 근거가 됩니다. 5개 시드(`--seed 0..4`)는 필수입니다.

### 6.2 학습 실행과 모니터링

```bash
conda activate ppo_rl
for s in 0 1 2 3 4; do python <dam>_B3_FullMask_config.py --seed $s; done
tensorboard --logdir result_<folder>/
```

TensorBoard 에서 확인할 곡선:

| 곡선 | 정상 | 이상 징후 |
|---|---|---|
| episode return | 상승 후 안정 | 계속 하락 → 페널티 과대, 계수 재조정 |
| `lagrangian/lambda` | 초기 상승 후 임계값 근처에서 진동·안정 | 무한 상승 → 임계값이 달성 불가능(마스크·규정 충돌) |
| `constraint_cost_normalized` | 임계값 아래로 수렴 | 임계값 위 정체 → 마스크 강화 또는 임계값 상향 |
| entropy | 서서히 감소 | 급락 → `entropy_weight` 증가 |

### 6.3 임계값과 승수 파라미터 정하기

- `constraint_threshold`(논문 0.05): 검증 데이터에서 **rule curve 운영의 정규화 비용**을 먼저 계산하고, 그 값의 50–100% 로 시작. 0 은 달성 불가능하며 λ 가 발산합니다.
- `learning_rate`(승수, 논문 0.01): λ 가 20–50 iteration 안에 반응하면 적절. 너무 크면 진동.
- `initial_penalty`(논문 1.0): rule curve 비용이 임계값을 크게 넘으면 2–5 로 시작.
- `dual_clip`: 실제 적용에서는 상한(예: 20)을 두어 발산 방지 권장.

### 6.4 학습 시간 추정

논문 설정(5 env × 60 에피소드 × 1460일 = 438,000 스텝, 3,000 gradient step)은 GPU 1장에서 변형·시드당 수 시간 수준입니다. 10 변형 × 5 시드 전체는 며칠이 걸리므로 A3 → B3 → C2 순으로 우선순위를 두십시오.

---

## 7. 6단계 — 평가와 스트레스 테스트

### 7.1 필수 지표 (부록 E 요약 시트와 동일)

```python
def evaluate_log(log: pd.DataFrame, rules) -> dict:
    """log: 일별 운영 로그 (level, inflow, release, energy, power_mw, cost_* 열)"""
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

### 7.2 베이스라인

최소 두 개와 비교합니다.
1. **현행 운영 곡선(rule curve)** — 동일 시나리오에서 규칙 기반 방류.
2. **과거 실제 운영** — 시험 기간의 실측 기록(부록 F 형식).
가능하면 3. **완전정보 최적해**(DP/LP, 미래 유입량을 아는 상한선)를 추가해 정책이 상한의 몇 %인지 보고합니다.

### 7.3 시험 프로토콜

| 시험 | 방법 | 합격 기준(예시) |
|---|---|---|
| 시험 연도 | 학습에 없는 실측 연도 ×5 시드 | 보증률 ≥ rule curve, 발전량 ≥ rule curve |
| 극한 스트레스 | 30년 합성(부록 E 방식), 유입량 ×0.6 / ×1.4 | 사수위·제한수위 위반 0일, 액션 위반율 < 0.1% |
| 예보 오차 | 예보에 ±20–30% 잡음(`obs_noise_*`) | 지표 열화 < 5% |
| 초기 조건 | 시작 수위 5개 × 시작 월 4개 | 모든 조합에서 위반 0 |
| 일반화 | 다른 댐 데이터(부록 F 방식) | 재학습 없이 위반 0 (선택) |
| 시드 분산 | 5 시드 평균±표준편차 보고 | 표준편차 < 평균의 5% |

### 7.4 의사결정 로그 형식

부록 E 의 30개 열(일수, 수위, 미수위, 저수량, 유입, 방류, 발전유량, 여수, 발전량, 보증출력, 물수지, 정책 액션, 실제 액션, 마스크 최소·최대, 액션 위반, 보상 및 6개 페널티, 6개 관측)을 그대로 채택하면 논문과 직접 비교할 수 있습니다. 노트북 05 의 `COLS_E` 가 영문 열 이름입니다.

---

## 8. 7단계 — 의사결정 지원 배치

### 8.1 일일 파이프라인

```
06:00  관측 수집(수위, 어제 유입량) + 유입량 예보(1–7일) 수신
06:10  관측 벡터 구성 → 마스크 계산 → 정책 추론(5 시드 앙상블)
06:15  권고안 생성: 권고 방류량, 허용 범위, 예상 수위·발전량, 위반 위험 항목
06:30  운영자 검토 → 승인/수정 → 실제 방류 결정
익일   실측 반영, 의사결정 로그 저장, 권고 vs 실제 차이 기록
```

### 8.2 추론 코드

```python
import torch, numpy as np

class Recommender:
    def __init__(self, env, models, mapping="square"):
        self.env, self.models = env, models          # models: 5개 시드의 학습된 ActorCritic

    @torch.no_grad()
    def recommend(self, level, inflow_today, inflow_fc, date):
        self.env.set_state(level=level, inflow=inflow_today, forecast=inflow_fc, date=date)
        obs, mask = self.env.observe(), self.env.action_mask()
        probs = np.mean([m(torch.as_tensor(obs), torch.as_tensor(mask))[0].probs.numpy() for m in self.models], axis=0)
        a = int(probs.argmax())
        q_rec = float(self.env.grid[a])
        q_lo, q_hi = self.env.grid[mask].min(), self.env.grid[mask].max()
        z_next = self.env.preview_level(q_rec)          # 물수지로 내일 수위 미리 계산
        return dict(release_m3s=q_rec, allowed_range=(q_lo, q_hi), expected_level=z_next,
                    confidence=float(probs[a]), n_allowed=int(mask.sum()))
```

(`set_state`, `observe`, `preview_level` 은 `<dam>_env.py` 에 추가할 메서드입니다. `ToyReservoirEnv` 의 `_obs`, `action_mask`, `feasible_release_range` 를 그대로 활용하면 됩니다.)

### 8.3 운영 안전장치

| 장치 | 내용 |
|---|---|
| 사람 승인 | 권고안은 운영자가 승인해야 실행. 시스템은 방류 설비를 직접 제어하지 않음 |
| 폴백 | 관측 결측, 마스크 공집합, 모델 오류, 권고안이 허용 범위 밖 → rule curve 방류량 제시 |
| 앙상블 불일치 경고 | 5 시드의 권고 방류량 표준편차 > 10% 이면 "불확실" 표시 |
| 홍수 경보 모드 | 예보 유입량이 설계 홍수의 일정 비율 초과 시 정책 권고를 비활성화하고 홍수 조절 규정만 적용 |
| 로그 보관 | 입력·마스크·권고·실제·사유를 모두 저장(부록 E 형식 + 승인자) |

---

## 9. 8단계 — 운영 중 모니터링과 재학습

| 감시 항목 | 기준 | 조치 |
|---|---|---|
| 권고 vs 실제 방류량 차이 | 30일 평균 > 15% | 운영자 인터뷰 → 보상 계수·마스크 재검토 |
| 환경 예측 오차 | 예상 수위 − 실측 수위 > 0.3 m 지속 | 특성곡선 재보정(퇴사 등) |
| 유입량 분포 변화 | 최근 5년 월평균이 학습 분포의 ±2σ 밖 | 최근 데이터 포함해 재학습 |
| 규정 개정 | 제한수위·최소유량 변경 | 마스크 즉시 수정(학습 불필요) → 성능 재평가 → 필요 시 재학습 |
| 정기 재학습 | 연 1회 | 최신 5년 추가, 5 시드, §7 프로토콜 재통과 |

재학습 시에도 **설정 파일을 그대로 보관**하고(이 저장소의 방식), 어떤 데이터·시드·규정으로 학습했는지 추적 가능하게 유지합니다.

---

## 10. 분석 코드 위치 요약

| 분석 | 코드 |
|---|---|
| 설정 파일 로드·비교 | 노트북 01 `load_config()`, 스위치 표 |
| 환경 물수지·제약 비용 | `toy_reservoir.py`, 노트북 02 |
| 마스크·매핑 시각화 | 노트북 03 |
| PPO / PPO-Lagrangian 학습 루프(참고 구현) | 노트북 04 `train()` |
| 부록 로그 읽기·지표 | 노트북 05 `COLS_E`, `LABELS` |
| 데이터 품질 검사 | 이 문서 §3.4 |
| 환경 보정·단위 테스트 | 이 문서 §4.2–4.3 |
| 평가 지표 | 이 문서 §7.1 |
| 일일 추론 | 이 문서 §8.2 |

---

## 11. 한계와 주의사항

- **toy 환경은 실제 댐이 아닙니다.** 튜토리얼의 수치·결과는 메커니즘 이해용이며, 실제 적용 성능을 보장하지 않습니다.
- **논문 코드 비공개:** 환경·정책 코드는 재구현이 필요합니다. 설정 파일은 하이퍼파라미터의 근거 자료로만 사용하십시오.
- **분포 외 상황:** 학습 데이터에 없는 극한 사상에서 정책의 거동은 보장되지 않습니다. 마스크와 홍수 경보 모드가 이를 보완하지만, 규정 준수의 최종 책임은 운영자에게 있습니다.
- **단일 목적 편향:** 보상 계수를 모두 1.0 으로 두면 발전량이 우세할 수 있습니다. 급수·환경 목적이 중요하면 계수 민감도 분석을 수행하십시오.
- **법적 지위:** 대부분의 관할권에서 AI 권고안은 참고 자료이며, 방류 결정의 법적 책임 주체는 변하지 않습니다.

---

## 12. 적용 체크리스트

- [ ] §2 문제 정의표 확정, 규정 조항 대응표 작성
- [ ] 유입량 ≥20년, 결측 <5%, 특성곡선 최신본, 운영 규정 파라미터화(`rules.yaml`)
- [ ] 환경 보정: 수위 RMSE <0.5 m, 발전량 편차 <5%
- [ ] 물수지·마스크 공집합·관측 범위 단위 테스트 통과
- [ ] A3 → B3 → C2 순 학습, 5 시드, TensorBoard 곡선 정상
- [ ] 시험 연도·스트레스·예보 오차·초기 조건 프로토콜 통과, rule curve 대비 개선 확인
- [ ] 의사결정 지원 파이프라인, 폴백, 로그, 승인 절차 구축
- [ ] 모니터링 기준과 재학습 주기 문서화
