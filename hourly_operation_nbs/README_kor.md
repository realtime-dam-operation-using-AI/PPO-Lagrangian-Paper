# 시간 단위 댐 운영 노트북

> 🇬🇧 English version: [README.md](README.md)

논문의 2층 안전 강화학습 프레임워크(액션 마스킹 + PPO-Lagrangian)를 **시간 단위 운영 의사결정 지원**으로 확장한 것으로, 운영기관의 요구사항을 다음과 같이 구현했습니다.

| # | 요구사항 | 구현 위치 |
|---|---|---|
| 1 | 제약 우선순위: 홍수 > 가뭄 > 기타, 발전보다 우선 | `HourlyReservoirEnv.mask_intervals()` 의 우선순위 마스크 층; `ppo_lagrangian.train()` 의 두 라그랑주 승수(홍수, 가뭄)와 서로 다른 임계값 |
| 2 | 발전은 제약이 아님 | 발전은 보상의 작은 항(`w_energy=0.05`)일 뿐 비용·마스크에 없음 |
| 3 | 입력: 댐수위, 하류 주요지점 유량, 기상예측 기반 유입량 | 19차원 관측(`HourlyReservoirEnv.OBS_NAMES`) |
| 4 | 매월 1일 기준 상·하한 운영수위, 밴드 안에서 운영 | `UPPER_GUIDE`, `LOWER_GUIDE` 를 `guide_levels()` 로 보간; 밴드 비용 |
| 5 | 운영자가 상한 근처 / 중간 / 하한 근처 선택 | `target_position ∈ [0,1]` 을 정책 입력으로; 하나의 모델이 모든 선호를 처리 |
| 6 | 72시간 hourly 홍수예측, 이후 일 단위 예측; 연간 안정성 검토 | `make_forecasts()` (72h hourly + 7일 daily), 노트북 04·05 |

모든 데이터는 합성입니다. 실제 데이터로 교체할 표와 생성기는 `hourly_reservoir.py` 와 노트북 01 마지막 절에 정리되어 있습니다.

## 환경 설정

`tutorial_nbs/` 와 같은 conda 환경(`bash tutorial_nbs/setup_env.sh`, 커널 "Python (ppo_rl)")을 사용합니다. 이 폴더를 작업 디렉터리로 두고 순서대로 실행하십시오. 노트북 03 이 `models/policy.pt` 를 저장하고 04, 05 가 이를 불러옵니다.

```bash
conda activate ppo_rl && cd hourly_operation_nbs
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"; done
```

## 노트북

| # | 노트북 | 내용 |
|---|---|---|
| 01 | `01_problem_setup_and_data.ipynb` | 댐 제원, 월별 운영수위 밴드와 운영자 목표, 합성 시간 단위 유입량, 두 종류의 예측과 정확도, 관측 벡터, 실제 데이터 교체표 |
| 02 | `02_hourly_environment_and_rule_baseline.ipynb` | 최근접 경계 완화 방식의 우선순위 마스크(홍수 예시), 비용 구조, 목표수위 추종 규칙 베이스라인 1년 × 선호 3가지, 홍수 사상 확대 |
| 03 | `03_train_ppo_lagrangian.ipynb` | 규칙을 모방한 행동복제 초기화, 두 승수를 가진 PPO, 학습 곡선, 규칙과의 연간 비교 |
| 04 | `04_decision_support_72h.ipynb` | 아침 상황 → 72시간 방류 계획, 30개 예측 앙상블, 위험 확률, 의사결정 표, 운영자 선호별·규칙 대비 비교, 배치 절차 |
| 05 | `05_annual_stability_review.ipynb` | 건조/평년/습윤년 × 선호 3가지 × (정책, 규칙), 월별 안정성 표, 예측 오차 민감도, 검토 보고서 작성 요령 |

## 모듈

- `hourly_reservoir.py` — 환경, 예측, 마스크, 규칙 베이스라인, 우선순위 순 지표 `summarize()`.
- `ppo_lagrangian.py` — 배치 PPO-Lagrangian(승수 2개), 리턴 정규화, 행동복제 사전학습, `load_policy()`, `PolicyAgent`.

## 주요 설계 결정

- **마스크 층을 버리지 않고 완화**: 층을 우선순위 순으로 구간 교집합하고, 만족할 수 없는 낮은 층은 가장 가까운 경계로 완화합니다. 댐 안전은 항상 지켜지고 하류·변화율 한도는 최소한으로만 위반됩니다. 로그의 `mask_relaxed` 열이 그런 시각을 기록합니다.
- **60개 액션, square 매핑**: 갈수기 유입량이 20–60 m³/s 이므로 최소 방류 근처의 해상도가 필요합니다.
- **운영수위 밴드는 최하위(5순위) 하드 마스크 층**(`band_hard=True`). 안전·하류·유지유량·변화율과 충돌하지 않는 한 요구사항 4 를 기계적으로 보장하며, 동시에 소프트 비용으로도 들어갑니다. `band_hard=False` 로 두면 소프트 비용만 남습니다.
- **워밍업과 밴드 밖 시작**: 무작위 정책은 하루 만에 저수지를 비우므로 규칙을 행동복제한 정책에서 PPO 를 시작하고, 에피소드 초기수위를 하한 −1 m ~ 상한 +2.5 m 범위에서 샘플링해 홍수·가뭄 후 복귀 운영을 학습시킵니다.
