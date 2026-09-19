# DRL_training_config — 파일 설명

> 🇬🇧 English version: [description.md](description.md)

이 폴더에는 논문 실험에 사용된 학습 설정 파일 50개가 있습니다. 실험 변형 10종 × 독립 실행 5회(`R1`–`R5`)입니다. 각 파일은 DI-engine 스타일 설정(Python + `EasyDict`)으로, `main_config`와 `create_config` 두 객체를 만들고 `__main__`으로 실행하면 학습 파이프라인에 전달합니다.

## 공통 구조

50개 파일은 모두 같은 본문을 공유합니다. 위에서 아래로 읽으면 다음과 같습니다.

| 구간 | 정의하는 내용 |
|---|---|
| import 및 `sys.path` 설정 | `../../../`, `../../../../`를 경로에 추가하여 비공개 DI-engine 코드베이스(`agent.*`, `ding`)를 import 할 수 있게 합니다. 커스텀 정책 `agent.tool_function.ppo_lagrangian`과 Gym 환경 `ResevoirAgent-v0`를 등록합니다. |
| `filename_without_ext` | 원래 학습 타임스탬프가 붙은 실행 이름. 같은 변형의 `R1`–`R5` 사이에서 유일하게 다른 줄입니다. |
| 학습 규모 상수 | collector 환경 `CO_NUM=5`, 환경당 `MAX_EPISODE=60`(총 300), 일 단위 `MAX_EPISODE_STEP=365*4`, `N_SAMPLE`, `BATCH_SIZE`, `UPDATE_NUM=10`, `LR=0.001`. |
| 보상 계수 | `POWER_`, `LEVEL_`, `FLOOD_`, `ECOLOGICAL_`, `POWERLIMIT_`, `NAVIGATION_`, `ACTION_REWARD_COEFF`, 모두 `1.0`. |
| PPO-Lagrangian 블록 | `USE_PPO_LAGRANGIAN`, 임계값 `0.05`, 승수 학습률 `0.01`, 초기 페널티 `1.0`, 미니배치 4회마다 갱신, 양의 위반에만 ReLU 적용. |
| 실험 스위치 | `CONSTRAINT_TYPES`, `OBSERVATION_TYPES=5`, `ACTION_SPACE_TYPE='discrete'`와 `ACTION_SHAPE=100`, `ENABLE_ACTION_MASK`, `MASK_RULE_TYPE`, `ENABLE_NONLINEAR_MAPPING`, `NONLINEAR_MAPPING_TYPE`. |
| 비활성화된 부가 기능 | `ADVERSARIAL_*`, `CURRICULUM_LEARNING`, `OBS_NOISE_ENABLE`는 모두 `False`. 비공개 코드베이스에 존재하지만 논문에서는 사용하지 않았습니다. |
| `main_config` | `env`(에피소드 수, 보상 계수, 유입량 파일 `ResInflowEnhan.xlsx`, `experimental_setup`, 데이터 내보내기, TensorBoard)와 `policy`(모델 크기, PPO learn/collect/eval 설정, `lagrangian` 블록). |
| `create_config` | 환경 유형 `resevoiragent-v2`, `base` 환경 매니저, 정책 유형 `ppo_lagrangian` 또는 `ppo`. |
| `__main__` | `--seed`를 파싱하고 정책 클래스를 고른 뒤 `resrevoiragent_pipeline_onpolicy([main_config, create_config], seed=...)`를 호출합니다. |

모델: 인코더를 공유하는 Actor–Critic, 인코더 `[512, 512, 256]`, 각 head에 256 크기 은닉층 1개, 관측 6차원, 이산 액션 100개.

## 10가지 변형

변형들은 체인 구조입니다. 각 변형은 바로 앞 변형에서 스위치 하나만 바꿉니다. `A3_Lagrangian`이 B, C 그룹의 기준입니다.

### 그룹 A — 제약 처리 방식 (논문 실험 A)

| 파일 접두어 | `USE_PPO_LAGRANGIAN` | `CONSTRAINT_TYPES` | 의미 |
|---|---|---|---|
| `A1_Correction` | `False` | `2` | 일반 PPO. 액션이 제약을 위반하면 환경이 보정하되 페널티는 주지 않습니다. |
| `A2_Penalty` | `False` | `3` | 일반 PPO. 환경이 액션을 보정하고 보상에 고정 페널티를 더합니다 (reward-shaping 베이스라인). |
| `A3_Lagrangian` | `True` | `3` | PPO-Lagrangian. 보정과 페널티 신호는 같지만, 페널티 가중치가 라그랑주 승수이며 정규화된 제약 비용을 `0.05` 이하로 유지하도록 쌍대 방법으로 갱신됩니다. |

`CONSTRAINT_TYPES` 값: 1 = 위반 시 에피소드 종료(미사용), 2 = 보정만, 3 = 보정 + 페널티.

### 그룹 B — 액션 마스킹 (논문 실험 B)

둘 다 `A3_Lagrangian`(`USE_PPO_LAGRANGIAN=True`, `CONSTRAINT_TYPES=3`) 위에 구성됩니다.

| 파일 접두어 | `ENABLE_ACTION_MASK` | `MASK_RULE_TYPE` | 의미 |
|---|---|---|---|
| `B2_WaterMask` | `True` | `'basic'` | 샘플링 전에 기본 수위/유량 한계를 위반하는 액션을 마스킹합니다. 논문의 "BaseMask". |
| `B3_FullMask` | `True` | `'enhanced'` | 삼협(Three Gorges) 운영 규정 제약을 마스크에 추가합니다. 논문의 "FullMask". **C1 선형 매핑** 베이스라인으로도 사용됩니다. |

논문의 B1(마스크 없음) 설정은 `A3_Lagrangian`입니다.

### 그룹 C — 비선형 액션 매핑 (논문 실험 C)

모두 `B3_FullMask` 위에 구성되며 `ENABLE_NONLINEAR_MAPPING=True`로 설정합니다. 100개의 이산 액션을 선형이 아니라 선택한 함수로 연속 방류량 범위에 매핑합니다.

| 파일 접두어 | `NONLINEAR_MAPPING_TYPE` |
|---|---|
| `C2_Square` | `'square'` |
| `C3_Sqrt` | `'sqrt'` |
| `C4_Cubic` | `'cubic'` |
| `C5_Exp` | `'exp'` |
| `C6_Log` | `'log'` |

선형 베이스라인(논문 C1)은 `ENABLE_NONLINEAR_MAPPING=False`인 `B3_FullMask`이며, 별도의 `C1_*` 파일은 없습니다.

### 실행 `R1`–`R5`

한 변형의 파일 5개는 `filename_without_ext`만 다릅니다. 랜덤 시드는 파일에 저장되지 않고 명령줄로 전달합니다(`--seed 0` … `--seed 4`).

## 이 파일들을 단독으로 실행할 수 있나요?

**아니요, 이 저장소만으로는 실행할 수 없습니다.** 각 파일은 비공개 DI-engine 기반 학습 파이프라인의 진입점이며, 다음이 모두 필요하지만 여기에는 하나도 포함되어 있지 않습니다.

1. Python 패키지 `easydict`, `pytz`, `torch`, `gym`, DI-engine(`ding`).
2. 이 폴더보다 두 단계 위에 있어야 하는 비공개 패키지 `agent.*`:
   - `agent.tool_function.ppo_lagrangian` — PPO-Lagrangian 정책 클래스
   - `agent.reservoir_single_agent.envs.reservoir_env_gym`, `reservoir_env_v2` — 저수지 시뮬레이션 환경(물수지, 수력발전, 제약, 마스킹, 비선형 매핑)
   - `agent.reservoir_single_agent.entry.resevoiragent_entry_onpolicy` — 학습 루프
3. 기관 접근 제한 대상인 과거 유입량 파일 `ResInflowEnhan.xlsx`.

여기서 파일을 실행하면 첫 줄에서 `ModuleNotFoundError: No module named 'easydict'`로 멈추고, 이를 설치하더라도 `from agent.tool_function import ppo_lagrangian`에서 실패합니다. 실제로 학습하려면 이 파일들을 원본 코드베이스의 `DI-engine/agent/reservoir_single_agent/<folder>/` 안에 두고, 유입량 데이터를 확보한 뒤 `python <config>.py --seed <k>`로 실행해야 합니다.
