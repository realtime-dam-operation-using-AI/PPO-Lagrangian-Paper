# DRL 학습 설정 파일 (공개 배포용)

> 🇬🇧 English version: [README.md](README.md)

이 폴더에는 아래 논문의 실험에 사용된 **학습 설정 파일(training configuration files)** 이 포함되어 있습니다.

> **A Two-Layer Safe Reinforcement Learning Framework for Multi-Constraint Reservoir Operation: Integrating Action Masking and Lagrangian Dual Method**

구현의 **투명성과 재현성**을 높이기 위해 이 설정 파일들을 공개합니다 (구현 세부사항에 대한 리뷰어 의견 반영).

## 포함 내용

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

각 `*_config.py`는 독립적인 DI-engine 스타일 설정 파일입니다 (Python + `EasyDict`). 대부분의 설정 파일에는 직접 실행할 수 있도록 선택적인 `__main__` 진입점이 포함되어 있습니다.

`AppE_ExtremeStressEval_C2Square_R1_Episode2.xlsx`는 부록 E에 보고된 극한 스트레스 테스트 평가표(단일 실행 예시)입니다. 두 개의 시트로 구성됩니다.

- 시트 1, `삼협(Three Gorges)`: 일 단위 운영 과정 로그(시계열 기록)
- 시트 2, `요약(Summary)`: 부록 요약표에 사용된 핵심 집계 지표

`AppF_CrossReservoir_DecisionLogs_Public_Xiluodu_Xiangjiaba.xlsx`에는 부록 F에 보고된 저수지 간(cross-reservoir) 의사결정 로그가 들어 있습니다.

## 파일명 해석 방법

파일명 패턴:

```
<Group>_<Variant>_R<k>_config.py
```

- `Group`은 **논문의 실험 그룹**에 대응합니다.
	- **A***: 제약 처리 방식 비교
	- **B***: 액션 마스킹 비교
	- **C***: 비선형 액션 이산화/매핑 비교
- `R1`–`R5`는 논문에 보고된 **5회의 독립 실행**(5개의 랜덤 시드)을 의미합니다.

### 논문 설정과의 대응

| 논문 항목 | 이 폴더 | 설정 파일의 핵심 스위치 |
|---|---|---|
| A1 (Correction) | `A1_Correction_R*_config.py` | `USE_PPO_LAGRANGIAN=False` (또는 policy type = PPO) + 환경 보정(correction) 활성화 |
| A2 (Penalty) | `A2_Penalty_R*_config.py` | 고정 페널티 / reward shaping 베이스라인 |
| A3 (PPO-Lagrangian) | `A3_Lagrangian_R*_config.py` | `USE_PPO_LAGRANGIAN=True` |
| B2 (BaseMask) | `B2_WaterMask_R*_config.py` | `ENABLE_ACTION_MASK=True`, `MASK_RULE_TYPE='basic'` |
| B3 (FullMask) | `B3_FullMask_R*_config.py` | `ENABLE_ACTION_MASK=True`, `MASK_RULE_TYPE='enhanced'` |
| C1 (선형 매핑) | **`B3_FullMask_R*_config.py` 사용** | `ENABLE_NONLINEAR_MAPPING=False` (선형 베이스라인) |
| C2–C5 (비선형 매핑) | `C2_*`–`C6_*` 설정 파일 | `ENABLE_NONLINEAR_MAPPING=True`, `NONLINEAR_MAPPING_TYPE=...` |

참고:
- 우리 코드베이스에서 실험 C의 선형 베이스라인은 비선형 매핑을 **비활성화**(`ENABLE_NONLINEAR_MAPPING=False`)하는 방식으로 구현되어 있으므로, `B3_FullMask_*` 설정 파일을 재사용합니다.
- `C2...C6` 파일 인덱스는 내부 명명 규칙이며, 매핑 유형은 `NONLINEAR_MAPPING_TYPE`으로 명확히 확인할 수 있습니다.

## 설정 파일에 기록된 주요 구현 세부사항

이 설정 파일들에는 다음 내용이 명시적으로 기록되어 있습니다 (일부만 나열).

### 신경망 구조

`policy.model` 블록 (예: `A3_Lagrangian_R1_config.py`):

- 인코더를 공유하는 Actor–Critic 구조 (DI-engine VAC 스타일)
- `encoder_hidden_size_list = [512, 512, 256]`
- `actor_head_layer_num = 1`, `critic_head_layer_num = 1`
- `actor_head_hidden_size = 256`, `critic_head_hidden_size = 256`
- 이산 액션 공간, `action_shape = 100`

### PPO 학습 파이프라인

기록된 설정 예시:

- `collector_env_num = 5`
- collector 환경당 `episode_num = 60`
- `max_episode_step = 365*4` (일 단위 스텝, 4년 기간)
- `epoch_per_collect = 10`
- `batch_size = 365*4`
- `n_sample = int(max_episode_step/2 * collector_env_num)`
- `learning_rate = 0.001`, `clip_ratio = 0.2`, `discount_factor = 0.99`, `gae_lambda = 0.95`
- `entropy_weight = 0.016`, `value_weight = 0.6`
- `adv_norm = True`, `value_norm = True`

**중요한 설명 (에피소드 수):**

논문에서는 **실행당 총 학습 에피소드 수**를 보고합니다. 이 설정 파일에서 `episode_num`은 **collector 환경당** 값입니다. `collector_env_num=5`, `episode_num=60`이므로 총 에피소드 수는 다음과 같습니다.

$$
N_{\text{episodes,total}} = N_{\text{collector env}} \times N_{\text{episode per env}} = 5 \times 60 = 300.
$$

이 때문에 설정 파일에는 `MAX_EPISODE = 60`으로 표시되어 있지만 논문에서는 "300 에피소드"로 언급될 수 있습니다.

### Lagrangian (primal–dual) 설정

PPO-Lagrangian 메커니즘은 `policy.lagrangian` 블록으로 제어됩니다. 예:

- `constraint_key = 'constraint_cost_normalized'`
- `constraint_threshold = 0.05`
- `learning_rate = 0.01`
- `update_interval = 4`
- `initial_penalty = 1.0`

## 실행 방법 (코드베이스가 있는 경우)

이 설정 파일들은 DI-engine 기반 학습 파이프라인(이 논문용 폴더에는 포함되지 않음)과 함께 사용하도록 설계되었습니다.

해당 코드베이스가 있다면 일반적으로 다음과 같이 학습을 시작할 수 있습니다.

- 설정 파일을 직접 실행하고 시드를 전달:
	- `python A3_Lagrangian_R1_config.py --seed 0`

`R1`–`R5`는 5회의 독립 실행에 대응하며, 어떤 5개의 시드를 사용해도 됩니다 (별도의 시드 목록을 유지하지 않는다면 `0,1,2,3,4`를 권장).

## 튜토리얼

`tutorial_nbs/` 에는 Jupyter 노트북 5개(설정 파일 해부, toy 저수지 환경, 액션 마스킹/비선형 매핑, PPO vs PPO-Lagrangian, 부록 데이터)와 설정 파일이 import 하는 모든 의존성을 갖춘 `ppo_rl` conda 환경을 만드는 `setup_env.sh` 가 있습니다. `tutorial_nbs/README_kor.md` 를 참고하세요. `hourly_operation_nbs/` 는 이를 홍수/가뭄 우선순위, 월별 운영수위 밴드, 운영자 선호를 반영한 시간 단위 의사결정 지원으로 확장한 것입니다(해당 README 참고).

## 데이터 및 모델 가중치

- **과거 유입량 데이터**는 기관의 접근 제한 대상이므로 여기에 **포함되지 않습니다**.
- 설정 파일은 학습 환경이 요구하는 유입량 파일명(예: `ResInflowEnhan.xlsx`)을 참조합니다.
- 부록 수준의 보조 출력 표는 다음 파일로 공개됩니다.
	- `AppE_ExtremeStressEval_C2Square_R1_Episode2.xlsx`
	- `AppF_CrossReservoir_DecisionLogs_Public_Xiluodu_Xiangjiaba.xlsx`
- **학습된 모델 체크포인트/가중치**와 전체 실험 출력은 교신저자에게 합리적인 요청 시 제공받을 수 있습니다 (논문의 Data availability statement 참조).

## GitHub 배포

이 설정 파일과 보조 데이터는 다음 주소에 공개되어 있습니다.

- https://github.com/lovemevol/PPO-Lagrangian-Paper

## 문의

제한된 데이터 접근 및 학습된 모델 파라미터에 관해서는 논문에 명시된 교신저자에게 문의해 주십시오.
