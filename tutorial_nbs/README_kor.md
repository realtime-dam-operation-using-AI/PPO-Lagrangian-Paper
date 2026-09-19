# 튜토리얼 노트북

> 🇬🇧 English version: [README.md](README.md)

논문 *"A Two-Layer Safe Reinforcement Learning Framework for Multi-Constraint Reservoir Operation"* 과 공개된 학습 설정 파일을 이해하기 위한 실습 노트북입니다. 논문의 환경/정책 코드는 비공개이므로, 같은 메커니즘(다중 제약 보상, `CONSTRAINT_TYPES`, 액션 마스킹, 비선형 액션 매핑, PPO-Lagrangian 이 쓰는 정규화 제약 비용)을 재현한 작은 **toy 저수지 환경**(`toy_reservoir.py`)을 사용합니다.

## 환경 설정

```bash
bash tutorial_nbs/setup_env.sh      # conda env "ppo_rl"(Python 3.10) 생성 + Jupyter 커널 등록
conda activate ppo_rl
jupyter lab tutorial_nbs            # "Python (ppo_rl)" 커널 선택
```

이 스크립트는 `DRL_training_config/*_config.py` 가 import 하는 모든 패키지(`easydict`, `pytz`, `torch`, `gym==0.25.1`, `DI-engine==0.5.3`)와 `pandas`, `openpyxl`, `matplotlib`, `jupyter` 를 설치합니다. PyTorch 는 RTX 50 시리즈 GPU 에서 동작하도록 CUDA 12.8 휠 인덱스에서 설치되며, GPU 가 없으면 자동으로 CPU 로 동작합니다. DI-engine 0.5.3 이 Python 3.7–3.10 만 지원하고 `gym==0.25.1`, `numpy<2` 를 고정하므로 Python 3.10 을 사용합니다.

모든 노트북을 명령줄에서 다시 실행하려면:

```bash
conda activate ppo_rl && cd tutorial_nbs
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$nb"; done
```

## 노트북 목록

| # | 노트북 | 배우는 내용 |
|---|---|---|
| 01 | `01_config_anatomy.ipynb` | 비공개 `agent` 패키지를 stub 으로 대체해 실제 `*_config.py` 를 로드하고, 하이퍼파라미터 표, "300 에피소드" 재계산, 10개 변형 사이의 단일 스위치 체인, DI-engine `PPOPolicy` 기본값과의 비교. |
| 02 | `02_reservoir_environment.ipynb` | 물수지, 합성 유입량, 7개 보상 항, 6차원 관측(타입 5), `CONSTRAINT_TYPES` 1/2/3 의 의미. |
| 03 | `03_action_mask_and_mapping.ipynb` | `basic` vs `enhanced` 액션 마스크(실험 B), 선형 vs square/sqrt/cubic/exp/log 액션 격자(실험 C). |
| 04 | `04_ppo_vs_ppo_lagrangian.ipynb` | 설정 파일의 `policy.lagrangian` 파라미터를 그대로 쓰는 PyTorch PPO / PPO-Lagrangian(primal–dual) 구현. toy 환경에서 A1, A2, A3, B3, C2 유사 설정을 학습해 보상, 제약 비용, 승수 λ 를 비교. |
| 05 | `05_appendix_data.ipynb` | 공개된 부록 E(`C2_Square_R1` 30년 스트레스 테스트)와 부록 F(시뤄두/샹자바) 로그를 영문 열 이름으로 읽고 학습된 운영 패턴을 시각화. |

노트북 04 는 기본값 `N_ITERS = 40` 으로 몇 분 걸립니다. 값을 늘리면 더 수렴된 곡선을 볼 수 있습니다.

## 실제 댐 운영에 적용하기

`operation_guide_kor.md` 는 목적, 필요한 데이터와 조건, 환경 보정, 마스크 설계, 학습, 평가 프로토콜, 의사결정 지원 배치, 모니터링까지의 단계별 절차와 실행 가능한 분석 코드를 담은 적용 가이드입니다.

## 파일

- `toy_reservoir.py` — 노트북 02–04 가 공유하는 toy 환경과 헬퍼. 논문의 환경이 아닙니다.
- `setup_env.sh`, `requirements.txt` — 환경 정의.
- `operation_guide.md` / `operation_guide_kor.md` — 실제 댐 적용 절차서.

## 실제 설정 파일을 여기서 실행할 수 있나요?

아니요. `../DRL_training_config/description_kor.md` 를 참고하세요. `ppo_rl` 환경에서 `python A3_Lagrangian_R1_config.py --seed 0` 을 실행하면 서드파티 import 는 통과하지만 비공개 코드베이스인 `agent` 에서 `ModuleNotFoundError: No module named 'agent'` 로 실패합니다.
