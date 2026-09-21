# CLAUDE_kor.md

> 🇬🇧 English version: [CLAUDE.md](CLAUDE.md) — 영문판이 원본이며, 두 파일은 항상 동일한 내용을 유지해야 합니다.

이 파일은 이 저장소에서 작업하는 Claude Code (claude.ai/code)를 위한 지침을 제공합니다.

## 이 저장소는 무엇인가

논문 *"A Two-Layer Safe Reinforcement Learning Framework for Multi-Constraint Reservoir Operation: Integrating Action Masking and Lagrangian Dual Method"* 의 공개 보조 자료입니다. 포함된 내용은 다음뿐입니다.

- `DRL_training_config/*_config.py` — DI-engine 스타일 학습 설정 파일 50개 (실험 변형 10종 × 시드 5개).
- 부록 `.xlsx` 표 2개 (부록 E 스트레스 테스트 로그: 시트 2개, `삼협(Three Gorges)` 일 단위 로그와 `요약(Summary)`; 부록 F 저수지 간 의사결정 로그).
- `README.md` / `README_kor.md` — 논문 실험과 설정 파일의 공식 대응표. 설정 파일을 수정하기 전에 먼저 읽을 것.
- `DRL_training_config/description.md` / `description_kor.md` — 변형별 설정 파일 설명과 단독 실행이 불가능한 이유.
- `tutorial_nbs/` — Jupyter 노트북 5개와 `toy_reservoir.py`. 논문의 메커니즘(제약 처리 타입, 액션 마스크, 비선형 매핑, 제약 비용)을 재현한 작은 대체 환경입니다. 각 노트북 설명은 해당 폴더의 README 참고. `operation_guide_kor.md` 는 실제 댐 적용 절차서입니다.
- `hourly_operation_nbs/` — 시간 단위 의사결정 지원 확장(`hourly_reservoir.py`, `ppo_lagrangian.py`, 노트북 5개). 운영기관 요구사항(홍수 > 가뭄 우선, 발전 비제약, 월별 운영수위 밴드, 운영자 목표 위치, 72h hourly + daily 예측) 기반. 노트북 03 이 `models/policy.pt` 를 쓰고 04, 05 가 필요로 함.

설정 파일 자체에 대한 빌드, 린트, 테스트 도구는 없습니다. 설정 파일은 **이 저장소에서 실행할 수 없습니다.** `sys.path`를 `../../../`, `../../../../`로 조작하여 `DI-engine/agent/reservoir_single_agent/<folder>/` 위치에 있다고 가정하며, `agent.tool_function.ppo_lagrangian`, `ding`, 비공개 저수지 환경을 import 합니다. 유입량 데이터(`ResInflowEnhan.xlsx`)와 학습된 가중치도 포함되어 있지 않습니다. 단독 실행이 되도록 import나 경로를 "고치려" 하지 마십시오. 이 파일들은 실제로 학습에 사용된 설정을 그대로 기록한 것으로 공개된 것입니다.

비공개 코드베이스가 있다면 의도된 실행 방법은 다음과 같습니다.

```
python A3_Lagrangian_R1_config.py --seed 0
```

## 튜토리얼 환경과 노트북

```bash
bash tutorial_nbs/setup_env.sh                 # conda env "ppo_rl"(Python 3.10) + Jupyter 커널 "Python (ppo_rl)"
conda activate ppo_rl
jupyter lab tutorial_nbs
# 노트북 하나를 headless 로 재실행 (04 는 학습에 몇 분 소요)
cd tutorial_nbs && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 01_config_anatomy.ipynb
```

- Python 3.10 은 의도적인 선택입니다. DI-engine 0.5.3 은 3.7–3.10 만 지원하며 `gym==0.25.1`, `numpy<2`, `setuptools<=66.1.1` 을 고정합니다. gymnasium 이나 numpy 2 로 올리지 마십시오.
- 개발 머신이 RTX 5090 이므로 PyTorch 는 CUDA 12.8 휠 인덱스(`--index-url https://download.pytorch.org/whl/cu128`)에서 설치합니다.
- 노트북 01 은 `agent`, `agent.tool_function`, `agent.tool_function.ppo_lagrangian` stub 모듈을 `sys.modules` 에 주입한 뒤 `runpy.run_path(..., run_name="__tutorial__")` 로 실제 설정 파일을 로드하여 `__main__` 학습 실행을 건너뜁니다. 설정 파일을 수정하지 말고 이 패턴을 재사용하십시오.
- 노트북 02–04 는 노트북 폴더의 `toy_reservoir.py` 를 import(`sys.path.insert(0, ".")`)하므로 작업 디렉터리를 `tutorial_nbs` 로 두고 실행해야 합니다. 출력이 저장된 상태이므로 `toy_reservoir.py` 를 바꾸면 재실행하십시오.
- 노트북 마크다운은 한국어, 코드·주석·열 이름은 영어입니다.
- `hourly_operation_nbs/` 도 같은 환경을 씁니다. 마스크는 구간 교집합 + 최근접 경계 완화(층을 버리지 않음)입니다. `groupby(...).level` 은 pandas 의 `GroupBy.level` 과 충돌하므로 `g["level"]` 을 쓰십시오.

## 설정 파일의 구조

모든 설정 파일은 약 300줄의 동일한 파일입니다. 50개 파일 전체에서 서로 다른 줄은 다음뿐입니다.

1. `filename_without_ext` (약 20번째 줄) — 원래 타임스탬프가 붙은 실험 실행 이름. 예: `A3_Lagrangian_R1_202511180027`. 한 변형의 `R1`–`R5` 사이의 **유일한** 차이가 이 줄이며, 시드는 파일에 내장되지 않고 `--seed`로 전달됩니다.
2. 약 60–96번째 줄 근처의 모듈 수준 스위치 몇 개.

각 변형은 `A3_Lagrangian`(B, C 그룹의 기준이 되는 논문의 전체 방법)에서 시작하여 스위치를 하나씩 바꾸는 체인 구조를 이룹니다.

| 변형 | 이전 변형 대비 변경 사항 |
|---|---|
| `A3_Lagrangian` | `USE_PPO_LAGRANGIAN=True`, `CONSTRAINT_TYPES=3`, 마스크 없음, 비선형 매핑 없음 |
| `A2_Penalty` | A3 대비: `USE_PPO_LAGRANGIAN=False` (고정 페널티 PPO) |
| `A1_Correction` | A3 대비: `USE_PPO_LAGRANGIAN=False`, `CONSTRAINT_TYPES=2` (보정만 하고 페널티 없음) |
| `B2_WaterMask` | A3 대비: `ENABLE_ACTION_MASK=True` (`MASK_RULE_TYPE='basic'`) |
| `B3_FullMask` | B2 대비: `MASK_RULE_TYPE='enhanced'`. **C1 (선형 매핑)** 으로도 사용됨 |
| `C2_Square` | B3 대비: `ENABLE_NONLINEAR_MAPPING=True`, `NONLINEAR_MAPPING_TYPE='square'` |
| `C3_Sqrt` / `C4_Cubic` / `C5_Exp` / `C6_Log` | C2 대비: `NONLINEAR_MAPPING_TYPE`만 변경 |

B, C 변형은 `USE_PPO_LAGRANGIAN=True`를 유지합니다. 즉, 일반 PPO가 아니라 Lagrangian 방법 위에 쌓인 구조입니다.

설정 파일을 읽거나 설명할 때 알아둘 사항:

- `CONSTRAINT_TYPES`: 1 = 위반 시 에피소드 종료; 2 = 보정하되 페널티 없음; 3 = 보정하고 페널티 부여.
- `MAX_EPISODE=60`은 **collector 환경당** 값이며, `CO_NUM=5`이므로 논문에서 보고하는 "300 에피소드"가 됩니다.
- `POLICY_TYPE`은 `USE_PPO_LAGRANGIAN`에 따라 `'ppo_lagrangian'` 또는 `'ppo'`로 결정됩니다. `policy.lagrangian` 블록은 항상 존재하지만 `enable`로 게이트됩니다.
- 적대적 학습(`ADVERSARIAL_*`), 커리큘럼, `OBS_NOISE_ENABLE` 블록은 공개된 모든 설정 파일에서 **꺼져** 있습니다. 비공개 코드베이스에 남아 있는 기능일 뿐 논문 실험이 아닙니다.
- 출력 디렉터리는 설정 파일의 상위 폴더 이름에서 `test`를 `result`로 치환하여 결정됩니다.
- 인라인 주석은 중국어입니다. 수정 시 한 파일 안에서 언어를 섞지 말고 이 관례를 유지하십시오.

## 수정 규칙

- 50개 파일이 거의 동일하므로 공통 내용을 바꿀 때는 모든 파일에 적용해야 합니다 (예: `DRL_training_config/*.py`에 `sed` 적용). 그 다음 파일 한 쌍을 `diff`하여 `filename_without_ext`와 의도한 스위치만 다른지 확인하십시오.
- 변형의 스위치가 바뀌면 `README.md`, `README_kor.md`, 그리고 위 스위치 표(`CLAUDE.md`와 `CLAUDE_kor.md` 모두)를 함께 갱신하십시오.
- 문서는 영문판과 한국어판(`README.md` ↔ `README_kor.md`, `CLAUDE.md` ↔ `CLAUDE_kor.md`, `description.md` ↔ `description_kor.md`, `tutorial_nbs/README.md` ↔ `tutorial_nbs/README_kor.md`, `tutorial_nbs/operation_guide.md` ↔ `operation_guide_kor.md`, `hourly_operation_nbs/README.md` ↔ `README_kor.md`)을 쌍으로 관리합니다. 한쪽을 수정하면 반드시 다른 쪽도 같은 내용으로 수정하십시오.
