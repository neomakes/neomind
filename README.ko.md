<p align="center">
  <img src="assets/banner.svg" alt="human-wm Banner" width="100%" />
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://jupyter.org"><img src="https://img.shields.io/badge/Jupyter-Notebook-orange.svg" alt="Jupyter"></a>
  <a href="https://ml-explore.github.io/mlx/build/html/index.html"><img src="https://img.shields.io/badge/Framework-MLX-black.svg" alt="MLX"></a>
  <a href="https://hydra.cc/"><img src="https://img.shields.io/badge/Config-Hydra-89b8cd.svg" alt="Hydra"></a>
  <a href="https://wandb.ai"><img src="https://img.shields.io/badge/Logging-W%26B-yellow.svg" alt="W&B"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href=""><img src="https://img.shields.io/badge/Status-Research%20%2F%20On%20Hold-orange.svg" alt="Status"></a>
</p>

<p align="center">
  <b>VRAE 기반 인간 행동 월드 모델</b><br>
  희소한 건강 데이터로부터 인간의 의사결정 역학을 모델링하는 이론 레이어 — <a href="https://github.com/neomakes/neopip">NeoPIP</a>의 백본.
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## human-wm이란?

**human-wm**은 NeoMakes 리서치 스택의 **이론 레이어** 입니다 — [eigen-llm](https://github.com/neomakes/eigenllm) (LLM 분해)과 [neural-field](https://github.com/neomakes/neural-field) (연속 시간 신경장)와 같은 위상에 있습니다. eigen-llm이 대형 모델을 분해하고, neural-field가 진동 기반 연산을 탐구하는 동안, human-wm은 희소한 행동 데이터로부터 **인간이 불확실성 하에서 어떻게 의사결정하는지**를 모델링합니다.

[NeoPIP](https://github.com/neomakes/neopip) (Personal Intelligence Platform)의 **ML 백본** 역할을 하며, 개인화된 웰니스 인텔리전스를 구동하는 생성 모델을 제공합니다.

---

## 목차

- [배경 및 동기](#배경-및-동기)
- [주요 기능](#주요-기능)
- [아키텍처](#아키텍처)
- [설치](#설치)
- [사용법](#사용법)
- [설정](#설정)
- [프로젝트 구조](#프로젝트-구조)
- [이론 레이어 생태계](#이론-레이어-생태계)
- [현재 상태](#현재-상태)
- [로드맵](#로드맵)
- [기여하기](#기여하기)
- [라이선스](#라이선스)

---

## 배경 및 동기

건강 및 웰니스 데이터는 본질적으로 **희소(sparse)** 합니다 — 사용자가 모든 식사, 운동, 기분 변화를 기록하지는 않기 때문입니다. 기존 모델은 이러한 불규칙성에 취약합니다. human-wm은 **변분 순환 오토인코더(VRAE)** 를 통해 불완전한 데이터로부터 의사결정 역학을 학습합니다.

핵심 통찰: 인간의 행동은 세 가지 잠재 요인으로 분해할 수 있습니다:
- **초기 상태 다양성** (z_a) — 개인의 기본 특성
- **행동 스타일** (z_b) — 활동적 vs. 비활동적 경향
- **생리적 반응** (z_c) — 행동에 대한 신체 반응 방식

이 세 요인의 조합을 샘플링하면 (5 x 5 x 5 = **125가지 다양한 궤적**), 동일한 초기 조건에서 그럴듯한 행동 미래의 스펙트럼을 생성할 수 있습니다.

BT 기반 다중 로봇 제어 연구 — 불확실성 하의 에이전트 의사결정 모델링 — 에서 영감을 받았습니다.

---

## 주요 기능

- **3-잠재변수 VRAE** — 세 개의 독립적인 잠재 변수(z_a: 16D, z_b: 32D, z_c: 32D)가 서로 다른 행동 차원을 포착
- **마스킹 기반 손실** — 유효한 타임스텝에서만 손실을 계산하여 희소/결측 데이터를 처리 (학습 데이터 중 63% 유효, 37% 결측)
- **정책 네트워크** — `pi(action | state, context; z_b)` 학습: 주어진 상태에서 사용자의 행동 예측
- **전이 네트워크** — `tau(next_state | state, action, context; z_c)` 학습: 행동 후 상태 변화 예측
- **자기회귀 롤아웃** — 추론 시 정책 + 전이 네트워크를 연쇄하여 전체 궤적 생성
- **4가지 거리 지표** — RMSE, MAE, MAPE, Huber (기본값) — 설정으로 선택 가능
- **Hydra 설정** — 모든 하이퍼파라미터를 YAML + CLI 오버라이드로 제어
- **W&B 연동** — 손실 곡선, 하이퍼파라미터 로깅을 통한 실험 추적

---

## 아키텍처

```
입력: [actions(7D), states(2D), context(1D), mask(1D)] x T 타임스텝

Step 1: 인코더 (BiGRU + 마스크 어텐션 풀링)
  → mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c

Step 2: 샘플링 (재매개변수화 트릭)
  z_a ~ N(mu_a, sigma_a)  [K=5 샘플]  — 초기 상태 다양성
  z_b ~ N(mu_b, sigma_b)  [K=5 샘플]  — 행동 스타일
  z_c ~ N(mu_c, sigma_c)  [K=5 샘플]  — 생리적 반응
  → 5 x 5 x 5 = 125 조합

Step 3: 디코더 (BiGRU)
  [z_a, z_b, z_c, context] → 복원된 actions + states

Step 4: 정책 네트워크 (MLP)
  pi(a_t | s_t, w_t; z_b) → 예측된 행동

Step 5: 전이 네트워크 (MLP)
  tau(s_{t+1} | s_t, a_t, w_t; z_c) → 예측된 다음 상태

Step 6: 롤아웃 (추론 전용)
  정책 + 전이를 자기회귀적으로 연쇄 → 125개 미래 궤적
```

### 손실 함수

```
L_total = w_vae * L_VAE + w_action * L_action + w_transition * L_transition + w_rollout * L_rollout

구성:
  L_VAE       = L_reconstruction + beta * L_KL  (KL 어닐링: 0 → 1)
  L_action    = masked distance(predicted_action, true_action)
  L_transition = masked distance(predicted_state, true_state)
  L_rollout   = 125개 생성 궤적의 평균 거리
```

### 데이터 스키마 (타임스텝당 10개 특성)

| 카테고리 | 특성 | 차원 |
|:--|:--|:--|
| 행동 | sleep_hours, workout_type, location, steps, calories, distance, active_minutes | 7D |
| 상태 | heart_rate_avg, mood | 2D |
| 컨텍스트 | weather_conditions | 1D |
| 마스크 | 유효/결측 지시자 | 1D |

---

## 설치

### 사전 요구사항

- Python 3.10+
- Apple Silicon Mac 권장 (MLX는 Apple GPU에 최적화)

### 설치 방법

```bash
git clone https://github.com/neomakes/human-wm.git
cd human-wm
pip install mlx hydra-core wandb numpy tqdm pandas
```

---

## 사용법

### 빠른 테스트 (1 에포크)

```bash
python scripts/train.py training.epochs=1 training.batch_size=32
```

### 전체 학습

```bash
python scripts/train.py \
  training.epochs=100 \
  training.batch_size=32 \
  training.learning_rate=0.001 \
  model.hidden_dim=256
```

### W&B 추적과 함께

```bash
wandb login
python scripts/train.py \
  training.use_wandb=true \
  wandb.project="human-wm-vrae" \
  training.epochs=100
```

### 순차 실험

여러 설정을 조기 종료와 함께 자동으로 실행:

```bash
python scripts/run_experiments.py

# 또는 백그라운드 실행
nohup python scripts/run_experiments.py > logs/experiments.log 2>&1 &
```

### 추론 및 시각화

`analysis.ipynb`에서:
- 특정 사용자의 잠재 변수 분포 추출
- t-SNE를 통한 사용자 클러스터 시각화
- 125개 궤적 시나리오 생성 및 비교

---

## 설정

모든 하이퍼파라미터는 Hydra로 관리됩니다. CLI에서 원하는 설정을 오버라이드할 수 있습니다.

### 모델 (`conf/model/vrae.yaml`)

| 파라미터 | 기본값 | 설명 |
|:--|:--|:--|
| `hidden_dim` | 64 | RNN 은닉 차원 |
| `num_layers` | 2 | RNN 레이어 수 |
| `latent_action_dim` | 16 | z_a 차원 |
| `latent_behavior_dim` | 32 | z_b 차원 |
| `latent_context_dim` | 32 | z_c 차원 |
| `k_a`, `k_b`, `k_c` | 5 | 잠재 변수별 샘플 수 |
| `distance_type` | huber | 거리 지표 (rmse/mae/mape/huber) |

### 학습 (`conf/training/default.yaml`)

| 파라미터 | 기본값 | 설명 |
|:--|:--|:--|
| `learning_rate` | 0.001 | Adam 학습률 |
| `batch_size` | 32 | 배치 크기 (사용자 단위) |
| `epochs` | 100 | 학습 에포크 |
| `kl_annealing_end` | 20 | KL 가중치가 1.0에 도달하는 에포크 |
| `w_vae` | 1.0 | VAE 손실 가중치 |
| `w_action` | 0.5 | 정책 손실 가중치 |
| `w_transition` | 0.5 | 전이 손실 가중치 |
| `w_rollout` | 0.3 | 롤아웃 손실 가중치 |

```bash
# 예시: 대형 모델 + 느린 KL 어닐링
python scripts/train.py \
  model.hidden_dim=512 \
  model.latent_behavior_dim=64 \
  training.kl_annealing_end=50 \
  training.learning_rate=0.0005
```

---

## 프로젝트 구조

```
human-wm/
├── conf/                        # Hydra 설정
│   ├── config.yaml              # 메인 설정 (데이터 경로, W&B)
│   ├── model/
│   │   └── vrae.yaml            # 모델 하이퍼파라미터
│   └── training/
│       └── default.yaml         # 학습 하이퍼파라미터
├── models/
│   └── vrae.py                  # VRAE, PolicyNetwork, TransitionNetwork
├── scripts/
│   ├── train.py                 # KL 어닐링 포함 학습 루프
│   ├── run_experiments.py       # 순차 실험 실행기
│   └── quick_test.py            # 빠른 검증 실행
├── data/
│   └── fitness_tracker_data.npz # 학습 데이터 (999명 사용자, 1000 타임스텝)
├── docs/                        # 설계 문서 및 전처리 노트북
├── analysis.ipynb               # 추론, 시각화, t-SNE 분석
├── logs/                        # 체크포인트 및 실험 결과
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── README.md
```

---

## 이론 레이어 생태계

human-wm은 NeoMakes 리서치 스택의 세 가지 **이론 레이어** 중 하나입니다:

| 레이어 | 저장소 | 초점 |
|:--|:--|:--|
| **human-wm** | 이 저장소 | 인간 행동 월드 모델 — VRAE 기반 의사결정 역학 |
| **eigen-llm** | [neomakes/eigenllm](https://github.com/neomakes/eigenllm) | LLM 분해 — 대형 범용 AI → 소형 특화 AI |
| **neural-field** | [neomakes/neural-field](https://github.com/neomakes/neural-field) | 연속 시간 신경장 — Kuramoto + Free Energy |

### 관련 응용 레이어

- **[NeoPIP](https://github.com/neomakes/neopip)** — Personal Intelligence Platform. human-wm은 NeoPIP의 웰니스 인텔리전스 기능을 위한 ML 백본 역할을 합니다.
- **[NeoSense](https://github.com/neomakes/neosense)** — 멀티모달 센서 로깅. 행동 모델링에 필요한 원시 물리 데이터 패턴을 제공합니다.

---

## 현재 상태

**Research / On Hold** — 모델 아키텍처는 100% 완성. 학습 수렴에 대한 추가 조사가 필요합니다.

### 동작하는 것

- 모델 아키텍처: 모든 구성요소 구현 완료 (VRAE 인코더/디코더, 정책 네트워크, 전이 네트워크)
- 4가지 손실 함수 완전 통합 (VAE + 정책 + 전이 + 롤아웃)
- NaN 문제 해결: 로그-분산 매개변수화 + 클리핑으로 학습 안정화
- 학습이 에러 없이 실행되며, 에포크에 걸쳐 손실이 감소

### 알려진 문제

- **학습 수렴**: 모델은 수렴하지만 생성된 궤적의 다양성이 기대에 미치지 못함
- **추정 원인**: z_b/z_c 사후 붕괴(posterior collapse) — KL 항이 모델로 하여금 이 잠재 변수들을 무시하게 할 가능성
- **결정**: 다른 우선순위에 집중하기 위해 일시 중단; 아키텍처 기반은 견고함

---

## 로드맵

- [ ] KL 어닐링 스케줄 조정을 통한 사후 붕괴 조사
- [ ] beta-VAE 접근법 시도 (잠재 변수별 개별 beta 가중치)
- [ ] 궤적 다양성 지표 추가 (커버리지, 궤적 간 거리)
- [ ] 정량적 성능 평가 스크립트
- [ ] 실제 사용자 데이터 연동 (NeoSense 파이프라인)

---

## 기여하기

가이드라인은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

이 프로젝트는 [Code of Conduct](CODE_OF_CONDUCT.md)를 따릅니다.

---

## 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다 — 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
