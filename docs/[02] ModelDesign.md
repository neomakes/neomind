# VRAE (Variational Recurrent Autoencoder) 모델 설계 문서

## 1. 개념 정의

### 1.1 상태, 행동, 컨텍스트 분류

**마스크 (Missing Data Indicator)** [필수]:
```
m_t ∈ {0, 1}  (시점 t에서의 데이터 유효성)
  0: 실제 관측 데이터 (활동 있음)
  1: 결측값 (활동 없음, 63.42% 유효 vs 36.58% 결측)
  
역할: 결측값 처리를 위한 필수 지표 (loss 계산 시 유효 시점만 사용)
```

**상태 (State - 사용자가 제어 불가능한 생리적 상태)**:
```
s_t ∈ ℝ^2  (시점 t에서의 내생적(endogenous) 상태)
  [0] heart_rate_avg     ∈ [0, 179]      (심박수: 신체 상태)
  [1] mood               ∈ {0, 1, ..., 4} (기분: 심리 상태, 5 클래스)
  
특징:
  - 사용자가 직접 제어 불가능
  - 행동의 결과로서 변화 (운동 후 심박수 증가 등)
  - 다음 행동의 원인이 됨 (기분이 안 좋으면 운동 안 함)
```

**행동 (Action - 사용자가 의도적으로 제어 가능)**:
```
a_t ∈ ℝ^7  (시점 t에서의 사용자 행동/선택)
  [0] sleep_hours        ∈ [0, 12]       (수면: 의도적 선택)
  [1] workout_type       ∈ {0, 1, ..., 7} (운동 유형: 의도적 선택, 8 클래스)
  [2] location           ∈ {0, 1, ..., 5} (위치: 의도적 선택, 6 클래스)
  [3] steps              ∈ [0, 29999]    (걸음 수: 활동량)
  [4] calories_burned    ∈ [0, 4000]     (칼로리 소모: 활동 결과)
  [5] distance_km        ∈ [0, 20]       (이동 거리: 활동 결과)
  [6] active_minutes     ∈ [0, 1439]     (활동 시간: 활동량)
  
특징:
  - 사용자가 의도적으로 선택/제어
  - 행동 = 현재 상태의 결과 → 다음 상태의 원인 (마르코프 성질)
  - 정책 π(a_t | s_t, w_t)로 모델링
```

**컨텍스트 (Context - 외부 환경, 사용자가 제어 불가)**:
```
w_t ∈ {0, 1, ..., 4}  (시점 t에서의 외부 환경 조건)
  [0] weather_conditions ∈ {0, 1, ..., 4} (날씨: 외부 환경, 5 클래스)
  
특징:
  - 사용자가 제어 불가능한 외부 환경
  - 상태와 행동에 영향을 미치는 조건 (비오는 날씨 → 실내 운동 선택)
  - 주어진 조건 (과거 관측, 미래는 모름)
  - CVAE에서 조건부 입력으로 작용
```

**요약**:
```
데이터 흐름:
  (상태 s_t, 컨텍스트 w_t) 
    ↓ (정책)
  행동 a_t 선택
    ↓ (천이)
  다음 상태 s_{t+1} 결정 (+ 컨텍스트 w_{t+1})
```

### 1.2 모델의 목표

**학습 목표**: 사용자 활동의 의사결정 역학(decision dynamics) 학습
```
관찰된 시계열: 
  - 상태: s_0, s_1, ..., s_T (T=1000)
  - 행동: a_0, a_1, ..., a_T
  - 컨텍스트: w_0, w_1, ..., w_T
  - 마스크: m_0, m_1, ..., m_T (유효성 지시자)

목표: 
  1) 행동의 확률 분포 p(a_t | s_t, w_t) 학습 [정책]
  2) 상태 천이 분포 p(s_{t+1} | s_t, a_t, w_t) 학습 [천이]
  3) 시계열 데이터의 확률 분포 p(s_{0:T}, a_{0:T} | w_{0:T}, m_{0:T}) 학습 [VAE]
```

**추론 목표**: 다양한 시나리오 궤적 생성
```
입력: 초기 상태 s_0, 컨텍스트 w_{0:T}, 마스크 m_{0:T}
프로세스:
  1) z_a 샘플 (K_a=5): 초기 상태의 다양성
  2) z_b 샘플 (K_b=5): 정책 스타일의 다양성 (행동 선택 경향)
  3) z_c 샘플 (K_c=5): 천이 패턴의 다양성 (상태 변화 경향)
  4) 125개 (z_a, z_b, z_c) 조합 → 125개의 가능한 궤적 생성

출력: K_a × K_b × K_c = 125개의 다양한 궤적
  τ^{(k_a, k_b, k_c)} = [a_0, a_1, ..., a_T]  (각 125개 행동 시계열)
  + 대응하는 상태 변화 s_{0:T}
```

---

## 2. 모델 아키텍처

### 2.1 전체 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                     VRAE MODEL OVERVIEW                     │
└─────────────────────────────────────────────────────────────┘

데이터 구성 (10 features):
  s_t: 상태 (2D) - [heart_rate_avg, mood]
  a_t: 행동 (7D) - [sleep_hours, workout_type, location, steps, calories_burned, distance_km, active_minutes]
  w_t: 컨텍스트 (1D) - [weather_conditions]
  m_t: 마스크 (1D) - [유효성]

Input: [a_{0:T}, s_{0:T}, w_{0:T}, m_{0:T}] → (1000, 11)
       └─ a: (1000, 7)   action features
       └─ s: (1000, 2)   state features
       └─ w: (1000, 1)   context/covariates
       └─ m: (1000, 1)   mask

┌──────────────────────────────┐
│   1. VAE (조건부)              │
│   Input: [a_{0:T}, s_{0:T},  │
│          w_{0:T}, m_{0:T}]   │
└──────────────────────────────┘
         ↓
   ┌─────────────────────┐
   │ Encoder q(z|a,s,w)  │  → μ_a, σ_a (상태 다양성)
   ├─────────────────────┤     μ_b, σ_b (정책 스타일)
   │ Latent: z_a, z_b    │     μ_c, σ_c (천이 패턴)
   │         z_c         │
   └─────────────────────┘
         ↓ (샘플링)
   ┌─────────────────────┐
   │ k_a × k_b × k_c     │  125개 샘플 조합
   │ 샘플 생성             │
   └─────────────────────┘
         ↓
   ┌─────────────────────┐
   │ Decoder p(â|z,w)    │  → 재구성된 a_{0:T}
   └─────────────────────┘

┌──────────────────────────────────────────────────────┐
│   2. Policy Network (정책 네트워크)                     │
│   Input: [s_t, w_t; z_b]                             │
│   Output: predicted action a_t                       │
│   역할: 상태와 컨텍스트가 주어졌을 때 사용자 행동 예측           │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│   3. Transition Network (천이 네트워크)                 │
│   Input: [s_t, a_t, w_t; z_c]                        │
│   Output: predicted next state ŝ_{t+1}               │
│   역할: 상태와 행동에서 다음 상태(심박, 기분) 예측             │
└──────────────────────────────────────────────────────┘

Output: 125개의 롤아웃 행동 궤적 + 재구성 손실
```

---

### 2.2 상세 컴포넌트 설명

#### **A. VAE (Variational Autoencoder) - 조건부**

**목적**: 행동과 상태의 결합 확률 분포 학습 + 다양한 잠재 변수 샘플링

**입력 처리**:
```
Concatenated Input:
  X = [a_0, a_1, ..., a_T, s_0, s_1, ..., s_T, w_0, w_1, ..., w_T, m_0, m_1, ..., m_T]
  Shape: (T, 7+2+1+1) = (1000, 11)
  
마스킹 적용 (선택사항):
  Masked X = X * (1 - m)  # 결측값인 a_t, s_t를 0으로 마스킹
  
목적:
  - 행동 a의 분포를 학습 (정책)
  - 상태 s의 분포를 학습 (천이)
  - 외부 조건 w의 영향을 조건으로 반영
```

**Encoder 구조**:
```
Input: X ∈ ℝ^(T×11)  (마스킹된 입력: 행동+상태+컨텍스트+마스크)
  ↓
Embedding Layer (범주형 특성):
  a_emb = Embedding([workout_type, location])  ∈ ℝ^(T×16)
  w_emb = Embedding([weather_conditions])      ∈ ℝ^(T×4)
  
Concatenate (모든 정보):
  [a_continuous, a_emb, s, w_emb, m]  ∈ ℝ^(T×32)
  
BiLSTM Encoder (양방향):
  h_enc = BiLSTM(concat_input)  ∈ ℝ^(T×256)
  
Global Pooling (마스크 기반):
  h_agg = MaskedAttentionPool(h_enc, m)  ∈ ℝ^256
  
Output Layer (3개 잠재 변수 - 각각 마르코프 성질):
  μ_a, σ_a = FC(h_agg)  ∈ ℝ^(latent_dim_a × 2)  [z_a: 초기 상태 다양성]
  μ_b, σ_b = FC(h_agg)  ∈ ℝ^(latent_dim_b × 2)  [z_b: 정책 스타일]
  μ_c, σ_c = FC(h_agg)  ∈ ℝ^(latent_dim_c × 2)  [z_c: 천이 패턴]

권장 잠재 차원:
  latent_dim_a = 16  (초기 상태: 작음, 심박/기분 다양성)
  latent_dim_b = 32  (정책: 중간, 행동 선택 경향)
  latent_dim_c = 32  (천이: 중간, 상태 변화 패턴)
```

**샘플링 (Reparameterization Trick)**:
```
For k_a ∈ [0, K_a-1]:
  ε_a ~ N(0, I)
  z_a^{(k_a)} = μ_a + σ_a ⊙ ε_a  ∈ ℝ^16  (초기 상태 다양성)

For k_b ∈ [0, K_b-1]:
  ε_b ~ N(0, I)
  z_b^{(k_b)} = μ_b + σ_b ⊙ ε_b  ∈ ℝ^32  (정책 스타일)

For k_c ∈ [0, K_c-1]:
  ε_c ~ N(0, I)
  z_c^{(k_c)} = μ_c + σ_c ⊙ ε_c  ∈ ℝ^32  (천이 패턴)

조합 생성:
  For each (k_a, k_b, k_c):
    z^{(k_a, k_b, k_c)} = [z_a^{(k_a)}, z_b^{(k_b)}, z_c^{(k_c)}]
    → 125개의 (80,)-shaped 벡터
```

**Decoder 구조** (조건부):
```
Input: z^{(k_a, k_b, k_c)} ∈ ℝ^80, w_0:T ∈ ℝ^(T×1)

Expansion:
  z_expanded = Repeat(z, T)  ∈ ℝ^(T×80)
  
Concatenate with context:
  decoder_input = [z_expanded, w]  ∈ ℝ^(T×81)
  
BiLSTM Decoder:
  h_dec = BiLSTM(decoder_input)  ∈ ℝ^(T×256)
  
Output Layer (행동+상태 재구성):
  â = FC_action(h_dec)  ∈ ℝ^(T×7)   (행동 재구성)
  ŝ = FC_state(h_dec)   ∈ ℝ^(T×2)   (상태 재구성)
  
주의: 
  - w는 조건으로 주어짐 (예측 불필요)
  - 마스크 기반 손실 계산 (유효 시점만)
```

---

#### **B. Policy Network (정책 네트워크)**

**목적**: 주어진 상태와 컨텍스트 그리고 정책 스타일에서 행동 예측

**구조**:
```
Input:
  s_t ∈ ℝ^2         (현재 상태: 심박수, 기분)
  w_t ∈ ℝ^1         (현재 컨텍스트: 날씨)
  z_b ∈ ℝ^32        (정책 스타일 벡터 - 행동 선택 경향)
  
Embedding:
  w_emb = Embedding([weather_conditions])  ∈ ℝ^4
  
MLP (3 hidden layers, 정책 조건화):
  concat = [s_t, w_emb, z_b]  ∈ ℝ^38
  
  h1 = ReLU(FC(concat, 128))  ∈ ℝ^128
  h2 = ReLU(FC(h1, 128))      ∈ ℝ^128
  h3 = ReLU(FC(h2, 64))       ∈ ℝ^64
  
Output (행동 예측):
  â_continuous = FC(h3, 5)   ∈ ℝ^5  (sleep_hours, steps, calories_burned, distance_km, active_minutes)
  â_categorical = FC(h3, 2)  ∈ ℝ^2  (workout_type, location - softmax 적용)
  
  â_t = [â_continuous, â_categorical]  ∈ ℝ^7  (전체 행동)
```

**해석**:
```
정책 π(a_t | s_t, w_t; z_b):
  - z_b가 다르면 같은 상태/날씨 조건에서도 다른 행동을 예측
  - 예: z_b^(1)은 "활동적인 스타일" → 많은 운동
       z_b^(2)는 "휴식 스타일" → 적은 운동
  
데이터 흐름:
  상태 s_t (심박: 낮음) + 컨텍스트 w_t (맑은 날씨) + 정책 z_b
    → 행동 a_t 예측 (수면시간, 운동 유형, 위치, 칼로리 등)
```

---

#### **C. Transition Network (천이 네트워크)**

**목적**: 상태, 행동, 컨텍스트에서 다음 상태(심박, 기분) 예측

**구조**:
```
Input:
  s_t ∈ ℝ^2         (현재 상태: 심박수, 기분)
  a_t ∈ ℝ^7         (현재 행동: 수면, 운동, 위치, 활동량 등)
  w_t ∈ ℝ^1         (현재 컨텍스트: 날씨)
  z_c ∈ ℝ^32        (천이 패턴 벡터 - 상태 변화 경향)
  
Embedding:
  a_emb = Embedding([workout_type, location])  ∈ ℝ^16
  w_emb = Embedding([weather_conditions])      ∈ ℝ^4
  
MLP (3 hidden layers, 천이 조건화):
  concat = [s_t, a_continuous, a_emb, w_emb, z_c]  ∈ ℝ^63
  
  h1 = ReLU(FC(concat, 128))  ∈ ℝ^128
  h2 = ReLU(FC(h1, 128))      ∈ ℝ^128
  h3 = ReLU(FC(h2, 64))       ∈ ℝ^64
  
Output (다음 상태 예측):
  ŝ_{t+1} = FC(h3, 2)  ∈ ℝ^2  (다음 상태: 심박수, 기분)
```

**해석**:
```
천이 τ(s_{t+1} | s_t, a_t, w_t; z_c):
  - z_c가 다르면 같은 (상태, 행동) 조합에서도 다른 다음 상태 예측
  - 예: z_c^(1)은 "회복 패턴" → 운동 후 심박수 빠르게 회복
       z_c^(2)는 "유지 패턴" → 운동 후 심박수 천천히 회복

데이터 흐름:
  현재 상태 s_t (심박: 60) + 행동 a_t (30분 운동) + 날씨 (맑음) + 천이 z_c
    → 다음 상태 s_{t+1} 예측 (심박: 75, 기분: 좋음)
```

---

### 2.3 롤아웃 (Rollout) 계산

**목적**: 125개의 (z_a, z_b, z_c) 조합 각각에 대해 전체 행동 궤적 생성

**알고리즘**:
```
For each combination (k_a, k_b, k_c) ∈ [0, K_a) × [0, K_b) × [0, K_c):
  
  # 초기 상태 설정 (z_a에서 결정)
  s_0^{(k_a,k_b,k_c)} = z_a^{(k_a)}에서 유도된 초기 상태 (심박, 기분)
  
  # 순차적 롤아웃
  For t = 0 to T-1:
    # 정책에서 행동 생성 (정책 스타일 z_b 조건)
    a_t^{(k_a,k_b,k_c)} = Policy(s_t^{(k_a,k_b,k_c)}, w_t; z_b^{(k_b)})
    
    # 천이 네트워크에서 다음 상태 생성 (천이 패턴 z_c 조건)
    s_{t+1}^{(k_a,k_b,k_c)} = Transition(s_t^{(k_a,k_b,k_c)}, a_t^{(k_a,k_b,k_c)}, w_t; z_c^{(k_c)})
    
  # 최종 궤적 (행동 시계열)
  τ^{(k_a,k_b,k_c)} = [a_0, a_1, ..., a_T]  ∈ ℝ^(T×7)
  + 상응하는 상태 변화: [s_0, s_1, ..., s_T]  ∈ ℝ^(T×2)

생성된 행동 궤적들: {τ^{(k_a,k_b,k_c)} : (k_a,k_b,k_c) ∈ [0,125)}
```

**해석**:
```
- z_a 샘플: 5가지 "초기 기분/심박" 다양성
- z_b 샘플: 5가지 "행동 스타일" (활동적 ↔ 소극적)
- z_c 샘플: 5가지 "신체 반응 패턴" (민감 ↔ 무디)

결과: 125개의 가능한 "행동 시나리오"
  - 같은 조건 (초기 상태, 날씨)에서
  - 사용자의 다양한 의사결정 방식 반영
  - 각 시나리오별 예상되는 행동 궤적 시뮬레이션
```

---

## 3. 손실함수 (Loss Functions)

### 3.1 VAE 손실 (L_VAE)

**KL Divergence (쿨백-라이블러 발산)**:
```
L_KL = -0.5 * Σ(1 + log(σ^2) - μ^2 - σ^2)  [각 잠재 변수별]

L_KL_total = λ_a * L_KL(z_a) + λ_b * L_KL(z_b) + λ_c * L_KL(z_c)

권장: λ_a = 1.0, λ_b = 1.0, λ_c = 1.0 (학습 중 조정 가능)
```

**Reconstruction Loss (재구성 손실)**:
```
마스킹된 재구성 손실 (유효한 데이터만 계산):

L_Recon = L_Recon_action + L_Recon_state

1) 행동 재구성 손실:
   L_Recon_action = Σ_{t: m_t=0} distance_metric(â_t, a_t)
   
2) 상태 재구성 손실:
   L_Recon_state = Σ_{t: m_t=0} distance_metric(ŝ_t, s_t)

distance_metric ∈ {RMSE, MAE, MAPE, Huber}

각 메트릭 정의:
  1) RMSE (Root Mean Square Error):
     RMSE_t = √(MSE(ŷ_t, y_t))
     
  2) MAE (Mean Absolute Error):
     MAE_t = |ŷ_t - y_t|
     
  3) MAPE (Mean Absolute Percentage Error):
     MAPE_t = |ŷ_t - y_t| / (|y_t| + ε)  [ε = 1e-8 for numerical stability]
     
  4) Huber Loss:
     Huber_t(δ) = { 0.5 * (ŷ_t - y_t)^2        if |ŷ_t - y_t| ≤ δ
                  { δ * (|ŷ_t - y_t| - 0.5*δ)  otherwise
     권장: δ = 1.0
```

**VAE 총 손실**:
```
L_VAE = w_recon * L_Recon + w_kl * L_KL_total

권장:
  w_recon = 1.0
  w_kl = β (학습 중 증가시킬 수 있음, KL annealing)
  초기: β = 0.1, 점진적 증가 → 1.0
```

---

### 3.2 Action Prediction Loss (L_Action)

**정책 네트워크의 행동 예측 손실**:
```
행동 예측 손실 (마스크 기반):

L_Action = Σ_{t: m_t=0} distance_metric(a_t_pred, a_t_true)

여기서:
  a_t_pred = Policy(s_t, w_t; z_b)
  a_t_true = 실제 관측된 행동
  
주의:
  - 정책은 "현재 상태와 조건에서 사용자가 취할 행동"을 예측
  - 따라서 loss는 예측 행동과 실제 행동 비교
  - 마스크를 통해 유효한 시점만 학습
```

---

### 3.3 Rollout Loss (L_Rollout)

**125개 행동 궤적과 원본 궤적의 평균 거리**:
```
각 롤아웃 행동 궤적에 대해:

For each (k_a, k_b, k_c):
  τ^{(k_a,k_b,k_c)} = [a_0^{(k_a,k_b,k_c)}, ..., a_T^{(k_a,k_b,k_c)}]  (125개 행동 시계열)
  τ_true = [a_0_true, a_1_true, ..., a_T_true]  (1개 실제 행동 시계열)

롤아웃 손실:
  L_Rollout_single = (1/K_a*K_b*K_c) * Σ_{k_a,k_b,k_c} 
                       Σ_{t: m_t=0} distance_metric(τ_t^{(k_a,k_b,k_c)}, τ_t_true)

마스킹:
  - 결측 시점(m_t=1)은 제외
  - 유효 시점(m_t=0)만 누적
  
의미:
  - 125개의 생성 행동이 실제 행동과 얼마나 유사한지 측정
  - 정책과 천이 네트워크의 예측력을 평가
```

**거리 메트릭 선택**:
```
파라미터: distance_type ∈ {'rmse', 'mae', 'mape', 'huber'}

추천 조합:
  - Primary: 'huber' (이상치와 노이즈에 균형잡힘)
  - Backup: 'rmse' (큰 오차에 민감)
  - 비교: 'mae' (robust하고 해석 용이)
```

---

### 3.4 전체 손실함수 (Total Loss)

```
L_Total = w_vae * L_VAE 
        + w_action * L_Action
        + w_rollout * L_Rollout

권장 가중치:
  w_vae = 1.0       (기본)
  w_action = 0.5    (보조)
  w_rollout = 0.3   (정규화)

학습 전략:
  Phase 1 (Epoch 0-50):
    w_vae = 1.0, w_action = 0.1, w_rollout = 0.0  [VAE 사전학습]
  
  Phase 2 (Epoch 51-200):
    w_vae = 1.0, w_action = 0.5, w_rollout = 0.3  [전체 학습]
```

---

## 4. 구현 세부사항

### 4.1 모델 클래스 구조

```python
# models/vrae.py

class VRAE(nn.Module):
    """
    Variational Recurrent Autoencoder for Fitness Tracker Data
    """
    
    def __init__(
        self,
        # 데이터 차원
        state_dim: int = 6,           # s_t 차원
        context_dim: int = 4,         # w_t 차원
        embedding_dim: int = 8,       # 범주형 임베딩
        
        # 잠재 차원
        latent_dim_a: int = 16,       # z_a (초기 상태)
        latent_dim_b: int = 32,       # z_b (정책)
        latent_dim_c: int = 32,       # z_c (천이)
        
        # RNN 차원
        rnn_hidden_dim: int = 256,    # LSTM/BiLSTM 숨김 차원
        
        # MLP 차원
        mlp_hidden_dim: int = 128,    # 정책/천이 네트워크
        
        # 거리 메트릭
        distance_type: str = 'huber',
        huber_delta: float = 1.0,
        
        # 손실 가중치
        kld_weight: float = 1.0,
        action_weight: float = 0.5,
        rollout_weight: float = 0.3,
        
        # 샘플 개수
        k_a: int = 5,
        k_b: int = 5,
        k_c: int = 5,
    ):
        pass
    
    def encode(self, s, w, m):
        """
        인코더: [s, w, m] → (μ_a, σ_a, μ_b, σ_b, μ_c, σ_c)
        """
        pass
    
    def sample_latents(self, mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c):
        """
        샘플링: 각각 k_a, k_b, k_c개 샘플 생성
        Returns: z_a[k_a, d_a], z_b[k_b, d_b], z_c[k_c, d_c]
        """
        pass
    
    def decode(self, z_a, z_b, z_c, w):
        """
        디코더: z와 w → 재구성된 s
        Returns: ŝ[B, T, 6]
        """
        pass
    
    def policy_network(self, s, w, z_b):
        """
        정책: (s, w, z_b) → a
        Returns: a[B, T, 6]
        """
        pass
    
    def transition_network(self, s, a, w, z_c):
        """
        천이: (s, a, w, z_c) → ŝ_{t+1}
        Returns: s_next[B, T, 6]
        """
        pass
    
    def rollout(self, s0, w, z_a_samples, z_b_samples, z_c_samples):
        """
        롤아웃: 초기상태와 z에서 k_a*k_b*k_c 궤적 생성
        Returns: trajectories[k_a*k_b*k_c, T, 6]
        """
        pass
    
    def loss_function(self, model_output, s_true, w, m, kld_weight):
        """
        전체 손실 계산
        Returns: total_loss, recon_loss, kld_loss, action_loss, rollout_loss
        """
        pass
    
    def forward(self, x, w, m):
        """
        전체 forward pass
        """
        pass
```

---

### 4.2 거리 메트릭 구현

```python
def distance_metric(y_pred, y_true, metric_type='huber', huber_delta=1.0):
    """
    다양한 시계열 거리 메트릭
    
    Args:
        y_pred: 예측값 [B, T, D]
        y_true: 실제값 [B, T, D]
        metric_type: {'rmse', 'mae', 'mape', 'huber'}
        huber_delta: Huber loss의 delta 값
    
    Returns:
        손실값 (scalar)
    """
    
    diff = y_pred - y_true
    
    if metric_type == 'rmse':
        return mx.sqrt(mx.mean(diff ** 2))
    
    elif metric_type == 'mae':
        return mx.mean(mx.abs(diff))
    
    elif metric_type == 'mape':
        epsilon = 1e-8
        return mx.mean(mx.abs(diff) / (mx.abs(y_true) + epsilon))
    
    elif metric_type == 'huber':
        huber_loss = mx.where(
            mx.abs(diff) <= huber_delta,
            0.5 * diff ** 2,
            huber_delta * (mx.abs(diff) - 0.5 * huber_delta)
        )
        return mx.mean(huber_loss)
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
```

---

### 4.3 데이터 입력 형태

```
Batch Input:
  batch_x: shape (B, T, 11)  [action + state + context + mask]
    - batch_x[:, :, 0:7]    → action (a_t): [sleep_hours, workout_type, location, steps, calories_burned, distance_km, active_minutes]
    - batch_x[:, :, 7:9]    → state (s_t): [heart_rate_avg, mood]
    - batch_x[:, :, 9:10]   → context (w_t): [weather_conditions]
    - batch_x[:, :, 10]     → mask (m_t): [유효성]
  
  batch_a: shape (B, T, 7)  [행동 타겟]
    - 정책 네트워크 학습용 타겟
  
  batch_s: shape (B, T, 2)  [상태 타겟]
    - 천이 네트워크 + VAE 디코더 학습용 타겟
  
  batch_mask: shape (B, T, 1)  [마스킹]
    - 유효한 시점만 loss 계산

Batch Size B: 50 권장
Time Steps T: 1000 (고정)
Features: 11 (7 action + 2 state + 1 context + 1 mask)
Total Forward Passes: 50 × 125 = 6,250 (한 배치당)
```

---

## 5. 훈련 파라미터 요약

### 5.1 모델 하이퍼파라미터 (conf/model/vrae.yaml)

```yaml
model:
  name: vrae
  
  # 데이터 차원
  action_dim: 7                # a_t 차원: sleep_hours, workout_type, location, steps, calories_burned, distance_km, active_minutes
  state_dim: 2                 # s_t 차원: heart_rate_avg, mood
  context_dim: 1               # w_t 차원: weather_conditions
  embedding_dim: 8             # 범주형 임베딩 차원
  
  # 잠재 차원
  latent_dim_a: 16             # z_a (상태 다양성)
  latent_dim_b: 32             # z_b (정책 스타일)
  latent_dim_c: 32             # z_c (천이 패턴)
  
  # RNN/MLP 차원
  rnn_hidden_dim: 256          # LSTM/BiLSTM 숨김 차원
  mlp_hidden_dim: 128          # 정책/천이 네트워크
  
  # 거리 메트릭
  distance_type: huber         # or rmse, mae, mape
  huber_delta: 1.0
  
  # 손실 가중치
  kld_weight: 1.0
  action_weight: 0.5
  rollout_weight: 0.3
  
  # 샘플 개수
  k_a: 5
  k_b: 5
  k_c: 5
```

### 5.2 훈련 하이퍼파라미터 (conf/training/default.yaml)

```yaml
training:
  epochs: 200
  batch_size: 50
  learning_rate: 1.0e-3
  weight_decay: 1.0e-5
  
  # KL Annealing
  kl_annealing_start: 0
  kl_annealing_end: 100
  kl_start_weight: 0.1
  kl_end_weight: 1.0
  
  # 단계별 가중치
  phase1_epochs: 50      # VAE 사전학습
  phase2_epochs: 150     # 전체 학습
  
  # 체크포인팅
  save_interval: 10
  checkpoint_dir: ./logs/checkpoints
  
  # W&B
  use_wandb: true

wandb:
  project: fitness-tracker
  name: vrae_experiment
```

---

## 6. 예상 결과 및 평가 메트릭

### 6.1 정량적 평가

```
1) VAE Reconstruction Error:
   - RMSE on valid timesteps (m_t=0)
   - Lower is better
   
2) Policy Prediction Accuracy:
   - MAE/RMSE of a_t predictions
   - Evaluate on held-out test set
   
3) Trajectory Diversity:
   - 125개 생성 궤적의 다양성 측정
   - 예: 궤적 간 평균 거리
   
4) Trajectory Fidelity:
   - 생성 궤적이 실제 궤적과 얼마나 유사한가
   - 예: DTW, Frechet distance
```

### 6.2 정성적 평가

```
1) 해석성:
   - z_a: "초기 상태" 클러스터 분석
   - z_b: 정책 스타일 차이 시각화
   - z_c: 천이 패턴 차이 시각화
   
2) 롤아웃 시각화:
   - 125개 궤적 모두 시각화
   - 실제 궤적과 비교
   
3) 마스킹 영향:
   - 마스킹 있을 때 vs 없을 때 비교
```

---

## 7. 실행 순서

```
1. 모델 구현 (models/vrae.py)
   └─ Encoder, Decoder, Policy, Transition 모두 구현
   
2. 설정 파일 작성 (conf/*.yaml)
   └─ 모델/훈련 하이퍼파라미터 정의
   
3. train.py 수정
   └─ VRAE 모델 로드 및 손실함수 통합
   └─ 다중 거리 메트릭 지원
   
4. 학습 실행
   └─ 배치 단위로 학습
   └─ W&B 모니터링
   
5. 평가 및 분석
   └─ 생성 궤적 시각화
   └─ 성능 지표 계산
```

---

## 8. 주요 고려사항

### 8.1 메모리 최적화
```
- 125개 궤적 동시 생성 시 메모리 폭증
- 해결: 배치 단위로 처리 (한 번에 K 크기만 생성)
- 또는 학습 시에는 선택된 상위 K개만 사용
```

### 8.2 학습 안정성
```
- KL divergence가 0으로 붕괴 가능 (KL collapse)
- 해결: KL annealing (천천히 증가)
- 또는 β-VAE 사용 (β > 1)
```

### 8.3 마스킹 처리
```
- 결측값은 0으로 채워지지만 마스크로 표시
- Loss 계산 시 결측 시점 제외
- Policy/Transition도 마스킹 기반 학습
```

---

## 요약

| 항목 | 값 |
|------|-----|
| **Input 형태** | (999, 1000, 11) [a, s, w, m] |
| **행동 (a)** | 7D: sleep_hours, workout_type, location, steps, calories_burned, distance_km, active_minutes |
| **상태 (s)** | 2D: heart_rate_avg, mood |
| **컨텍스트 (w)** | 1D: weather_conditions |
| **마스크 (m)** | 1D: 유효성 지시자 (필수) |
| **Model Type** | CVAE + Policy + Transition |
| **Latent Dims** | z_a=16 (상태), z_b=32 (정책), z_c=32 (천이) |
| **Samples** | K_a=5, K_b=5, K_c=5 → 125 combinations |
| **Distance Metrics** | RMSE, MAE, MAPE, Huber (선택 가능) |
| **Total Params** | ~500K (예상) |
| **Batch Size** | 50 |
| **Training Time** | ~2-4시간 (Apple Silicon) |
| **Output** | 125개 생성 행동 궤적 + 재구성 + 손실 |

