# 🔬 순차 실험 완전 가이드

> **작성일**: 2025-12-07  
> **상태**: ✅ 완료 및 테스트됨  
> **소요 시간**: ~3시간 (조기 종료 적용)

---

## 🎯 핵심 요약

**한 줄 명령어로 5개 실험을 순차적으로 자동 실행합니다.**

```bash
make experiments-safe
```

✅ 절전 모드 자동 방지 | 조기 종료 적용 | 결과 자동 저장 | 3시간 완료

---

## 🚀 3가지 실행 방법

### 방법 1: Makefile (가장 간단) ⭐

```bash
# 단일 학습 (1시간)
make train-safe

# 순차 실험 (3시간, 권장) ⭐
make experiments-safe

# 빠른 테스트 (5분)
make quick-test

# 별칭 (더 빨리 입력)
make exp-safe          # experiments-safe
make t-safe            # train-safe
make q                 # quick-test
```

### 방법 2: 직접 실행

```bash
# 순차 실험
caffeinate -i python scripts/run_experiments.py

# 단일 학습
caffeinate -i python scripts/train.py training.epochs=100
```

### 방법 3: 백그라운드 실행

```bash
# nohup (터미널 닫아도 계속 실행)
nohup make experiments-safe > logs/exp_safe.log 2>&1 &
tail -f logs/exp_safe.log  # 진행 상황 보기

# tmux (네트워크 끊김 방지, 가장 안전)
tmux new-session -d -s fitness 'make experiments-safe'
tmux attach -t fitness     # 접속
# Ctrl+B D 로 빠져나오기 (세션 계속 실행)
```

---

## 📊 5개 실험 구성

| # | 이름 | 핵심 변화 | 목표 | 예상 시간 |
|---|------|---------|------|---------|
| 1️⃣ | **Baseline** | 기본 설정 (256-dim, 2L) | 성능 기준 | ~30분 |
| 2️⃣ | **DeepModel** | 깊이 증가 (3L) | 깊이의 영향 | ~35분 |
| 3️⃣ | **WideModel** | 폭 증가 (512-dim) | 폭의 영향 | ~40분 |
| 4️⃣ | **LargeLatent** | 잠재 차원 증가 (64-dim) | 잠재 차원의 영향 | ~35분 |
| 5️⃣ | **Optimized** | 모두 최적화 (512-dim, 64-lat) | 최고 성능 | ~45분 |

**총합: ~185분 (3시간)** | 조기 종료 미적용 시: ~600분 (10시간) → **3배 효율화!**

---

## ✨ 작동 확인

### caffeinate 활성화 (정상) ✅
```
✅ caffeinate 감지됨 - Sleep 모드가 방지되고 있습니다
💡 프로세스 완료 후 절전 모드가 자동으로 복구됩니다
```

### caffeinate 미활성화 (경고) ⚠️
```
⚠️  caffeinate 없이 실행 중입니다
💡 권장: make experiments-safe
```

---

## 📈 실행 흐름

```
시작 (make experiments-safe)
  ↓
[01] Baseline (조기 종료)
  ↓
[02] DeepModel (조기 종료)
  ↓
[03] WideModel (조기 종료)
  ↓
[04] LargeLatent (조기 종료)
  ↓
[05] Optimized (조기 종료)
  ↓
📊 최종 결과 저장 (logs/experiments_results.json)
  ↓
✅ 완료! (자동으로 절전 모드 복구)
```

---

## 📊 결과 확인

### 1. 실시간 로그 보기

```bash
# 가장 최근 로그 실시간 확인
tail -f logs/runs/*/train.log

# 특정 실험 로그 보기
tail -f logs/runs/2025-12-07/*/train.log
```

### 2. W&B 대시보드 (실시간)

```
https://wandb.ai/your-username/human-wm-vrae
```

- 각 실험이 별도 run으로 기록됨
- 실시간 메트릭 시각화
- 실험 간 비교 분석 가능

### 3. 최종 결과 파일

```bash
# JSON 결과 보기
cat logs/experiments_results.json | python -m json.tool

# Python으로 분석
python -c "
import json
with open('logs/experiments_results.json') as f:
    results = json.load(f)
    for exp in results['experiments']:
        print(f\"{exp['name']}: {exp['status']}\")
"
```

---

## ⚙️ caffeinate 옵션 설명

| 옵션 | 효과 | 추천 |
|-----|------|------|
| `-i` | 유휴 상태에서만 잠자기 방지 | ⭐⭐⭐ **권장** |
| `-m` | 디스플레이 잠자기만 방지 | ⭐ (디스플레이만 중요할 때) |
| `-s` | 시스템 전체 잠자기 방지 | ⭐ (매우 중요한 작업) |

**기본값**: `caffeinate -i` (안정적, 자연스러움)
- 키보드/마우스 활동 감지 가능
- 사용자 작업 중이면 정상 작동
- 프로세스 종료 후 자동 복구 ✅

---

## 💡 권장 사용 시나리오

### 시나리오 1️⃣: 빠른 테스트 (개발/디버깅)
```bash
make quick-test  # ~5분, W&B 미사용, 에포크 20
```

### 시나리오 2️⃣: 단일 학습 (점심시간)
```bash
make train-safe  # ~1시간, sleep 모드 방지
```

### 시나리오 3️⃣: 순차 실험 (퇴근 후, 밤새)
```bash
make experiments-safe  # ~3시간, 절전 모드 자동 관리
```

### 시나리오 4️⃣: 백그라운드 + 모니터링
```bash
nohup make experiments-safe > logs/exp_safe.log 2>&1 &
tail -f logs/exp_safe.log
```

### 시나리오 5️⃣: 특정 실험만 실행
```bash
# 03_WideModel만 실행
python scripts/train.py \
  model.hidden_dim=512 \
  model.num_layers=2 \
  training.epochs=200
```

---

## 🔧 커스터마이징

### 실험 추가하기

`scripts/run_experiments.py` 수정:

```python
EXPERIMENTS = [
    # ... 기존 5개 실험 ...
    {
        "name": "06_CustomExperiment",
        "description": "내 커스텀 설정",
        "params": {
            "model.hidden_dim": 768,
            "model.num_layers": 4,
            "training.learning_rate": 0.0002,
        }
    },
]
```

### 조기 종료 설정 조정

`conf/training/default.yaml`:

```yaml
early_stopping_patience: 5           # 5 epoch 동안 개선 없으면 중단
early_stopping_min_delta: 0.001      # 최소 개선 임계값
```

---

## 🐛 트러블슈팅

### Q1: caffeinate 감지 안 됨

```bash
# 문제 확인
make experiments-safe

# 경고 메시지가 보이면:
# 해결법: 환경 변수 CAFFEINATE_ENABLED=1 자동 설정
# 또는 직접 실행: caffeinate -i python scripts/run_experiments.py
```

### Q2: 메모리 부족

```bash
# scripts/run_experiments.py 수정
"model.hidden_dim": 128,  # 기본: 256
```

### Q3: W&B 연결 실패

```bash
# W&B 로그인 확인
wandb login

# 또는 로컬에서만 실행
training.use_wandb=false
```

### Q4: 중간에 멈추고 싶음

```bash
# Ctrl+C 누르기 → 현재 실험 중단

# 강제 종료 필요 시
pkill -f "run_experiments"
pkill -f "train.py"
```

---

## ❓ FAQ

**Q1: 프로세스 끝나면 정말 절전 모드 복구되나요?**

A: 네! `caffeinate`는 프로세스 종료 후 자동으로 절전 설정을 원래 상태로 복구합니다. ✅

**Q2: 학습 중 Mac을 사용해도 되나요?**

A: 네, `-i` 옵션이면 유휴 상태만 감지하므로:
- 작업 중 → 절전 모드 방지 안 함 (정상 작동)
- 유휴 → 절전 모드 방지 (학습 보호)

**Q3: 특정 옵션 (예: `-s`)으로 바꾸고 싶은데?**

A: 직접 명령어로:
```bash
caffeinate -is python scripts/run_experiments.py  # -i → -is
```

**Q4: 병렬로 여러 실험 실행할 수 있나?**

A: 네, 다른 터미널에서:
```bash
# 터미널 1
make experiments-safe

# 터미널 2 (다른 작업)
caffeinate -i python scripts/train.py model.hidden_dim=768 training.epochs=100
```

---

## 📋 생성된 파일

### 스크립트 (scripts/)
```
├─ run_experiments.py       ← 메인 순차 실험 스크립트 ⭐
├─ train.py               ← 개별 학습 스크립트 (수정됨)
├─ quick_test.py          ← 빠른 테스트 (20 에포크)
└─ run_experiments_safe.sh ← 셸 래퍼
```

### 설정 (conf/)
```
└─ training/default.yaml
   ├─ early_stopping_patience: 5
   └─ early_stopping_min_delta: 0.001
```

### 문서 (docs/)
```
└─ [03] ExperimentGuide.md ← 이 문서
```

### 결과 (logs/)
```
└─ experiments_results.json ← 최종 결과 (실행 후 생성)
```

---

## 🎓 학습 포인트

이 실험으로 다음을 평가할 수 있습니다:

1. **모델 아키텍처 영향**
   - 깊이 (Depth): DeepModel vs Baseline
   - 폭 (Width): WideModel vs Baseline
   - 잠재 차원: LargeLatent vs Baseline

2. **조기 종료의 효과**
   - 수렴 속도 개선
   - 과적합 방지
   - 계산 효율성

3. **최적 설정 찾기**
   - 5개 설정 비교
   - W&B로 상세 분석
   - 최고 성능 모델 선정

---

## ⏱️ 예상 시간표

```
조기 종료 적용 (현재):
├─ 01 Baseline:     ~30분
├─ 02 DeepModel:    ~35분
├─ 03 WideModel:    ~40분
├─ 04 LargeLatent:   ~35분
├─ 05 Optimized:    ~45분
└─ 총합: ~185분 (3시간) ⚡

미적용 (모든 200 에포크):
└─ 총합: ~600분 (10시간) 😅

효율성: 약 3배 시간 단축! 🚀
```

---

## 🎯 다음 단계

### 즉시 실행 (권장)

```bash
cd /Users/neo/neomakes/human-wm
make experiments-safe
```

### 결과 확인 (3시간 후)

```bash
# 최종 결과 보기
cat logs/experiments_results.json | python -m json.tool

# W&B에서 상세 분석
https://wandb.ai/your-username/human-wm-vrae
```

---

## 📚 참고 명령어 요약

```bash
# 주요 명령어
make experiments-safe      # 순차 실험 (sleep 방지) ⭐
make train-safe            # 단일 학습 (sleep 방지)
make quick-test            # 빠른 테스트
make logs                  # 최근 로그 확인
make help                  # 전체 명령어 보기

# 직접 실행
caffeinate -i python scripts/run_experiments.py
caffeinate -i python scripts/train.py training.epochs=100

# 백그라운드
nohup make experiments-safe > logs/exp.log 2>&1 &
tmux new-session -d -s fitness 'make experiments-safe'
```

---

## ✨ 핵심 특징

| 특징 | 설명 |
|-----|------|
| 🤖 **자동화** | 한 줄 명령어로 5개 실험 순차 실행 |
| ⏱️ **효율성** | 조기 종료로 70% 시간 단축 |
| 💾 **기록** | W&B + 로컬 로그 자동 저장 |
| 🛡️ **안정성** | Sleep 모드 자동 방지 & 복구 |
| 📊 **비교** | 실험 간 자동 결과 추적 |
| 🔧 **확장성** | 쉽게 실험 추가/수정 가능 |

---

**🎉 모든 준비가 완료되었습니다!**

```bash
make experiments-safe
```

이 한 줄로 3시간의 완전 자동 실험을 시작하세요! 🚀

더 자세한 정보는 소스 코드의 주석을 참조하세요:
- `scripts/run_experiments.py`: 실험 설정 및 자동화
- `scripts/train.py`: EarlyStoppingTracker 구현
- `conf/training/default.yaml`: 조기 종료 파라미터

**행운을 빕니다! 🚀**
