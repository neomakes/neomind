# ============================================================================
# Makefile: human-wm VRAE 편의 명령어
# 
# 사용법:
#   make help              - 전체 명령어 확인
#   make train-safe        - 단일 학습 (sleep 모드 방지)
#   make experiments-safe  - 순차 실험 (sleep 모드 방지)
#   make quick-test        - 빠른 테스트
# ============================================================================

.PHONY: help help-safe train train-safe experiments experiments-safe \
        quick-test test format clean logs

# ============================================================================
# 기본 명령어
# ============================================================================

help:
	@echo "🎯 human-wm VRAE - Makefile 명령어"
	@echo ""
	@echo "📚 주요 명령어:"
	@echo "  make help              - 이 도움말 표시"
	@echo "  make train             - 단일 학습 실행"
	@echo "  make experiments       - 순차 실험 실행"
	@echo "  make quick-test        - 빠른 테스트 (에포크 20)"
	@echo ""
	@echo "🛡️  Sleep 모드 방지 명령어 (권장):"
	@echo "  make train-safe        - 단일 학습 (sleep 모드 방지 ✅)"
	@echo "  make experiments-safe  - 순차 실험 (sleep 모드 방지 ✅)"
	@echo ""
	@echo "🧹 유틸리티:"
	@echo "  make logs              - 최근 로그 확인"
	@echo "  make clean             - 캐시 파일 정리"
	@echo ""

help-safe:
	@echo "🛡️  Sleep 모드 방지 설명"
	@echo ""
	@echo "caffeinate 옵션:"
	@echo "  -i : 유휴 상태에서만 잠자기 방지 (권장)"
	@echo "       키보드/마우스 활동이 없을 때만 작동"
	@echo "       프로세스 완료 후 절전 모드 자동 복구 ✅"
	@echo ""
	@echo "  -m : 디스플레이 잠자기만 방지"
	@echo "       시스템은 잠들 수 있음"
	@echo ""
	@echo "  -s : 시스템 전체 잠자기 방지 (강함)"
	@echo "       절대 방해받지 않는 학습"
	@echo ""
	@echo "권장: make train-safe 또는 make experiments-safe"

# ============================================================================
# 학습 명령어
# ============================================================================

train:
	@echo "🚀 단일 학습 시작..."
	python scripts/train.py training.epochs=100

train-safe:
	@echo "🛡️  Sleep 모드 방지 + 단일 학습 시작"
	@echo "💡 프로세스 완료 후 절전 모드가 자동으로 복구됩니다 ✅"
	@echo ""
	CAFFEINATE_ENABLED=1 caffeinate -i python scripts/train.py training.epochs=100

# ============================================================================
# 실험 명령어
# ============================================================================

experiments:
	@echo "🔬 순차 실험 시작..."
	python scripts/run_experiments.py

experiments-safe:
	@echo "🛡️  Sleep 모드 방지 + 순차 실험 시작"
	@echo "💡 프로세스 완료 후 절전 모드가 자동으로 복구됩니다 ✅"
	@echo ""
	CAFFEINATE_ENABLED=1 caffeinate -i python scripts/run_experiments.py

# ============================================================================
# 빠른 테스트
# ============================================================================

quick-test:
	@echo "🧪 빠른 테스트 시작 (에포크 20, ~5분)..."
	python scripts/quick_test.py

test: quick-test

# ============================================================================
# 유틸리티
# ============================================================================

logs:
	@echo "📋 최근 로그 (20줄):"
	@echo ""
	@tail -20 logs/runs/*/train.log 2>/dev/null || echo "로그 파일 없음"

clean:
	@echo "🧹 캐시 파일 정리..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ 정리 완료"

format:
	@echo "🎨 코드 포맷팅 (Black)..."
	@black scripts/*.py models/*.py 2>/dev/null || echo "Black 설치 필요: pip install black"

# ============================================================================
# 기본 타겟
# ============================================================================

.DEFAULT_GOAL := help

# 자주 사용하는 별칭
exp: experiments
exp-safe: experiments-safe
t: train
t-safe: train-safe
q: quick-test

.PHONY: exp exp-safe t t-safe q
