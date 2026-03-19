#!/bin/bash

# ============================================================================
# run_experiments_safe.sh: caffeinate를 포함한 안전한 순차 실험 실행
# 
# macOS에서 학습 중 Mac이 잠자기 모드로 전환되지 않도록 방지합니다.
# 프로세스가 완료되면 자동으로 절전 모드가 복구됩니다.
#
# 사용법:
#   bash scripts/run_experiments_safe.sh
#   또는
#   ./scripts/run_experiments_safe.sh
# ============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$PROJECT_DIR/scripts"
LOG_DIR="$PROJECT_DIR/logs"

echo "🔬 human-wm VRAE - 순차 실험 시작 (Sleep 모드 방지)"
echo "════════════════════════════════════════════════════════════"
echo "📁 프로젝트 디렉토리: $PROJECT_DIR"
echo "⏰ 시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Python 경로
PYTHON_EXEC="$(which python)"
echo "🐍 Python: $PYTHON_EXEC"
echo ""

# 실험 디렉토리 생성
mkdir -p "$LOG_DIR"

# caffeinate 옵션 설정
# -i: 유휴 상태 감지 (키보드/마우스 활동이 없을 때만 방지)
# -m: 디스플레이 잠자기 방지 (시스템은 잠들 수 있음)
# -s: 시스템 전체 잠자기 방지 (가장 안전)
CAFFEINATE_CMD="caffeinate -i"  # 기본값: 유휴 상태만 방지

echo "🛡️  Sleep 모드 방지 옵션: $CAFFEINATE_CMD"
echo "💡 옵션 설명:"
echo "   -i : 유휴 상태에서만 잠자기 방지 (권장, 활동 감지 가능)"
echo "   프로세스 완료 후: 자동으로 절전 모드 복구 ✅"
echo ""

# 실험 실행 (caffeinate 포함)
$CAFFEINATE_CMD $PYTHON_EXEC "$SCRIPT_DIR/run_experiments.py"

RESULT=$?

echo ""
echo "════════════════════════════════════════════════════════════"
echo "⏰ 종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
if [ $RESULT -eq 0 ]; then
    echo "✅ 모든 실험 완료!"
    echo "🔄 절전 모드가 자동으로 복구되었습니다."
else
    echo "❌ 실험 중 오류 발생 (종료 코드: $RESULT)"
    echo "🔄 절전 모드가 자동으로 복구되었습니다."
fi
echo "════════════════════════════════════════════════════════════"

exit $RESULT
