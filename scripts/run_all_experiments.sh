#!/bin/zsh
# run_all_experiments.sh: 모든 실험을 순차적으로 실행

cd "$(dirname "$0")/.." || exit 1

echo "🚀 human-wm VRAE 순차 실험 시작"
echo "환경: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# Python 스크립트 실행
python scripts/run_experiments.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ 모든 실험이 완료되었습니다!"
else
    echo ""
    echo "⚠️  실험 중 오류가 발생했습니다."
fi

exit $exit_code
