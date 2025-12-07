#!/usr/bin/env python3
"""
run_experiments.py: 순차적 실험 실행 스크립트

여러 실험을 순차적으로 실행하며, 각 실험이 조기 종료되면 
자동으로 다음 실험이 시작됩니다.

caffeinate를 통해 실행 시 macOS에서 Sleep 모드가 방지됩니다.
프로세스 완료 후 절전 모드가 자동으로 복구됩니다.

실험 구성:
1. Baseline (기본 설정)
2. Deep Model (더 깊은 모델)
3. Wide Model (더 넓은 모델)
4. Large Latent (큰 잠재 차원)
5. Optimized (최적화된 설정)
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# macOS Sleep 모드 감지 (caffeinate 확인)
# ============================================================================

def check_caffeinate():
    """
    macOS에서 caffeinate로 실행 중인지 확인
    환경 변수를 통해 감지
    """
    if sys.platform == "darwin":  # macOS만
        try:
            # 방법 1: 환경 변수 확인 (Makefile/shell에서 설정)
            if os.environ.get('CAFFEINATE_ENABLED') == '1':
                logger.info("✅ caffeinate 감지됨 - Sleep 모드가 방지되고 있습니다")
                logger.info("💡 프로세스 완료 후 절전 모드가 자동으로 복구됩니다")
                return True
            
            # 방법 2: 부모 프로세스 확인
            try:
                parent_pid = os.getppid()
                result = subprocess.run(
                    ["ps", "-p", str(parent_pid)],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                # caffeinate는 일반적으로 상위 프로세스에 있을 수 있음
                if "caffeinate" in result.stdout:
                    logger.info("✅ caffeinate 감지됨 - Sleep 모드가 방지되고 있습니다")
                    logger.info("💡 프로세스 완료 후 절전 모드가 자동으로 복구됩니다")
                    return True
            except:
                pass
            
            # caffeinate를 찾지 못한 경우
            logger.warning("⚠️  caffeinate 없이 실행 중입니다")
            logger.warning("💡 권장: caffeinate -i python scripts/run_experiments.py")
            logger.warning("💡 또는: make experiments-safe")
            return False
            
        except Exception as e:
            logger.debug(f"caffeinate 확인 실패: {e}")
            return False


# ============================================================================
# 실험 설정
# ============================================================================

EXPERIMENTS = [
    {
        "name": "01_Baseline",
        "description": "기본 설정",
        "params": {
            "model.hidden_dim": 256,
            "model.num_layers": 2,
            "model.latent_state_dim": 16,
            "model.latent_policy_dim": 32,
            "model.latent_transition_dim": 32,
            "training.batch_size": 32,
            "training.learning_rate": 0.001,
            "training.epochs": 200,
            "training.early_stopping_patience": 5,
            "training.use_wandb": True,
        }
    },
    {
        "name": "02_DeepModel",
        "description": "더 깊은 모델 (3 layers)",
        "params": {
            "model.hidden_dim": 256,
            "model.num_layers": 3,
            "model.latent_state_dim": 16,
            "model.latent_policy_dim": 32,
            "model.latent_transition_dim": 32,
            "training.batch_size": 32,
            "training.learning_rate": 0.001,
            "training.epochs": 200,
            "training.early_stopping_patience": 5,
            "training.use_wandb": True,
        }
    },
    {
        "name": "03_WideModel",
        "description": "더 넓은 모델 (512 hidden)",
        "params": {
            "model.hidden_dim": 512,
            "model.num_layers": 2,
            "model.latent_state_dim": 16,
            "model.latent_policy_dim": 32,
            "model.latent_transition_dim": 32,
            "training.batch_size": 32,
            "training.learning_rate": 0.001,
            "training.epochs": 200,
            "training.early_stopping_patience": 5,
            "training.use_wandb": True,
        }
    },
    {
        "name": "04_LargeLatent",
        "description": "큰 잠재 차원",
        "params": {
            "model.hidden_dim": 256,
            "model.num_layers": 2,
            "model.latent_state_dim": 32,
            "model.latent_policy_dim": 64,
            "model.latent_transition_dim": 64,
            "training.batch_size": 32,
            "training.learning_rate": 0.001,
            "training.epochs": 200,
            "training.early_stopping_patience": 5,
            "training.use_wandb": True,
        }
    },
    {
        "name": "05_Optimized",
        "description": "최적화된 설정 (512 hidden + large latent)",
        "params": {
            "model.hidden_dim": 512,
            "model.num_layers": 2,
            "model.latent_state_dim": 32,
            "model.latent_policy_dim": 64,
            "model.latent_transition_dim": 64,
            "training.batch_size": 32,
            "training.learning_rate": 0.0005,
            "training.epochs": 200,
            "training.early_stopping_patience": 5,
            "training.use_wandb": True,
        }
    },
]


# ============================================================================
# 유틸리티 함수
# ============================================================================

def build_command(experiment: Dict[str, Any]) -> str:
    """
    실험 설정으로부터 학습 커맨드 구성
    
    Args:
        experiment: 실험 설정 딕셔너리
    
    Returns:
        python scripts/train.py ... 형태의 커맨드
    """
    params = experiment["params"]
    param_strs = []
    
    for key, value in params.items():
        if isinstance(value, bool):
            value_str = str(value).lower()
        elif isinstance(value, str):
            value_str = f'"{value}"'
        else:
            value_str = str(value)
        param_strs.append(f"{key}={value_str}")
    
    cmd = f"python scripts/train.py {' '.join(param_strs)}"
    return cmd


def run_experiment(experiment: Dict[str, Any], experiment_idx: int, total_experiments: int) -> bool:
    """
    단일 실험 실행
    
    Args:
        experiment: 실험 설정
        experiment_idx: 현재 실험 번호 (1-indexed)
        total_experiments: 전체 실험 개수
    
    Returns:
        성공 여부
    """
    exp_name = experiment["name"]
    exp_desc = experiment["description"]
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"[{experiment_idx}/{total_experiments}] 실험 시작: {exp_name}")
    logger.info(f"설명: {exp_desc}")
    logger.info("=" * 80)
    
    # 커맨드 구성
    cmd = build_command(experiment)
    logger.info(f"\n실행 커맨드:")
    logger.info(f"{cmd}\n")
    
    try:
        # 실험 실행
        result = subprocess.run(
            cmd,
            shell=True,
            cwd="/Users/neo/ACCEL/WienerMachine/Benchmark/FitnessTracker"
        )
        
        if result.returncode == 0:
            logger.info(f"✅ [{exp_name}] 완료")
            return True
        elif result.returncode == 130:
            # Ctrl+C (KeyboardInterrupt)
            logger.warning(f"⚠️  [{exp_name}] 사용자에 의해 중단됨 (Ctrl+C)")
            return False
        else:
            logger.error(f"❌ [{exp_name}] 실패 (Exit code: {result.returncode})")
            return False
    
    except Exception as e:
        logger.error(f"❌ [{exp_name}] 예외 발생: {e}")
        return False


def main():
    """메인 실행 함수"""
    # macOS Sleep 모드 확인
    check_caffeinate()
    
    logger.info("=" * 80)
    logger.info("🚀 순차 실험 시작")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    results = {
        "start_time": start_time.isoformat(),
        "experiments": [],
        "total": len(EXPERIMENTS),
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }
    
    for idx, experiment in enumerate(EXPERIMENTS, 1):
        logger.info(f"\n📊 진행 상황: {idx-1}/{len(EXPERIMENTS)} 완료")
        
        success = run_experiment(experiment, idx, len(EXPERIMENTS))
        
        if success:
            results["completed"] += 1
            results["experiments"].append({
                "name": experiment["name"],
                "status": "completed"
            })
        else:
            results["experiments"].append({
                "name": experiment["name"],
                "status": "failed" if success is False else "cancelled"
            })
            # 실패 시 다음 실험으로 계속 진행할지 묻기
            if idx < len(EXPERIMENTS):
                logger.warning(f"\n⚠️  실험 {experiment['name']}이(가) 실패했습니다.")
                response = input("다음 실험을 계속하시겠습니까? (y/n): ").strip().lower()
                if response != 'y':
                    logger.info("👋 실험 중단")
                    results["cancelled"] = len(EXPERIMENTS) - idx
                    break
    
    # 최종 통계
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("📈 최종 결과")
    logger.info("=" * 80)
    logger.info(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"소요 시간: {duration}")
    logger.info(f"완료: {results['completed']}/{results['total']}")
    logger.info(f"실패: {len([e for e in results['experiments'] if e['status'] == 'failed'])}")
    logger.info(f"중단됨: {results['cancelled']}")
    
    # 각 실험 결과
    logger.info("\n실험 결과:")
    for exp_result in results["experiments"]:
        status_emoji = "✅" if exp_result["status"] == "completed" else "❌"
        logger.info(f"  {status_emoji} {exp_result['name']}: {exp_result['status']}")
    
    # 결과 파일 저장
    results_file = Path("/Users/neo/ACCEL/WienerMachine/Benchmark/FitnessTracker/logs") / "experiments_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📁 결과 저장: {results_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n👋 프로그램 종료 (Ctrl+C)")
        sys.exit(1)
