#!/usr/bin/env python3
"""
quick_test.py: 순차 실험 스크립트 빠른 테스트

작은 에포크로 설정해서 빠르게 테스트할 수 있습니다.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 빠른 테스트용 실험 설정 (에포크: 20)
QUICK_EXPERIMENTS = [
    {
        "name": "01_Baseline_Quick",
        "params": {
            "model.hidden_dim": 256,
            "model.num_layers": 2,
            "model.latent_state_dim": 16,
            "model.latent_policy_dim": 32,
            "model.latent_transition_dim": 32,
            "training.batch_size": 64,
            "training.learning_rate": 0.001,
            "training.epochs": 20,  # 빠른 테스트
            "training.early_stopping_patience": 1,
            "training.use_wandb": False,  # W&B 미사용
        }
    },
    {
        "name": "02_WideModel_Quick",
        "params": {
            "model.hidden_dim": 512,
            "model.num_layers": 2,
            "model.latent_state_dim": 16,
            "model.latent_policy_dim": 32,
            "model.latent_transition_dim": 32,
            "training.batch_size": 32,
            "training.learning_rate": 0.001,
            "training.epochs": 20,
            "training.early_stopping_patience": 3,
            "training.use_wandb": False,
        }
    },
]


def build_command(experiment):
    """실험 커맨드 생성"""
    params = experiment["params"]
    param_strs = []
    
    for key, value in params.items():
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)
        param_strs.append(f"{key}={value_str}")
    
    return f"python scripts/train.py {' '.join(param_strs)}"


def main():
    logger.info("=" * 80)
    logger.info("🧪 빠른 테스트 모드 (에포크: 20, W&B 미사용)")
    logger.info("=" * 80)
    
    results = []
    
    for idx, exp in enumerate(QUICK_EXPERIMENTS, 1):
        logger.info(f"\n[{idx}/{len(QUICK_EXPERIMENTS)}] 실험: {exp['name']}")
        
        cmd = build_command(exp)
        logger.info(f"실행: {cmd}\n")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd="/Users/neo/neomakes/human-wm"
            )
            
            if result.returncode == 0:
                logger.info(f"✅ 완료\n")
                results.append((exp['name'], "success"))
            else:
                logger.error(f"❌ 실패 (code: {result.returncode})\n")
                results.append((exp['name'], "failed"))
        
        except Exception as e:
            logger.error(f"❌ 에러: {e}\n")
            results.append((exp['name'], "error"))
    
    # 최종 결과
    logger.info("=" * 80)
    logger.info("📊 테스트 결과")
    logger.info("=" * 80)
    
    for name, status in results:
        status_emoji = "✅" if status == "success" else "❌"
        logger.info(f"{status_emoji} {name}: {status}")


if __name__ == "__main__":
    main()
