"""
train.py: VRAE 모델 학습 스크립트

상태(s), 행동(a), 컨텍스트(w), 마스크(m)를 사용하여:
1. VAE로 행동과 상태의 확률분포 학습
2. 정책과 천이 네트워크로 의사결정 역학 학습

특성 구성 (10):
  - 행동 (7): sleep_hours, workout_type, location, steps, calories_burned, distance_km, active_minutes
  - 상태 (2): heart_rate_avg, mood
  - 컨텍스트 (1): weather_conditions
  - 마스크 (1): 유효성 지시자

데이터 처리:
  - 마스크를 기반으로 유효한 시점만 loss 계산
  - 결측값은 0으로 채워짐 (마스크로 구분)
"""

import os
import sys
import time
import numpy as np
import pandas as pdf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

# 모델 임포트
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model import VRAE

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from tqdm import tqdm
import logging
from datetime import datetime

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# macOS Sleep 모드 감지 (caffeinate 확인)
# ============================================================================

def check_caffeinate():
    """
    macOS에서 caffeinate로 실행 중인지 확인
    프로세스 완료 후 절전 모드가 자동으로 복구됨
    """
    if sys.platform == "darwin":  # macOS만
        try:
            # 환경 변수로 caffeinate 활성화 여부 확인
            if os.environ.get('CAFFEINATE_ENABLED') == '1':
                logger.info("✅ caffeinate 감지됨 - Sleep 모드가 방지되고 있습니다")
                logger.info("💡 프로세스 완료 후 절전 모드가 자동으로 복구됩니다")
                return True
            else:
                logger.warning("⚠️  caffeinate 없이 실행 중입니다")
                logger.warning("💡 권장: caffeinate -i python scripts/train.py ...")
                logger.warning("💡 또는: make train-safe")
                return False
        except Exception as e:
            logger.debug(f"caffeinate 확인 실패: {e}")


# ============================================================================
# Early Stopping 클래스
# ============================================================================

class EarlyStoppingTracker:
    """
    검증 손실 기반 조기 종료 모니터링
    
    Args:
        patience: 개선이 없을 때 몇 epoch을 기다릴지
        min_delta: 최소 개선 기준 (이보다 작은 개선은 무시)
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.wait_count = 0
        self.best_epoch = 0
        self.best_model_state = None
    
    def step(self, val_loss: float, epoch: int) -> Tuple[bool, bool]:
        """
        검증 손실을 기반으로 조기 종료 여부 결정
        
        Args:
            val_loss: 현재 검증 손실
            epoch: 현재 epoch
        
        Returns:
            (should_stop, improved): 멈춰야 하는지, 개선되었는지
        """
        improvement = self.best_val_loss - val_loss
        
        if improvement > self.min_delta:
            # 개선됨
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.wait_count = 0
            return False, True
        else:
            # 개선 없음
            self.wait_count += 1
            if self.wait_count >= self.patience:
                return True, False  # 멈춤
            return False, False
    
    def get_info(self) -> str:
        """현재 상태 정보 반환"""
        return (
            f"Best Epoch: {self.best_epoch}, "
            f"Best Loss: {self.best_val_loss:.4f}, "
            f"Wait Count: {self.wait_count}/{self.patience}"
        )


def generate_experiment_name(cfg: DictConfig) -> str:
    """
    모델 구성에 기반한 실험명 생성
    예: exp_z16_64_64_k3_3_3_h64_l2
    
    포함 정보:
    - z: 잠재 차원 (latent_state_dim_latent_policy_dim_latent_transition_dim)
    - k: 샘플 개수 (k_a_k_b_k_c)
    - h: 은닉층 차원 (hidden_dim)
    - l: 레이어 수 (num_layers)
    """
    z_dims = f"{cfg.model.latent_state_dim}_{cfg.model.latent_policy_dim}_{cfg.model.latent_transition_dim}"
    k_samples = f"{cfg.model.k_a}_{cfg.model.k_b}_{cfg.model.k_c}"
    hidden = cfg.model.hidden_dim
    layers = cfg.model.num_layers
    
    return f"exp_z{z_dims}_k{k_samples}_h{hidden}_l{layers}"


def setup_logging_dir(cfg: DictConfig, hydra_dir: Path, use_wandb: bool = False) -> Tuple[Path, Any]:
    """
    로깅 디렉토리 설정 및 wandb 초기화
    
    Hydra의 기본 구조(logs/runs/날짜/시간/)를 유지하면서,
    .hydra와 함께 .wandb 폴더도 생성하여 W&B 파일 관리
    
    Args:
        cfg: Hydra 설정
        hydra_dir: Hydra가 생성한 로그 디렉토리 (logs/runs/날짜/시간/)
        use_wandb: W&B 사용 여부
    
    Returns:
        (experiment_dir, wandb_run): 실험 디렉토리와 W&B 실행 객체
    """
    # .wandb 디렉토리 생성 (Hydra의 .hydra와 동일 위치)
    wandb_dir = hydra_dir / ".wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    
    # 실험명 생성
    exp_name = generate_experiment_name(cfg)
    
    if use_wandb:
        # W&B 초기화 - .wandb 디렉토리 내에서 실행
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(wandb_dir),
        )
        
        logger.info(f"W&B run initialized: {wandb_run.name} (ID: {wandb_run.id})")
        logger.info(f"W&B directory: {wandb_dir}")
        
        return hydra_dir, wandb_run
    else:
        logger.info(f"W&B disabled - logs will be saved to: {hydra_dir}")
        return hydra_dir, None


# ============================================================================
# 1. 데이터 로더 클래스
# ============================================================================

class TrajectoryDataset:
    """
    fitness_tracker_data.npz에서 데이터를 로드하는 데이터셋
    
    구성 (10개 특성):
      - 행동 (7): [sleep_hours, workout_type, location, steps, calories_burned, distance_km, active_minutes]
      - 상태 (2): [heart_rate_avg, mood]
      - 컨텍스트 (1): [weather_conditions]
      - 마스크 (1): [유효성]
    """
    
    def __init__(self, data_path: str, normalize: bool = True):
        """
        Args:
            data_path: fitness_tracker_data.npz 파일 경로
            normalize: 연속형 변수 정규화 여부
        """
        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # .npz 파일 로드
        data = np.load(data_path, allow_pickle=True)
        features = data['features'].astype(np.float32)  # (999, 1000, 10)
        mask = data['mask'].astype(np.float32)          # (999, 1000, 1)
        
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Mask shape: {mask.shape}")
        logger.info(f"Mask validity: {(mask == 0).mean():.2%}")
        
        # 특성 분리
        # features: [steps, calories_burned, distance_km, active_minutes, sleep_hours, heart_rate_avg, 
        #            workout_type, weather_conditions, location, mood]
        
        # 행동 (a): steps, calories_burned, distance_km, active_minutes, sleep_hours, workout_type, location
        # 상태 (s): heart_rate_avg, mood
        # 컨텍스트 (w): weather_conditions
        
        self.action = np.concatenate([
            features[:, :, 0:5],      # steps, calories_burned, distance_km, active_minutes, sleep_hours
            features[:, :, 6:7],      # workout_type
            features[:, :, 8:9],      # location
        ], axis=-1).astype(np.float32)  # (999, 1000, 7)
        
        self.state = features[:, :, [5, 9]].astype(np.float32)  # heart_rate_avg, mood (999, 1000, 2)
        self.context = features[:, :, 7:8].astype(np.float32)   # weather_conditions (999, 1000, 1)
        self.mask = mask.astype(np.float32)  # (999, 1000, 1)
        
        logger.info(f"Action shape: {self.action.shape}")
        logger.info(f"State shape: {self.state.shape}")
        logger.info(f"Context shape: {self.context.shape}")
        logger.info(f"Mask shape: {self.mask.shape}")
        
        # 정규화
        if normalize:
            self._normalize()
    
    def _normalize(self):
        """연속형 변수 정규화"""
        # 행동 정규화 (연속형: 0-4 indices = steps, calories, distance, active_minutes, sleep_hours)
        for idx in range(5):
            feat = self.action[:, :, idx]
            valid_mask = (self.mask[:, :, 0] == 0).astype(bool)
            valid_values = feat[valid_mask]
            
            if len(valid_values) > 0:
                mean = valid_values.mean()
                std = valid_values.std() + 1e-8
                self.action[:, :, idx] = (feat - mean) / std
                logger.info(f"Normalized action[{idx}]: mean={mean:.2f}, std={std:.2f}")
        
        # 상태 정규화 (heart_rate는 연속형)
        feat = self.state[:, :, 0]  # heart_rate
        valid_mask = (self.mask[:, :, 0] == 0).astype(bool)
        valid_values = feat[valid_mask]
        
        if len(valid_values) > 0:
            mean = valid_values.mean()
            std = valid_values.std() + 1e-8
            self.state[:, :, 0] = (feat - mean) / std
            logger.info(f"Normalized state[heart_rate]: mean={mean:.2f}, std={std:.2f}")
    
    def __len__(self):
        return len(self.action)
    
    def __getitem__(self, idx: int):
        """사용자 인덱스에서 데이터 반환"""
        return {
            'action': self.action[idx],
            'state': self.state[idx],
            'context': self.context[idx],
            'mask': self.mask[idx],
        }

# ============================================================================
# 2. 학습 스텝 함수
# ============================================================================

def eval_step(
    model: nn.Module,
    batch_a: mx.array,
    batch_s: mx.array,
    batch_w: mx.array,
    batch_m: mx.array,
    kld_weight: float,
    distance_type: str = "huber",
    huber_delta: float = 1.0,
    w_vae: float = 1.0,
    w_action: float = 0.5,
    w_transition: float = 0.5,
    w_rollout: float = 0.3,
) -> Dict[str, mx.array]:
    """단일 평가 스텝 (그래디언트 계산 없음)"""
    # 1. 인코더
    mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c = model.encode(batch_a, batch_s, batch_w, batch_m)

    # 2. 샘플링 (평가 시에는 분포의 평균을 사용해 결정적인 결과를 얻음)
    # rollout과 차원을 맞추기 위해 k값 만큼 복제
    z_a_eval = mx.expand_dims(mu_a, axis=1)
    z_b_eval = mx.expand_dims(mu_b, axis=1)
    z_c_eval = mx.expand_dims(mu_c, axis=1)
    z_a = mx.repeat(z_a_eval, model.k_a, axis=1)
    z_b = mx.repeat(z_b_eval, model.k_b, axis=1)
    z_c = mx.repeat(z_c_eval, model.k_c, axis=1)

    # 3. 디코더
    a_recon, s_recon = model.decode(z_a, z_b, z_c, batch_w)

    # 4. 손실 계산
    vae_loss, vae_metrics = model.loss_function(
        batch_a, batch_s, batch_w, batch_m,
        mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c,
        a_recon, s_recon, kld_weight=kld_weight
    )
    action_loss = model.compute_policy_loss(batch_a, batch_s, batch_w, batch_m, z_b_eval[:, 0], distance_type, huber_delta)
    transition_loss = model.compute_transition_loss(batch_s, batch_a, batch_w, batch_m, z_c_eval[:, 0], distance_type, huber_delta)
    rollout_loss = model.compute_rollout_loss(batch_a, batch_s[:, 0], batch_w, batch_m, z_a, z_b, z_c, distance_type, huber_delta)

    total_loss = w_vae * vae_loss + w_action * action_loss + w_transition * transition_loss + w_rollout * rollout_loss

    metrics = {
        'total_loss': total_loss, 'vae_loss': vae_loss, 'action_loss': action_loss,
        'transition_loss': transition_loss, 'rollout_loss': rollout_loss,
        'recon_loss_action': vae_metrics['recon_loss_action'],
        'recon_loss_state': vae_metrics['recon_loss_state'],
        'kl_loss': mx.mean(vae_metrics['kl_loss']),
    }
    return metrics


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch_a: mx.array,
    batch_s: mx.array,
    batch_w: mx.array,
    batch_m: mx.array,
    kld_weight: float,
    distance_type: str = "huber",
    huber_delta: float = 1.0,
    w_vae: float = 1.0,
    w_action: float = 0.5,
    w_transition: float = 0.5,
    w_rollout: float = 0.3,
) -> Tuple[Dict[str, mx.array], Dict[str, float]]:
    """
    단일 학습 스텝 (VAE + 정책 + 천이 + 롤아웃 손실)
    시간 측정 포함
    
    Args:
        model: VRAE 모델
        optimizer: 옵티마이저
        batch_a: (B, T, action_dim) 행동
        batch_s: (B, T, state_dim) 상태
        batch_w: (B, T, context_dim) 컨텍스트
        batch_m: (B, T, 1) 마스크
        kld_weight: KL 손실 가중치
        distance_type: 거리 메트릭
        huber_delta: Huber loss delta
        w_vae, w_action, w_transition, w_rollout: 손실 가중치
    
    Returns:
        (metrics_dict, timings_dict)
    """
    timings = {}
    
    def loss_fn(model):
        # 1. 인코더
        t_encode = time.time()
        mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c = model.encode(batch_a, batch_s, batch_w, batch_m)
        timings['encode'] = time.time() - t_encode

        # 2. 샘플링
        t_sample = time.time()
        z_a, z_b, z_c = model.sample_latents(mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c)
        timings['sample'] = time.time() - t_sample
        
        # 3. 디코더 (재구성)
        t_decode = time.time()
        a_recon, s_recon = model.decode(z_a, z_b, z_c, batch_w)
        timings['decode'] = time.time() - t_decode

        # 4. VAE 손실
        t_vae = time.time()
        vae_loss, vae_metrics = model.loss_function(
            batch_a, batch_s, batch_w, batch_m,
            mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c,
            a_recon, s_recon,
            kld_weight=kld_weight,
        )
        timings['vae_loss'] = time.time() - t_vae

        # 5. 정책 손실: π(a_t | s_t, w_t; z_b)
        t_policy = time.time()
        action_loss = model.compute_policy_loss(
            batch_a, batch_s, batch_w, batch_m, z_b[:, 0], distance_type, huber_delta
        )
        timings['policy_loss'] = time.time() - t_policy

        # 6. 천이 손실: τ(s_{t+1} | s_t, a_t, w_t; z_c)
        t_transition = time.time()
        transition_loss = model.compute_transition_loss(
            batch_s, batch_a, batch_w, batch_m, z_c[:, 0], distance_type, huber_delta
        )
        timings['transition_loss'] = time.time() - t_transition

        # 7. 롤아웃 손실: 125개 생성 궤적과 실제 궤적의 거리
        t_rollout = time.time()
        rollout_loss = model.compute_rollout_loss(
            batch_a, batch_s[:, 0], batch_w, batch_m, z_a, z_b, z_c, distance_type, huber_delta
        )
        timings['rollout_loss'] = time.time() - t_rollout

        # 8. 전체 손실 (가중치 적용 및 배치 평균)
        total_loss = (
            w_vae * vae_loss +
            w_action * action_loss +
            w_transition * transition_loss +
            w_rollout * rollout_loss
        )

        # 로깅을 위한 메트릭 딕셔너리
        metrics = {
            'total_loss': total_loss,
            'vae_loss': vae_loss,
            'action_loss': action_loss,
            'transition_loss': transition_loss,
            'rollout_loss': rollout_loss,
            'recon_loss_action': vae_metrics['recon_loss_action'],
            'recon_loss_state': vae_metrics['recon_loss_state'],
            'kl_loss': mx.mean(vae_metrics['kl_loss']),
        }
        
        return total_loss, metrics
    
    t_total = time.time()
    
    # 손실과 그래디언트 계산
    t_grad = time.time()
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    (loss, metrics), grads = loss_and_grad_fn(model)
    timings['grad_compute'] = time.time() - t_grad
    
    # NaN 체크 (Epoch 2+ 디버깅)
    total_loss_value = metrics['total_loss']
    if mx.isnan(total_loss_value):
        logger.warning(f"⚠️  NaN loss detected! VAE: {metrics['vae_loss']:.4f}, Action: {metrics['action_loss']:.4f}, Transition: {metrics['transition_loss']:.4f}, Rollout: {metrics['rollout_loss']:.4f}")
    
    # 그래디언트 클리핑 (기울기 폭발 방지)
    t_clip = time.time()
    def clip_gradient(g):
        if hasattr(g, 'ndim') and g.ndim > 0:
            return mx.clip(g, a_min=-1.0, a_max=1.0)
        return g
    
    grads = tree_map(clip_gradient, grads)
    timings['grad_clip'] = time.time() - t_clip
    
    # 옵티마이저 업데이트
    t_optim = time.time()
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    timings['optimizer'] = time.time() - t_optim
    
    timings['total'] = time.time() - t_total
    
    return metrics, timings

# ============================================================================
# 3. 메인 학습 함수
# ============================================================================

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    메인 학습 루프
    
    Args:
        cfg: Hydra 설정
    """
    # ========================================================================
    # 0. Sleep 모드 확인 (macOS)
    # ========================================================================
    check_caffeinate()
    
    # ========================================================================
    # 1. 로깅 설정
    # ========================================================================
    # HydraConfig에서 런타임 정보 얻기
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    
    # Hydra의 출력 디렉토리 (logs/runs/날짜/시간/)
    hydra_dir = Path(hydra_cfg.runtime.output_dir)
    
    # W&B 설정 및 .wandb 폴더 생성
    experiment_dir, wandb_run = setup_logging_dir(cfg, hydra_dir, cfg.training.use_wandb)
    
    # 로그 파일을 실험 디렉토리에 저장
    log_file = experiment_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("FitnessTracker VRAE Model Training")
    logger.info("=" * 80)
    logger.info(f"Experiment: {generate_experiment_name(cfg)}")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # ========================================================================
    # 1. 데이터 로드
    # ========================================================================
    logger.info("\n[1/5] Loading data...")
    dataset = TrajectoryDataset(
        data_path=cfg.data.path,
        normalize=cfg.data.normalize,
    )
    
    # NumPy 배열로 유지 (배치 인덱싱용)
    actions = dataset.action     # (999, 1000, 7)
    states = dataset.state       # (999, 1000, 2)
    contexts = dataset.context   # (999, 1000, 1)
    masks = dataset.mask         # (999, 1000, 1)
    
    # 데이터 분할 (Train: 90%, Validation: 10%)
    num_users = len(dataset)
    indices = np.arange(num_users)
    np.random.shuffle(indices)
    
    val_split = int(num_users * cfg.data.val_split_ratio)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]
    
    logger.info(f"Data split: {len(train_indices)} train, {len(val_indices)} validation samples")
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Action shape: {actions.shape}")
    logger.info(f"State shape: {states.shape}")
    logger.info(f"Context shape: {contexts.shape}")
    logger.info(f"Mask shape: {masks.shape}")
    
    # ========================================================================
    # 2. 모델 생성
    # ========================================================================
    logger.info("\n[2/5] Creating VRAE model...")
    
    # VRAE 모델 초기화
    model = VRAE(
        action_dim=cfg.model.action_dim,
        state_dim=cfg.model.state_dim,
        context_dim=cfg.model.context_dim,
        latent_dim_a=cfg.model.latent_state_dim,
        latent_dim_b=cfg.model.latent_policy_dim,
        latent_dim_c=cfg.model.latent_transition_dim,
        rnn_hidden_dim=cfg.model.hidden_dim,
        mlp_hidden_dim=cfg.model.hidden_dim,  # MLP도 동일 차원 사용
        distance_type=cfg.model.distance_type,
        huber_delta=cfg.model.huber_delta,
        kld_weight=cfg.model.get('w_vae', 1.0),
        action_weight=cfg.model.get('w_action', 0.5),
        rollout_weight=cfg.model.get('w_rollout', 0.3),
        k_a=cfg.model.k_a,
        k_b=cfg.model.k_b,
        k_c=cfg.model.k_c,
    )
    mx.eval(model.parameters())

    # 모델 파라미터 개수 계산
    total_params = 0
    for _, v in tree_flatten(model.parameters()):
        if hasattr(v, 'size'):
            total_params += v.size
    logger.info(f"Model created successfully")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Model architecture:")
    logger.info(f"  - Action dim: {cfg.model.action_dim}")
    logger.info(f"  - State dim: {cfg.model.state_dim}")
    logger.info(f"  - Context dim: {cfg.model.context_dim}")
    logger.info(f"  - Latent dims: z_a={cfg.model.latent_state_dim}, z_b={cfg.model.latent_policy_dim}, z_c={cfg.model.latent_transition_dim}")
    
    # ========================================================================
    # 3. 옵티마이저 설정
    # ========================================================================
    logger.info("\n[3/5] Setting up optimizer...")
    optimizer = optim.Adam(learning_rate=cfg.training.learning_rate)
    
    logger.info(f"Learning rate: {cfg.training.learning_rate}")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info(f"Epochs: {cfg.training.epochs}")
    
    # ========================================================================
    # 4. 체크포인트 디렉토리 설정
    # ========================================================================
    logger.info("\n[4/5] Setting up checkpoint directory...")
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # ========================================================================
    # 5. 학습 루프
    # ========================================================================
    logger.info("\n[5/5] Starting training...")

    batch_size = cfg.training.batch_size
    num_train_batches = (len(train_indices) + batch_size - 1) // batch_size
    num_val_batches = (len(val_indices) + batch_size - 1) // batch_size
    
    # 체크포인트 디렉토리 설정
    if cfg.training.use_wandb and wandb.run is not None:
        # W&B 사용 시, W&B 실행 디렉토리 내에 저장
        checkpoint_dir = Path(wandb.run.dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to W&B directory: {checkpoint_dir}")
    else:
        # W&B 미사용 시, 설정 파일의 기본 경로 사용
        checkpoint_dir = Path(cfg.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 조기 종료 초기화
    early_stopping_patience = getattr(cfg.training, 'early_stopping_patience', 5)
    early_stopping_min_delta = getattr(cfg.training, 'early_stopping_min_delta', 0.001)
    early_stopper = EarlyStoppingTracker(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta
    )
    logger.info(f"Early stopping configured: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    
    for epoch in range(cfg.training.epochs):
        # 학습률 스케줄러 (매 에포크 시작 시 업데이트)
        if cfg.training.use_lr_scheduler:
            new_lr = cfg.training.learning_rate * (cfg.training.lr_decay_rate ** epoch)
            optimizer.learning_rate = mx.array(new_lr)
            current_lr = new_lr
        else:
            current_lr = cfg.training.learning_rate

        # KL annealing
        if epoch < cfg.training.kl_annealing_end:
            progress = epoch / cfg.training.kl_annealing_end
            kld_weight = cfg.training.kl_start_weight + \
                        (cfg.training.kl_end_weight - cfg.training.kl_start_weight) * progress
        else:
            kld_weight = cfg.training.kl_end_weight
        
        # 배치 인덱스 생성
        np.random.shuffle(train_indices)
        
        # 에포크 메트릭 초기화
        epoch_metrics = {
            'total_loss': 0.0, 'vae_loss': 0.0, 'action_loss': 0.0, 'transition_loss': 0.0, 'rollout_loss': 0.0,
            'recon_loss_action': 0.0, 'recon_loss_state': 0.0, 'kl_loss': 0.0}
        
        # --- 훈련 루프 ---
        model.train()
        pbar = tqdm(range(num_train_batches), desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]")
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_indices))
            batch_indices = train_indices[start_idx:end_idx]
            
            # 배치 구성
            batch_a = mx.array(actions[batch_indices])
            batch_s = mx.array(states[batch_indices])   # (B, T, 2)
            batch_w = mx.array(contexts[batch_indices]) # (B, T, 1)
            batch_m = mx.array(masks[batch_indices])    # (B, T, 1)
            
            # 학습 스텝
            metrics, timings = train_step(
                model=model,
                optimizer=optimizer,
                batch_a=batch_a,
                batch_s=batch_s,
                batch_w=batch_w,
                batch_m=batch_m,
                kld_weight=kld_weight,
                distance_type=cfg.model.distance_type,
                huber_delta=cfg.model.huber_delta,
                w_vae=cfg.training.get('w_vae', 1.0),
                w_action=cfg.training.get('w_action', 0.5),
                w_transition=cfg.training.get('w_transition', 0.5),
                w_rollout=cfg.training.get('w_rollout', 0.3),
            )

            # 메트릭 누적
            for k, v in metrics.items():
                epoch_metrics[k] += v.item()

            # 배치 상세 로깅 (첫 배치와 문제 발생 시)
            if batch_idx == 0 or mx.isnan(metrics['total_loss']):
                # 시간 정보 로깅
                timing_str = (
                    f"[Timings] Encode: {timings['encode']:.3f}s | "
                    f"Sample: {timings['sample']:.3f}s | "
                    f"Decode: {timings['decode']:.3f}s | "
                    f"VAE: {timings['vae_loss']:.3f}s | "
                    f"Policy: {timings['policy_loss']:.3f}s | "
                    f"Transition: {timings['transition_loss']:.3f}s | "
                    f"Rollout: {timings['rollout_loss']:.3f}s | "
                    f"GradComp: {timings['grad_compute']:.3f}s | "
                    f"GradClip: {timings['grad_clip']:.3f}s | "
                    f"Optimizer: {timings['optimizer']:.3f}s | "
                    f"Total: {timings['total']:.3f}s"
                )
                logger.info(
                    f"  Batch {batch_idx}: Total={metrics['total_loss']:.4f} | VAE={metrics['vae_loss']:.4f} | "
                    f"Recon_A={metrics['recon_loss_action']:.4f} | Recon_S={metrics['recon_loss_state']:.4f} | "
                    f"KL={metrics['kl_loss']:.4f} | Action={metrics['action_loss']:.4f} | "
                    f"Transition={metrics['transition_loss']:.4f} | Rollout={metrics['rollout_loss']:.4f}"
                )
                logger.info(timing_str)
                
                # W&B에도 시간 정보 로깅
                if cfg.training.use_wandb and wandb_run is not None:
                    wandb.log({
                        'batch_timings/encode': timings['encode'],
                        'batch_timings/sample': timings['sample'],
                        'batch_timings/decode': timings['decode'],
                        'batch_timings/vae_loss': timings['vae_loss'],
                        'batch_timings/policy_loss': timings['policy_loss'],
                        'batch_timings/transition_loss': timings['transition_loss'],
                        'batch_timings/rollout_loss': timings['rollout_loss'],
                        'batch_timings/grad_compute': timings['grad_compute'],
                        'batch_timings/grad_clip': timings['grad_clip'],
                    'batch_timings/optimizer': timings['optimizer'],
                    'batch_timings/total': timings['total'],
                    })

            # 모든 배치마다 wandb 로깅 (use_wandb=True인 경우)
            if wandb_run is not None:
                step_log_data = {
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "global_step": epoch * num_train_batches + batch_idx,
                    "kld_weight": kld_weight,
                    "learning_rate": current_lr,
                }
                # 모든 손실 메트릭 로깅
                for k, v in metrics.items():
                    step_log_data[f"train/{k}"] = v.item()
                wandb_run.log(step_log_data)

            pbar.set_postfix({"loss": f"{metrics['total_loss'].item():.4f}", "kld_w": f"{kld_weight:.3f}", "lr": f"{current_lr:.6f}"})
        
        # 에포크 평균 메트릭 계산 및 로깅
        avg_train_metrics = {k: v / num_train_batches for k, v in epoch_metrics.items()}
        
        log_str = (
            f"Epoch {epoch+1} [Train] - "
            f"Total Loss: {avg_train_metrics['total_loss']:.4f} | "
            f"VAE: {avg_train_metrics['vae_loss']:.4f} | "
            f"Recon_A: {avg_train_metrics['recon_loss_action']:.4f} | "
            f"Recon_S: {avg_train_metrics['recon_loss_state']:.4f} | "
            f"KL: {avg_train_metrics['kl_loss']:.4f} | "
            f"Action: {avg_train_metrics['action_loss']:.4f} | "
            f"Transition: {avg_train_metrics['transition_loss']:.4f} | "
            f"Rollout: {avg_train_metrics['rollout_loss']:.4f}"
        )
        logger.info(log_str)
        
        # --- 검증 루프 ---
        model.eval()
        val_metrics = {k: 0.0 for k in epoch_metrics.keys()}
        pbar_val = tqdm(range(num_val_batches), desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Val]")
        for batch_idx in pbar_val:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(val_indices))
            batch_indices = val_indices[start_idx:end_idx]

            batch_a = mx.array(actions[batch_indices])
            batch_s = mx.array(states[batch_indices])
            batch_w = mx.array(contexts[batch_indices])
            batch_m = mx.array(masks[batch_indices])

            metrics = eval_step(
                model=model, batch_a=batch_a, batch_s=batch_s, batch_w=batch_w, batch_m=batch_m,
                kld_weight=kld_weight, distance_type=cfg.model.distance_type, huber_delta=cfg.model.huber_delta,
                w_vae=cfg.training.w_vae, w_action=cfg.training.w_action,
                w_transition=cfg.training.w_transition, w_rollout=cfg.training.w_rollout,
            )
            for k, v in metrics.items():
                val_metrics[k] += v.item()
            
            pbar_val.set_postfix({"val_loss": f"{metrics['total_loss'].item():.4f}"})

        avg_val_metrics = {k: v / num_val_batches for k, v in val_metrics.items()}
        log_str_val = (
            f"Epoch {epoch+1} [Val]   - "
            f"Total Loss: {avg_val_metrics['total_loss']:.4f} | "
            f"VAE: {avg_val_metrics['vae_loss']:.4f} | "
            f"Recon_A: {avg_val_metrics['recon_loss_action']:.4f} | "
            f"Recon_S: {avg_val_metrics['recon_loss_state']:.4f} | "
            f"KL: {avg_val_metrics['kl_loss']:.4f} | "
            f"Action: {avg_val_metrics['action_loss']:.4f} | "
            f"Transition: {avg_val_metrics['transition_loss']:.4f} | "
            f"Rollout: {avg_val_metrics['rollout_loss']:.4f}"
        )
        logger.info(log_str_val)
        
        # W&B 에포크별 요약 로깅
        if wandb_run is not None:
            epoch_log_data = {"epoch": epoch + 1}
            # Train 메트릭
            for k, v in avg_train_metrics.items():
                epoch_log_data[f"train_epoch/{k}"] = v
            # Val 메트릭
            for k, v in avg_val_metrics.items():
                epoch_log_data[f"val_epoch/{k}"] = v
            wandb_run.log(epoch_log_data)
        
        # 조기 종료 확인
        val_total_loss = avg_val_metrics['total_loss']
        should_stop, improved = early_stopper.step(val_total_loss, epoch)
        
        if improved:
            logger.info(f"✅ Validation loss improved to {val_total_loss:.4f}")
        else:
            logger.info(
                f"⚠️  No validation improvement | "
                f"{early_stopper.get_info()}"
            )
        
        if should_stop:
            logger.info(f"\n🛑 Early stopping triggered at Epoch {epoch + 1}")
            logger.info(f"   {early_stopper.get_info()}")
            break
        
        # 체크포인트 저장 (매 save_interval 에포크마다)
        if (epoch + 1) % cfg.training.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"model_epoch{epoch+1:03d}.npz"
            # MLX 모델 저장 (간단한 방식)
            import pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(mx.tree_flatten(model.parameters()), f)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # 학습 완료 로그
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    
    if wandb_run is not None:
        logger.info(f"W&B Run: {wandb_run.get_url()}")
        wandb_run.finish()
    
    logger.info(f"Logs saved to: {experiment_dir / 'train.log'}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
