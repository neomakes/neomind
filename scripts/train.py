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
import numpy as np
import pandas as pdf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# 모델 임포트
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model import VRAE

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from tqdm import tqdm
import logging

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
) -> Dict[str, mx.array]:
    """
    단일 학습 스텝 (VAE + 정책 + 천이 + 롤아웃 손실)
    
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
        metrics_dict
    """
    def loss_fn(model):
        # 이 함수는 전체 손실과 로깅을 위한 개별 메트릭을 함께 반환합니다.
        # 1. 인코더
        mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c = model.encode(batch_a, batch_s, batch_w, batch_m)

        # 2. 샘플링
        z_a, z_b, z_c = model.sample_latents(mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c)
        
        # 3. 디코더 (재구성)
        a_recon, s_recon = model.decode(z_a, z_b, z_c, batch_w)

        # 4. VAE 손실
        vae_loss, vae_metrics = model.loss_function(
            batch_a, batch_s, batch_w, batch_m,
            mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c,
            a_recon, s_recon,
            kld_weight=kld_weight,
        )

        # 5. 정책 손실: π(a_t | s_t, w_t; z_b)
        action_loss = model.compute_policy_loss(
            batch_a, batch_s, batch_w, batch_m, z_b[:, 0], distance_type, huber_delta
        )

        # 6. 천이 손실: τ(s_{t+1} | s_t, a_t, w_t; z_c)
        transition_loss = model.compute_transition_loss(
            batch_s, batch_a, batch_w, batch_m, z_c[:, 0], distance_type, huber_delta
        )

        # 7. 롤아웃 손실: 125개 생성 궤적과 실제 궤적의 거리
        rollout_loss = model.compute_rollout_loss(
            batch_a, batch_s[:, 0], batch_w, batch_m, z_a, z_b, z_c, distance_type, huber_delta
        )

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
    
    # 1. 손실과 그래디언트 계산
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    (loss, metrics), grads = loss_and_grad_fn(model)
    
    # 2. 옵티마이저 업데이트
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    return metrics

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
    logger.info("=" * 80)
    logger.info("FitnessTracker VRAE Model Training")
    logger.info("=" * 80)
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
        latent_dim_a=cfg.model.latent_action_dim,
        latent_dim_b=cfg.model.latent_behavior_dim,
        latent_dim_c=cfg.model.latent_context_dim,
        rnn_hidden_dim=cfg.model.hidden_dim,
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
    logger.info(f"  - Latent dims: z_a={cfg.model.latent_action_dim}, z_b={cfg.model.latent_behavior_dim}, z_c={cfg.model.latent_context_dim}")
    
    # ========================================================================
    # 3. 옵티마이저 설정
    # ========================================================================
    logger.info("\n[3/5] Setting up optimizer...")
    optimizer = optim.Adam(learning_rate=cfg.training.learning_rate)
    
    logger.info(f"Learning rate: {cfg.training.learning_rate}")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info(f"Epochs: {cfg.training.epochs}")
    
    # ========================================================================
    # 4. W&B 초기화 (선택사항)
    # ========================================================================
    if cfg.training.use_wandb:
        logger.info("\n[4/5] Initializing W&B...")
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info(f"W&B run: {wandb.run.name}")
    else:
        logger.info("\n[4/5] W&B disabled")
    
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
        logger.info(f"Checkpoints will be saved to W&B directory: {checkpoint_dir}")
    else:
        # W&B 미사용 시, 설정 파일의 기본 경로 사용
        checkpoint_dir = Path(cfg.training.checkpoint_dir)
    
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
            metrics = train_step(
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

            # W&B 로깅 (매 스텝)
            if cfg.training.use_wandb and (batch_idx % cfg.training.log_interval == 0):
                step_log_data = {"epoch": epoch + 1, "step": epoch * num_train_batches + batch_idx}
                # wandb에 로그할 때, loss/ 접두사 추가
                for k, v in metrics.items():
                    step_log_data[f"batch_loss/{k}"] = v.item()
                wandb.log(step_log_data)

            pbar.set_postfix({"loss": f"{metrics['total_loss'].item():.4f}", "kld_w": f"{kld_weight:.3f}", "lr": f"{current_lr:.6f}"})
        
        # 에포크 평균 메트릭 계산 및 로깅
        avg_train_metrics = {k: v / num_train_batches for k, v in epoch_metrics.items()}
        
        log_str = f"Epoch {epoch+1} [Train] - Total Loss: {avg_train_metrics['total_loss']:.4f} | VAE: {avg_train_metrics['vae_loss']:.4f}"
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
        log_str_val = f"Epoch {epoch+1} [Val]   - Total Loss: {avg_val_metrics['total_loss']:.4f} | VAE: {avg_val_metrics['vae_loss']:.4f}"
        logger.info(log_str_val)

        # W&B 에포크 로깅
        if cfg.training.use_wandb:
            log_data = {"epoch": epoch + 1, "kld_weight": kld_weight, "learning_rate": current_lr}
            
            # 훈련 손실 로그
            for k, v in avg_train_metrics.items():
                log_data[f"epoch_loss/{k}"] = v
            
            # 검증 손실 로그
            for k, v in avg_val_metrics.items():
                log_data[f"val_loss/{k}"] = v
            
            wandb.log(log_data)
        
        # 모델 체크포인팅
        if (epoch + 1) % cfg.training.save_interval == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}.safetensors"
            
            # MLX 모델 저장 (weight dictionary)
            weights = dict(tree_flatten(model.parameters()))
            
            try:
                import safetensors.mlx
                safetensors.mlx.save_file(weights, str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            except ImportError:
                logger.warning("safetensors not available, skipping checkpoint save")
    
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    
    if cfg.training.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
