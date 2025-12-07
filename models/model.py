"""
vrae.py: Variational Recurrent Autoencoder for Fitness Tracker Data

상태(s), 행동(a), 컨텍스트(w), 마스크(m)를 입력받아:
1. VAE: 행동과 상태의 확률분포 학습
2. Policy: 상태와 컨텍스트에서 행동 예측
3. Transition: 상태와 행동에서 다음 상태 예측

구조:
  VAE Encoder → z_a, z_b, z_c (3개 잠재변수)
  ↓ (샘플링)
  VAE Decoder → 행동/상태 재구성
  ↓
  Policy Network → 행동 예측 (z_b 조건)
  Transition Network → 다음 상태 예측 (z_c 조건)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 1. 거리 메트릭 유틸리티
# ============================================================================

class DistanceMetric:
    """다양한 거리 메트릭 구현"""
    
    @staticmethod
    def compute(
        y_pred: mx.array,
        y_true: mx.array,
        metric_type: str = "huber",
        mask: mx.array = None,
        **kwargs
    ) -> mx.array:
        """
        거리 메트릭 계산
        
        Args:
            y_pred: 예측값
            y_true: 실제값
            metric_type: {'rmse', 'mae', 'mape', 'huber'}
            mask: 손실을 계산할 요소에 대한 마스크 (1=유효, 0=무시)
            **kwargs: 메트릭별 추가 파라미터
        
        Returns:
            손실값 (scalar)
        """
        if metric_type == "rmse":
            loss = (y_pred - y_true) ** 2
        elif metric_type == "mae":
            loss = mx.abs(y_pred - y_true)
        elif metric_type == "mape":
            epsilon = kwargs.get("epsilon", 1e-8)
            loss = mx.abs(y_pred - y_true) / (mx.abs(y_true) + epsilon)
        elif metric_type == "huber":
            delta = kwargs.get("delta", 1.0)
            diff = y_pred - y_true
            is_small = mx.abs(diff) <= delta
            loss = mx.where(
                is_small,
                0.5 * diff ** 2,
                delta * (mx.abs(diff) - 0.5 * delta)
            )
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        # 마스크 적용
        if mask is not None:
            # loss와 mask의 차원을 맞추기 위해 브로드캐스팅
            if loss.ndim > mask.ndim:
                mask = mx.expand_dims(mask, axis=-1)
            
            masked_loss = loss * mask
            # 유효한 요소의 개수로만 나누어 평균 계산
            valid_count = mx.sum(mask)
            return mx.sum(masked_loss) / (valid_count + 1e-8)
        else:
            return mx.mean(loss)


# ============================================================================
# 2. 인코더/디코더 컴포넌트
# ============================================================================

class GRUEncoder(nn.Module):
    """양방향 GRU 인코더"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru_forward = nn.GRU(input_size, hidden_size)
        self.gru_backward = nn.GRU(input_size, hidden_size)
        self.hidden_size = hidden_size
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (T, input_size)
        
        Returns:
            h: (T, hidden_size * 2)
        """
        # Forward
        h_forward = self.gru_forward(x)[0]  # (T, hidden_size)
        
        # Backward (인덱싱으로 역순 구현)
        x_reversed = x[::-1]
        h_backward = self.gru_backward(x_reversed)[0]  # (T, hidden_size)
        h_backward = h_backward[::-1]
        
        # Concatenate
        h = mx.concatenate([h_forward, h_backward], axis=-1)  # (T, hidden_size*2)
        return h


class GRUDecoder(nn.Module):
    """양방향 GRU 디코더"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru_forward = nn.GRU(input_size, hidden_size)
        self.gru_backward = nn.GRU(input_size, hidden_size)
        self.hidden_size = hidden_size
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, input_size) 또는 (T, input_size)
        
        Returns:
            h: (B, T, hidden_size * 2) 또는 (T, hidden_size * 2)
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = mx.expand_dims(x, axis=0)

        # Forward
        h_forward = self.gru_forward(x)  # (B, T, hidden_size)
        
        # Backward (인덱싱으로 역순 구현)
        x_reversed = x[:, ::-1, :]
        h_backward = self.gru_backward(x_reversed)  # (B, T, hidden_size)
        h_backward = h_backward[:, ::-1, :]
        
        # Concatenate
        h = mx.concatenate([h_forward, h_backward], axis=-1)  # (T, hidden_size*2)
        if squeeze:
            h = mx.squeeze(h, axis=0)
        return h


class MaskedAttentionPool(nn.Module):
    """마스크 기반 어텐션 풀링"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, 1)
    
    def __call__(self, h: mx.array, mask: mx.array) -> mx.array:
        """
        Args:
            h: (T, hidden_size*2)
            mask: (T, 1)
        
        Returns:
            pooled: (hidden_size*2,)
        """
        # 어텐션 가중치
        weights = self.query(h)  # (T, 1)
        
        # 마스크 적용 (결측값은 -inf로 마스킹)
        weights = mx.where(
            mask == 0,  # 유효한 시점
            weights,
            mx.array(-1e9)
        )
        
        # Softmax
        weights = nn.softmax(weights, axis=0)  # (T, 1)
        
        # 가중평균
        pooled = mx.sum(h * weights, axis=0)  # (hidden_size*2,)
        return pooled

    def forward(self, h: mx.array, mask: mx.array) -> mx.array:
        """
        배치 처리를 위한 forward 메서드
        Args:
            h: (B, T, hidden_size*2)
            mask: (B, T, 1)
        Returns:
            pooled: (B, hidden_size*2)
        """
        # vmap을 사용하여 배치 차원에 대해 __call__을 반복 적용
        # (B, hidden_size*2)
        return mx.vmap(self.__call__)(h, mask)


# ============================================================================
# 3. 정책 네트워크
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    정책 네트워크: π(a_t | s_t, w_t; z_b)
    
    입력: 상태(s), 컨텍스트(w), 잠재변수(z_b)
    출력: 행동(a) 예측
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 7,
        context_dim: int = 1,
        context_embedding_dim: int = 4,
        latent_dim_b: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.latent_dim_b = latent_dim_b
        
        # 컨텍스트 임베딩 (범주형)
        self.context_embed = nn.Embedding(5, context_embedding_dim)  # weather: 5 classes
        
        # MLP
        input_dim = state_dim + context_embedding_dim + latent_dim_b
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        
        # 출력 (행동)
        self.fc_action = nn.Linear(64, action_dim)
    
    def __call__(self, s: mx.array, w: mx.array, z_b: mx.array) -> mx.array:
        """
        Args:
            s: (batch, state_dim) 또는 (state_dim,) [현재 상태]
            w: (batch, context_dim) 또는 (context_dim,) [컨텍스트]
            z_b: (batch, latent_dim_b) 또는 (latent_dim_b,) [정책 스타일]
        
        Returns:
            a: (batch, action_dim) 또는 (action_dim,) [예측 행동]
        """
        # z_b를 시간 축으로 브로드캐스팅
        # s: (B, T, D_s), z_b: (B, 1, D_z) -> (B, T, D_z)
        if s.ndim == 3 and z_b.ndim == 3:
            B, T, _ = s.shape
            z_b = mx.broadcast_to(z_b, (B, T, z_b.shape[-1]))
        
        # 컨텍스트 임베딩
        w = w.astype(mx.int32)
        w_emb = self.context_embed(w)  # (B, T, 1, D_emb) or (B, 1, D_emb)
        if w_emb.ndim > 2:
             w_emb = mx.squeeze(w_emb, axis=-2) # (B, T, D_emb) or (B, D_emb)
        
        # 연결
        x = mx.concatenate([s, w_emb, z_b], axis=-1)  # (batch, input_dim)
        
        # MLP
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = nn.relu(self.fc3(x))
        
        # 출력
        a = self.fc_action(x)  # (batch, action_dim)
        
        return a


# ============================================================================
# 4. 천이 네트워크
# ============================================================================

class TransitionNetwork(nn.Module):
    """
    천이 네트워크: τ(s_{t+1} | s_t, a_t, w_t; z_c)
    
    입력: 상태(s), 행동(a), 컨텍스트(w), 잠재변수(z_c)
    출력: 다음 상태(s_{t+1}) 예측
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 7,
        context_dim: int = 1,
        action_embedding_dim: int = 8,
        context_embedding_dim: int = 4,
        latent_dim_c: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.latent_dim_c = latent_dim_c
        
        # 행동 임베딩 (범주형: workout_type, location)
        self.action_embed = nn.Embedding(8, 4)  # workout_type: 8 classes
        self.location_embed = nn.Embedding(6, 4)  # location: 6 classes
        
        # 컨텍스트 임베딩 (범주형)
        self.context_embed = nn.Embedding(5, context_embedding_dim)  # weather: 5 classes
        
        # MLP
        # state(2) + action_continuous(5) + action_embed(4) + location_embed(4) + context_embed(4) + z_c(32)
        input_dim = 2 + 5 + 4 + 4 + 4 + latent_dim_c
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        
        # 출력 (다음 상태)
        self.fc_state = nn.Linear(64, state_dim)
    
    def __call__(
        self,
        s: mx.array,
        a: mx.array,
        w: mx.array,
        z_c: mx.array
    ) -> mx.array:
        """
        Args:
            s: (batch, state_dim) 또는 (state_dim,) [현재 상태]
            a: (batch, action_dim) 또는 (action_dim,) [현재 행동]
            w: (batch, context_dim) 또는 (context_dim,) [컨텍스트]
            z_c: (batch, latent_dim_c) 또는 (latent_dim_c,) [천이 패턴]
        
        Returns:
            s_next: (batch, state_dim) 또는 (state_dim,) [다음 상태]
        """
        # z_c를 시간 축으로 브로드캐스팅
        # s: (B, T, D_s), z_c: (B, 1, D_z) -> (B, T, D_z)
        if s.ndim == 3 and z_c.ndim == 3:
            B, T, _ = s.shape
            z_c = mx.broadcast_to(z_c, (B, T, z_c.shape[-1]))
        
        # 행동 분리 (연속: 5개, 범주: 2개)
        a_continuous = a[..., :5]
        a_workout = a[..., 5:6].astype(mx.int32)
        a_location = a[..., 6:7].astype(mx.int32)
        
        # 임베딩
        a_workout_emb = self.action_embed(a_workout)  # (B, T, 1, 4)
        a_workout_emb = mx.squeeze(a_workout_emb, axis=-2) # (B, T, 4)
        
        a_location_emb = self.location_embed(a_location)  # (B, T, 1, 4)
        a_location_emb = mx.squeeze(a_location_emb, axis=-2) # (B, T, 4)
        
        w = w.astype(mx.int32)
        w_emb = self.context_embed(w)  # (batch, 1, 4)
        
        for _ in range(w_emb.ndim - a_continuous.ndim):
            w_emb = mx.squeeze(w_emb, axis=-2)
        
        # 연결
        x = mx.concatenate(
            [s, a_continuous, a_workout_emb, a_location_emb, w_emb, z_c],
            axis=-1
        )
        
        # MLP
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = nn.relu(self.fc3(x))
        
        # 출력
        s_next = self.fc_state(x)  # (batch, state_dim)
        
        return s_next


# ============================================================================
# 5. 메인 VRAE 모델
# ============================================================================

class VRAE(nn.Module):
    """
    Variational Recurrent Autoencoder for Fitness Tracker Data
    
    구성:
    - VAE Encoder: [a, s, w, m] → z_a, z_b, z_c (μ, σ)
    - VAE Decoder: z, w → â, ŝ (재구성)
    - Policy Network: [s, w, z_b] → a (행동 예측)
    - Transition Network: [s, a, w, z_c] → s_next (상태 천이)
    """
    
    def __init__(
        self,
        # 데이터 차원
        action_dim: int = 7,
        state_dim: int = 2,
        context_dim: int = 1,
        
        # 임베딩 차원
        action_embedding_dim: int = 8,
        context_embedding_dim: int = 4,
        
        # 잠재 차원
        latent_dim_a: int = 16,
        latent_dim_b: int = 32,
        latent_dim_c: int = 32,
        
        # RNN 차원
        rnn_hidden_dim: int = 256,
        
        # MLP 차원
        mlp_hidden_dim: int = 128,
        
        # 거리 메트릭
        distance_type: str = "huber",
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
        super().__init__()
        
        # 하이퍼파라미터 저장
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.context_dim = context_dim
        
        self.latent_dim_a = latent_dim_a
        self.latent_dim_b = latent_dim_b
        self.latent_dim_c = latent_dim_c
        
        self.rnn_hidden_dim = rnn_hidden_dim
        self.distance_type = distance_type
        self.huber_delta = huber_delta
        
        self.kld_weight = kld_weight
        self.action_weight = action_weight
        self.rollout_weight = rollout_weight
        
        self.k_a = k_a
        self.k_b = k_b
        self.k_c = k_c
        
        # VAE 컴포넌트
        # 인코더 입력: a_continuous(5) + a_workout_emb(4) + a_location_emb(4) + s(2) + w_emb(4) + m(1) = 20
        encoder_input_dim = 5 + 4 + 4 + state_dim + 4 + 1  # 20
        
        self.action_embed = nn.Embedding(8, 4)  # workout_type
        self.location_embed = nn.Embedding(6, 4)  # location
        self.context_embed = nn.Embedding(5, 4)  # weather
        
        self.encoder = GRUEncoder(encoder_input_dim, rnn_hidden_dim)
        self.attention_pool = MaskedAttentionPool(rnn_hidden_dim * 2)
        
        # 잠재변수 출력층 (log_var 방식 사용)
        self.fc_mu_a = nn.Linear(rnn_hidden_dim * 2, latent_dim_a)
        self.fc_logvar_a = nn.Linear(rnn_hidden_dim * 2, latent_dim_a)
        
        self.fc_mu_b = nn.Linear(rnn_hidden_dim * 2, latent_dim_b)
        self.fc_logvar_b = nn.Linear(rnn_hidden_dim * 2, latent_dim_b)
        
        self.fc_mu_c = nn.Linear(rnn_hidden_dim * 2, latent_dim_c)
        self.fc_logvar_c = nn.Linear(rnn_hidden_dim * 2, latent_dim_c)
        
        # 디코더
        decoder_input_dim = latent_dim_a + latent_dim_b + latent_dim_c + context_dim + 4
        self.decoder = GRUDecoder(decoder_input_dim, rnn_hidden_dim)
        
        self.fc_decoder_a = nn.Linear(rnn_hidden_dim * 2, action_dim)
        self.fc_decoder_s = nn.Linear(rnn_hidden_dim * 2, state_dim)
        
        # Policy & Transition Networks
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            context_dim=context_dim,
            context_embedding_dim=context_embedding_dim,
            latent_dim_b=latent_dim_b,
            hidden_dim=mlp_hidden_dim,
        )
        
        self.transition = TransitionNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            context_dim=context_dim,
            action_embedding_dim=action_embedding_dim,
            context_embedding_dim=context_embedding_dim,
            latent_dim_c=latent_dim_c,
            hidden_dim=mlp_hidden_dim,
        )
    
    def encode(
        self,
        a: mx.array,
        s: mx.array,
        w: mx.array,
        m: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
        인코더: 행동/상태/컨텍스트 → 잠재변수 분포
        
        Args:
            a: (T, action_dim) 행동
            s: (T, state_dim) 상태
            w: (B, T, context_dim) 컨텍스트
            m: (B, T, 1) 마스크
        
        Returns:
            (mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c) 각 (B, latent_dim)
        """
        B, T, _ = a.shape
        
        # 임베딩
        a_continuous = a[..., :5]  # (B, T, 5)
        a_workout = a[..., 5:6].astype(mx.int32)
        a_location = a[..., 6:7].astype(mx.int32)
        
        a_workout_emb = self.action_embed(a_workout)  # (B, T, 1, 4)
        a_workout_emb = mx.squeeze(a_workout_emb, axis=2) # (B, T, 4)
        
        a_location_emb = self.location_embed(a_location)  # (B, T, 1, 4)
        a_location_emb = mx.squeeze(a_location_emb, axis=2) # (B, T, 4)
        
        w_int = w.astype(mx.int32)
        w_emb = self.context_embed(w_int)  # (B, T, 1, 4)
        w_emb = mx.squeeze(w_emb, axis=2) # (B, T, 4)
        
        # 연결
        x = mx.concatenate(
            [a_continuous, a_workout_emb, a_location_emb, s, w_emb, m],
            axis=-1
        )  # (B, T, encoder_input_dim)
        
        # 인코더
        # vmap을 사용하여 배치 차원에 대해 encoder.__call__을 반복 적용
        h = mx.vmap(self.encoder)(x) # (B, T, rnn_hidden_dim * 2)
        
        # 어텐션 풀링
        h_agg = self.attention_pool.forward(h, m)  # (B, rnn_hidden_dim * 2)
        
        # 잠재변수 분포 (log_var 방식으로 안정성 향상)
        mu_a = self.fc_mu_a(h_agg)  # (B, latent_dim_a)
        logvar_a = self.fc_logvar_a(h_agg)  # (B, latent_dim_a)
        logvar_a = mx.clip(logvar_a, a_min=-10.0, a_max=10.0)  # 범위 제한
        sigma_a = mx.exp(0.5 * logvar_a)  # (B, latent_dim_a)
        
        mu_b = self.fc_mu_b(h_agg)  # (B, latent_dim_b)
        logvar_b = self.fc_logvar_b(h_agg)  # (B, latent_dim_b)
        logvar_b = mx.clip(logvar_b, a_min=-10.0, a_max=10.0)  # 범위 제한
        sigma_b = mx.exp(0.5 * logvar_b)  # (B, latent_dim_b)
        
        mu_c = self.fc_mu_c(h_agg)  # (B, latent_dim_c)
        logvar_c = self.fc_logvar_c(h_agg)  # (B, latent_dim_c)
        logvar_c = mx.clip(logvar_c, a_min=-10.0, a_max=10.0)  # 범위 제한
        sigma_c = mx.exp(0.5 * logvar_c)  # (B, latent_dim_c)
        
        return mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c
    
    def sample_latents(
        self,
        mu_a: mx.array,
        sigma_a: mx.array,
        mu_b: mx.array,
        sigma_b: mx.array,
        mu_c: mx.array,
        sigma_c: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        샘플링: 각 잠재변수에서 k개씩 샘플 생성
        
        Args:
            mu_*, sigma_*: 평균과 표준편차
        
        Returns:
            (z_a, z_b, z_c)  각각 (k, latent_dim)
        """
        # 추론 시 (배치 차원이 없을 때)를 위한 처리
        if mu_a.ndim == 1:
            mu_a = mx.expand_dims(mu_a, 0)
            sigma_a = mx.expand_dims(sigma_a, 0)
            mu_b = mx.expand_dims(mu_b, 0)
            sigma_b = mx.expand_dims(sigma_b, 0)
            mu_c = mx.expand_dims(mu_c, 0)
            sigma_c = mx.expand_dims(sigma_c, 0)
        B = mu_a.shape[0]

        # 샘플링
        eps_a = mx.random.normal((B, self.k_a, self.latent_dim_a))
        # mu_a (B, dim) -> (B, 1, dim), sigma_a (B, dim) -> (B, 1, dim)
        z_a = mx.expand_dims(mu_a, 1) + mx.expand_dims(sigma_a, 1) * eps_a  # (B, k_a, latent_dim_a)
        
        eps_b = mx.random.normal((B, self.k_b, self.latent_dim_b))
        z_b = mx.expand_dims(mu_b, 1) + mx.expand_dims(sigma_b, 1) * eps_b  # (B, k_b, latent_dim_b)
        
        eps_c = mx.random.normal((B, self.k_c, self.latent_dim_c))
        z_c = mx.expand_dims(mu_c, 1) + mx.expand_dims(sigma_c, 1) * eps_c  # (B, k_c, latent_dim_c)
        
        return z_a, z_b, z_c
    
    def decode(
        self,
        z_a: mx.array,
        z_b: mx.array,
        z_c: mx.array,
        w: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        디코더: 잠재변수 → 행동/상태 재구성
        
        Args:
            z_a: (B, k_a, latent_dim_a)
            z_b: (B, k_b, latent_dim_b)
            z_c: (B, k_c, latent_dim_c)
            w: (B, T, context_dim)
        
        Returns:
            (a_recon, s_recon) 각각 (B, T, action_dim), (B, T, state_dim)
        """
        B, T, _ = w.shape
        
        # 첫 샘플 사용
        z_a_sample = z_a[:, 0]  # (B, latent_dim_a)
        z_b_sample = z_b[:, 0]  # (B, latent_dim_b)
        z_c_sample = z_c[:, 0]  # (B, latent_dim_c)
        
        # z 확장 (시간 축)
        z = mx.concatenate(
            [
                mx.expand_dims(z_a_sample, 1),
                mx.expand_dims(z_b_sample, 1),
                mx.expand_dims(z_c_sample, 1),
            ],
            axis=-1
        )  # (B, 1, latent_dim_a + latent_dim_b + latent_dim_c)
        z = mx.broadcast_to(z, (B, T, z.shape[-1])) # (B, T, sum_latent_dims)
        
        # 컨텍스트 임베딩
        w_int = w.astype(mx.int32)
        w_emb = self.context_embed(w_int)  # (B, T, 1, 4)
        w_emb = mx.squeeze(w_emb, axis=2)  # (B, T, 4)
        
        # 디코더 입력
        x = mx.concatenate([z, w, w_emb], axis=-1)  # (B, T, decoder_input_dim)
        
        # 디코더
        h = self.decoder(x) # (B, T, rnn_hidden_dim * 2)
        
        # 재구성
        # fc 레이어는 추가 차원을 자동으로 처리
        a_recon = self.fc_decoder_a(h)  # (B, T, action_dim)
        s_recon = self.fc_decoder_s(h)  # (B, T, state_dim)
        
        return a_recon, s_recon
    
    def rollout(
        self,
        s0: mx.array,
        w: mx.array,
        z_a: mx.array,
        z_b: mx.array,
        z_c: mx.array,
    ) -> mx.array:
        """
        롤아웃: 초기 상태에서 시작하여 125개 궤적 생성
        
        z_a를 활용하여 각 조합별로 다른 초기 특성을 반영
        
        Args:
            s0: (B, state_dim) 초기 상태
            w: (B, T, context_dim) 컨텍스트
            z_a: (B, k_a, latent_dim_a) - 초기 상태 다양성
            z_b: (B, k_b, latent_dim_b) - 행동 스타일
            z_c: (B, k_c, latent_dim_c) - 신체 반응 패턴
        
        Returns:
            trajectories: (B, k_a*k_b*k_c, T, action_dim) 생성 행동 궤적
        """
        # vmap을 사용하여 배치 차원에 대해 _rollout_single을 반복 적용
        return mx.vmap(self._rollout_single, in_axes=(0, 0, 0, 0, 0))(s0, w, z_a, z_b, z_c)

    def _rollout_single(self, s0, w, z_a, z_b, z_c):
        """단일 데이터 포인트에 대한 롤아웃"""
        num_trajectories = self.k_a * self.k_b * self.k_c

        # (k_a, k_b, k_c, dim) -> (k_a*k_b*k_c, dim)
        z_a_flat = mx.repeat(z_a, self.k_b * self.k_c, axis=0)
        z_b_flat = mx.repeat(mx.tile(z_b, (self.k_a, 1)), self.k_c, axis=0)
        z_c_flat = mx.tile(z_c, (self.k_a * self.k_b, 1))
        
        # 초기 상태 s0를 z_a의 영향을 받아 확장
        s0_expanded = mx.broadcast_to(s0, (num_trajectories, self.state_dim))
        s_initial = s0_expanded + mx.mean(z_a_flat, axis=-1, keepdims=True) * 0.1

        # for 루프를 사용하여 롤아웃 실행
        s_t = s_initial
        trajectories_list = []
        
        # w: (T, context_dim)
        T = w.shape[0]
        for t in range(T):
            w_t = w[t]
            w_t_batch = mx.broadcast_to(w_t, (num_trajectories, self.context_dim))

            # 정책에서 행동 생성
            a_t = self.policy(s_t, w_t_batch, z_b_flat)

            # 천이 네트워크에서 다음 상태 생성
            s_t = self.transition(s_t, a_t, w_t_batch, z_c_flat)
            trajectories_list.append(a_t)

        # (T, num_trajectories, action_dim) -> (num_trajectories, T, action_dim)
        trajectories = mx.stack(trajectories_list, axis=1)

        return trajectories
    
    def compute_distance(
        self,
        y_pred: mx.array,
        y_true: mx.array,
        metric_type: str = "huber",
        delta: float = 1.0,
    ) -> mx.array:
        """거리 메트릭 계산 헬퍼"""
        return DistanceMetric.compute(y_pred, y_true, metric_type, delta=delta)
    
    def compute_policy_loss(
        self,
        a: mx.array,
        s: mx.array,
        w: mx.array,
        m: mx.array,
        z_b_sample: mx.array,
        distance_type: str = "huber",
        huber_delta: float = 1.0,
    ) -> mx.array:
        """
        정책 손실 계산: π(a_t | s_t, w_t; z_b)
        
        마스킹 기반으로 유효한 시점만 손실 계산
        """
        a_pred = self.policy(s, w, mx.expand_dims(z_b_sample, 1))
        mask = (m < 0.5).astype(mx.float32)
        return DistanceMetric.compute(a_pred, a, distance_type, mask=mask, delta=huber_delta)

    def _compute_policy_loss_single(self, a, s, w, m, z_b_sample, distance_type, huber_delta):
        """단일 데이터에 대한 정책 손실 계산 (벡터화)"""
        a_pred = self.policy(s, w, mx.broadcast_to(z_b_sample, s.shape[:-1] + z_b_sample.shape))
        # 불리언 마스크를 0/1 마스크로 변환
        mask = (m < 0.5).astype(mx.float32)
        return DistanceMetric.compute(a_pred, a, distance_type, mask=mask, delta=huber_delta)
    
    def compute_transition_loss(
        self,
        s: mx.array,
        a: mx.array,
        w: mx.array,
        m: mx.array,
        z_c_sample: mx.array,
        distance_type: str = "huber",
        huber_delta: float = 1.0,
    ) -> mx.array:
        """
        천이 손실 계산: τ(s_{t+1} | s_t, a_t, w_t; z_c)
        
        마스킹 기반으로 유효한 시점만 손실 계산
        """
        s_t = s[:, :-1]
        s_t_plus_1 = s[:, 1:]
        a_t = a[:, :-1]
        w_t = w[:, :-1]
        m_t = m[:, :-1]

        s_next_pred = self.transition(s_t, a_t, w_t, mx.expand_dims(z_c_sample, 1))
        mask = (m_t < 0.5).astype(mx.float32)
        return DistanceMetric.compute(s_next_pred, s_t_plus_1, distance_type, mask=mask, delta=huber_delta)


    def _compute_transition_loss_single(self, s, a, w, m, z_c_sample, distance_type, huber_delta):
        """단일 데이터에 대한 천이 손실 계산 (벡터화)"""
        s_t = s[:-1]
        s_t_plus_1 = s[1:]
        a_t = a[:-1]
        w_t = w[:-1]
        m_t = m[:-1]
        
        s_next_pred = self.transition(s_t, a_t, w_t, mx.broadcast_to(z_c_sample, s_t.shape[:-1] + z_c_sample.shape))
        mask = (m_t < 0.5).astype(mx.float32)
        return DistanceMetric.compute(s_next_pred, s_t_plus_1, distance_type, mask=mask, delta=huber_delta)
    
    def compute_rollout_loss(
        self,
        a_true: mx.array,
        s0: mx.array,
        w: mx.array,
        m: mx.array,
        z_a: mx.array,
        z_b: mx.array,
        z_c: mx.array,
        distance_type: str = "huber",
        huber_delta: float = 1.0,
    ) -> mx.array:
        """
        롤아웃 손실 계산
        
        125개 생성 궤적과 실제 궤적의 거리
        마스킹 기반으로 유효한 시점만 손실 계산
        """
        # 롤아웃: 125개 궤적 생성
        trajectories = self.rollout(s0, w, z_a, z_b, z_c)  # (125, T, action_dim)
        
        # a_true: (B, T, D) -> (B, 1, T, D), trajectories: (B, N, T, D)
        a_true_expanded = mx.expand_dims(a_true, 1)
        # mask: (B, T, 1) -> (B, 1, T, 1)
        mask_float = mx.expand_dims((m < 0.5).astype(mx.float32), 1)
        
        # DistanceMetric.compute가 y_pred와 y_true의 브로드캐스팅 후 형태를 기준으로 마스크를 처리하므로,
        # 마스크의 마지막 차원을 action_dim과 동일하게 확장해줍니다.
        mask_expanded = mx.broadcast_to(mask_float, trajectories.shape)
        return DistanceMetric.compute(trajectories, a_true_expanded, distance_type, mask=mask_expanded, delta=huber_delta)

    def _compute_rollout_loss_single(self, trajectories, a_true, m, distance_type, huber_delta):
        """단일 데이터에 대한 롤아웃 손실 계산 (벡터화)"""
        # a_true: (T, D) -> (1, T, D), trajectories: (N, T, D)
        a_true_expanded = mx.expand_dims(a_true, 0)
        mask = (m < 0.5).astype(mx.float32) # (T, 1)
        return DistanceMetric.compute(trajectories, a_true_expanded, distance_type, mask=mask, delta=huber_delta)
    
    def loss_function(
        self,
        a: mx.array,
        s: mx.array,
        w: mx.array,
        m: mx.array,
        mu_a: mx.array,
        sigma_a: mx.array,
        mu_b: mx.array,
        sigma_b: mx.array,
        mu_c: mx.array,
        sigma_c: mx.array,
        a_recon: mx.array,
        s_recon: mx.array,
        kld_weight: float = 1.0,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        VAE 손실함수 계산
        
        Args:
            a: (T, action_dim) 실제 행동
            s: (T, state_dim) 실제 상태
            w: (T, context_dim) 컨텍스트
            m: (T, 1) 마스크 (0=유효, 1=결측)
            mu_*, sigma_*: 잠재변수 분포
            a_recon: (T, action_dim) 재구성 행동
            s_recon: (T, state_dim) 재구성 상태
            kld_weight: KL 손실 가중치
        
        Returns:
            (total_loss, metrics_dict)
        """
        # 1. KL divergence (안정화된 계산)
        eps = 1e-8
        kl_a = -0.5 * mx.sum(1 + 2 * mx.log(sigma_a + eps) - mu_a ** 2 - sigma_a ** 2, axis=-1)
        kl_b = -0.5 * mx.sum(1 + 2 * mx.log(sigma_b + eps) - mu_b ** 2 - sigma_b ** 2, axis=-1)
        kl_c = -0.5 * mx.sum(1 + 2 * mx.log(sigma_c + eps) - mu_c ** 2 - sigma_c ** 2, axis=-1)
        kl_loss = kl_a + kl_b + kl_c
        
        # 2. Reconstruction loss (마스킹 기반)
        mask = (m < 0.5).astype(mx.float32)
        recon_loss_a = DistanceMetric.compute(a_recon, a, self.distance_type, mask=mask, delta=self.huber_delta)
        recon_loss_s = DistanceMetric.compute(s_recon, s, self.distance_type, mask=mask, delta=self.huber_delta)
        
        # NaN 체크 및 안정화
        recon_loss_a = mx.where(mx.isnan(recon_loss_a), mx.array(0.0), recon_loss_a)
        recon_loss_s = mx.where(mx.isnan(recon_loss_s), mx.array(0.0), recon_loss_s)
        recon_loss = recon_loss_a + recon_loss_s
        
        # 3. VAE 총 손실
        # vae_loss는 (B,) 형태이므로 평균을 취함
        vae_loss = mx.mean(recon_loss + kld_weight * kl_loss)
        
        # NaN 최종 체크
        vae_loss = mx.where(mx.isnan(vae_loss), mx.array(0.0), vae_loss)
        
        # 메트릭
        metrics = {
            "total_loss": vae_loss,
            "recon_loss_action": recon_loss_a,
            "recon_loss_state": recon_loss_s,
            "kl_loss": kl_loss,
        }
        
        return vae_loss, metrics

    def _recon_loss_single(self, a_recon, a, s_recon, s, m):
        """단일 데이터에 대한 재구성 손실 계산 (벡터화)"""
        mask = (m < 0.5).astype(mx.float32) # (T, 1)
        loss_a = DistanceMetric.compute(a_recon, a, self.distance_type, mask=mask, delta=self.huber_delta)
        loss_s = DistanceMetric.compute(s_recon, s, self.distance_type, mask=mask, delta=self.huber_delta)
        return loss_a, loss_s

    def forward(
        self,
        a: mx.array,
        s: mx.array,
        w: mx.array,
        m: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
        전체 forward pass
        
        Args:
            a: (T, action_dim) 행동
            s: (T, state_dim) 상태
            w: (T, context_dim) 컨텍스트
            m: (T, 1) 마스크
        
        Returns:
            (mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c, a_recon, s_recon)
        """
        # 인코더
        mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c = self.encode(a, s, w, m)
        
        # 샘플링
        z_a, z_b, z_c = self.sample_latents(mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c)
        
        # 디코더
        a_recon, s_recon = self.decode(z_a, z_b, z_c, w)
        
        return mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c, a_recon, s_recon
    
    def inference(
        self,
        s0: mx.array,
        w: mx.array,
        mu_a: mx.array,
        sigma_a: mx.array,
        mu_b: mx.array,
        sigma_b: mx.array,
        mu_c: mx.array,
        sigma_c: mx.array,
    ) -> mx.array:
        """
        추론: 초기 상태에서 125개의 행동 궤적 생성
        
        Args:
            s0: (state_dim,) 초기 상태
            w: (T, context_dim) 컨텍스트
            mu_*, sigma_*: 인코더에서 얻은 잠재변수 분포
        
        Returns:
            trajectories: (k_a*k_b*k_c, T, action_dim)
        """
        # 샘플링
        z_a, z_b, z_c = self.sample_latents(mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c)
        
        # vmap을 위해 입력 텐서에 배치 차원(크기 1) 추가
        s0_batch = mx.expand_dims(s0, 0)
        w_batch = mx.expand_dims(w, 0)
        
        # 롤아웃
        # z_a, z_b, z_c는 sample_latents에서 이미 (1, k, dim) 형태임
        trajectories_batch = self.rollout(s0_batch, w_batch, z_a, z_b, z_c)
        
        # 불필요한 배치 차원 제거 후 반환
        return mx.squeeze(trajectories_batch, axis=0)
