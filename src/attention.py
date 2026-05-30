# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,       # 입력/출력 벡터 크기 (예: 512)
        n_heads: int,       # attention head 개수, d_model을 나눠 병렬 처리
        drop_rate: float = 0.1,  # dropout 확률 (과적합 방지)
        qkv_bias: bool = False,  # Q/K/V projection에 bias 항 추가 여부
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # 입력을 Q/K/V용 표현으로 바꾸는 projection입니다.
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        # 여러 head의 출력을 다시 하나의 d_model 표현으로 섞습니다.
        self.output_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        batch_size, seq_len, _ = x.shape

        # 한 번의 projection으로 Q, K, V를 이어붙인 텐서를 만듭니다.
        qkv = self.qkv_projection(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 마지막 차원 d_model을 (n_heads, head_dim)으로 나눕니다.
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # head별로 attention을 계산할 수 있도록 축 순서를 바꿉니다.
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 각 head에서 토큰 간 유사도 score를 계산합니다.
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal_mask:
            # 미래 토큰을 보지 못하도록 상삼각 영역을 mask 처리합니다.
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # score를 확률 분포로 바꾼 뒤 dropout을 적용합니다.
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attention weight로 V를 가중합한 결과는 (B, n_heads, T, head_dim)입니다.
        context_vec = (attn_weights @ v).transpose(1, 2)

        # head 축과 head_dim 축을 다시 합쳐 (B, T, d_model)로 복원합니다.
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.d_model)
        context_vec = self.output_projection(context_vec)
        context_vec = self.dropout(context_vec)

        if return_attention_weights:
            return context_vec, attn_weights
        return context_vec
