# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_length = context_length
        # PyTorch의 nn.Module은 내부적으로 __call__()이 구현되어 있어서 객체를 함수처럼 부를 수 있어요.
        # nn.Embedding(num_embeddings, embedding_dim)
        # num_embeddings = 총 몇 개의 ID를 임베딩할 것인가
        # embedding_dim = 각 ID를 몇 차원 벡터로 바꿀 것인가
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.position_embedding = nn.Embedding(self.context_length,
                                               self.emb_dim)
        # 학습 중 drop_rate 비율만큼 값을 랜덤하게 0으로 만들어 과적합을 줄인다.
        # self.dropout = Dropout.__init__(drop_rate)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        batch_size, seq_len = x.shape

        # 실제의미 token_embeds = self.token_embedding.forward(x)
        # x:
        # [[1,   2,   3  ],
        #  [0,   4,   2  ]]

        #             ↓ ID → 벡터로 교체

        # token_embeds:
        # [[[0.4, 0.5, 0.6],   ← ID 1의 벡터
        # [0.7, 0.8, 0.9],   ← ID 2의 벡터
        # [1.0, 1.1, 1.2]],  ← ID 3의 벡터

        # [[0.1, 0.2, 0.3],   ← ID 0의 벡터
        # [1.3, 1.4, 1.5],   ← ID 4의 벡터
        # [0.7, 0.8, 0.9]]]  ← ID 2의 벡터
        token_embeds = self.token_embedding(x)

        # 0 ~ seq_len - 1까지 숫자 만들고, x와 같은 장치(CPU, GPU)에 만들어라
        positions = torch.arange(seq_len, device=x.device)
        position_embeds = self.position_embedding(positions)

        x = token_embeds + position_embeds
        # self.dropout.forward(x)
        x = self.dropout(x)

        return x
