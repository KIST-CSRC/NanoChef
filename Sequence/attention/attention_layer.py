"""
@author : Hyuk Jun Yoo
@when : 2023-12-12
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import torch
import torch.nn as nn

from Sequence.attention.layer_norm import LayerNorm
from Sequence.attention.multi_head_attention import MultiHeadAttention
from Sequence.attention.position_wise_feed_forward import PositionwiseFeedForward


class AttentionLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, att_n_head, drop_prob):
        """
        Purpose
        ----------
        constructor of attention layer class (include MultiHeadAttention, PositionwiseFeedForward, LayerNorm)
            nn.Embedding 에서는 내가 원하는 길이의 임베딩 벡터를 임의로 만들어주고 학습 과정 동안 적절한 임베딩 벡터로 조정

        Parameters
        ----------
        - d_model : int
            - ***(Caution)*** the hidden size of multi-head attention layer --> d_model
        - ffn_hidden : int
            - ***(Caution)*** the hidden size of PositionwiseFeedForward layer --> ffn_hidden
        - att_n_head : int
            - the number of head in multi-head attention
        - drop_prob : int
            - the probability of dropout
        """
        super(AttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, att_n_head=att_n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        return x