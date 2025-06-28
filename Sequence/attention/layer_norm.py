"""
@author : Hyuk Jun Yoo
@when : 2023-12-12
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        """
        Purpose
        ----------
        constructor of LayerNorm class

        Parameters
        ----------
        - d_model : int
            - hidden size of (multi-head attention and PositionwiseFeedForward, 2개의 hidden layer가 다름.)
            - multi-head attention의 hidden size는 d_model을 사용, PositionwiseFeedForward의 hidden size는 ffn_hidden을 사용
        - eps=1e-12 : float
            - block the zero values in denominator
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out