"""
@author : Hyuk Jun Yoo
@when : 2023-12-12
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        Purpose
        ----------
        constructor of PositionwiseFeedForward layer class
            multi head attention layer를 거친 vector를 FFNN(Feed Forward NN)을 거치는 과정

        Parameters
        ----------
        - d_model : int
            - hidden size of (multi-head attention and PositionwiseFeedForward, 2개의 hidden layer가 다름.)
            - multi-head attention의 hidden size는 d_model을 사용, PositionwiseFeedForward의 hidden size는 ffn_hidden을 사용
        - ffn_hidden : int
            - hidden size of PositionwiseFeedForward 
        - drop_prob : int
            - the probability of dropout
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x