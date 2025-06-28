"""
@author : Hyuk Jun Yoo
@when : 2023-12-12
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import math
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        """
        Purpose
        ----------
        compute scale dot product attention

        Query : given sentence that we focused on (decoder)
        Key : every sentence to check relationship with Qeury(encoder)
        Value : every sentence same with Key (encoder)

        Return
        ----------
        Self-Attention value, Attention score
        """
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        # batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        # k_t = k.transpose(2, 3)  # transpose

        # [batch_size, head, length, d_tensor]
        length, d_tensor, head = k.size()
        # print("k", k)
        # print("length, d_tensor, head", length, d_tensor, head)

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(1, 2)  # transpose

        # score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        score = (q @ k_t) / math.sqrt(length)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        # print("v.shape",v.shape)
        v = score @ v
        # print("score.shape",score.shape)
        # print("v.shape",v.shape)
        return v, score # Self-Attention value, Attention score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, att_n_head):
        """
        Purpose
        ----------
        constructor of attention multi-head attention layer class (include ScaleDotProductAttention)

        Parameters
        ----------
        - d_model : int
            - hidden size of (multi-head attention and PositionwiseFeedForward, 2개의 hidden layer가 다름.)
            - multi-head attention의 hidden size는 d_model을 사용, PositionwiseFeedForward의 hidden size는 ffn_hidden을 사용
        - att_n_head : int
            - the number of head in multi-head attention
            - 시약 종류의 갯수를 파악할 만큼 넣어주면 됨.
            - ex) metal precursor, metal seed, surfactant, reductant ...
        """
        super(MultiHeadAttention, self).__init__()
        self.att_n_head = int(att_n_head)
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        # batch_size, length, d_model = tensor.size()

        # d_tensor = d_model // self.att_n_head
        # out = tensor.view(batch_size, length, self.att_n_head, d_tensor).transpose(1, 2)
        length, d_model = tensor.size() # 6,12

        d_tensor = d_model // self.att_n_head 
        # out.shape --> length, d_tensor, self.att_n_head
        # 6,4,3 --> 6,3,4
        # it is similar with group convolution (split by number of heads)
        # print("previous", tensor)
        out = tensor.view(-1, self.att_n_head, d_tensor).transpose(0, 1)
        # print("after", out)
        # print(out.shape)

        return out

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        # batch_size, head, length, d_tensor = tensor.size()
        # d_model = head * d_tensor

        # out = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        head, length, d_tensor, = tensor.size()
        # print("length, d_tensor, head", length, d_tensor, head)
        d_model = head * d_tensor
        # print("concat, tensor",tensor)
        out = tensor.transpose(0, 1).contiguous().view(length, d_model)
        # print("concat, tensor",out)
        return out
    