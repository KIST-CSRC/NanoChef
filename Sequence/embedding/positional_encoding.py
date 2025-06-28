"""
@author : Hyuk Jun Yoo
@when : 2023-12-12
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

class PositionalEncoding(nn.Module):

    def __init__(self, reagent_sequence:list, ps_dim:int, device:object, pos_enc_dim=512, pos_enc_len=512, seed_num=None):
        """
        Purpose
        ----------
        constructor of positional encoding class
        
        Parameters
        ----------
        - reagent_sequence: : list
            - include reagent type. ex) ["Ag seed","NaBH4","AgNO3","H2O2","citrate","H2O"]
        - ps_dim : int 
            - dimension of output vector 
            - if ps_dim=6, 
            ```python
                    >>> reagent_sequence=["Ag seed","NaBH4","AgNO3","H2O2","citrate","H2O"]
                    [[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000],
                    [ 0.8415,  0.5403,  0.8219,  0.5697,  0.8020],
                    [ 0.9093, -0.4161,  0.9364, -0.3509,  0.9581],
                    [ 0.1411, -0.9900,  0.2451, -0.9695,  0.3428],
                    [-0.7568, -0.6536, -0.6572, -0.7537, -0.5486],
                    [-0.9589,  0.2837, -0.9939,  0.1107, -0.9982]]
            ```
        - pos_enc_dim : int
            - dimension of model, only even number due to denominator (positional encoding 하나의 차원, Transformer에서는 512)
        - pos_enc_len: int
            - max sequence length, 입력 문장의 길이 (Transformer에서는 512, 512 넘어가는 문장을 넣으면 짤림)
        - seed_num=None : int 
            - seed number of torch
        """
        super(PositionalEncoding, self).__init__()

        if type(seed_num) == int:
            torch.manual_seed(seed_num)

        # input matrix(자연어 처리에선 임베딩 벡터)와 같은 size의 tensor 생성 --> 즉, (pos_enc_len, pos_enc_dim) size
        self.reagent_sequence=reagent_sequence
        self.ps_dim=ps_dim
        self.device=device
        self.encoding = torch.zeros(pos_enc_len, pos_enc_dim).to(self.device)
        self.encoding.requires_grad = False  # we don't need to compute gradient
        self.seed_num=seed_num

        # 위치 indexing용 벡터
        # pos는 pos_enc_len의 index를 의미한다.
        pos = torch.arange(0, pos_enc_len,).to(self.device)
        # 1D : (pos_enc_len, ) size -> 2D : (pos_enc_len, 1) size -> word의 위치를 반영하기 위해
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 (없어도 되긴 함)

        # i는 pos_enc_dim의 index를 의미한다. _2i : (pos_enc_dim, ) size
        # 즉, embedding size가 512일 때, i = [0,512]
        _2i = torch.arange(0, pos_enc_dim, step=2,).float().to(self.device)
        # _2i = torch.arange(0, pos_enc_dim, step=2,).float().to(self.device)
        # 'i' means index of pos_enc_dim (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        # compute positional encoding to consider positional information of words
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / pos_enc_dim))).to(self.device)
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / pos_enc_dim))).to(self.device)

        # 중복을 제거하고 고유한 단어들의 리스트를 만듭니다
        self.unique_words = sorted(list(set(self.reagent_sequence)))

        # 각 단어에 대해 고유한 인덱스를 할당하는 Dictionary를 생성합니다
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

    def forward(self, batch_rgn_sqn):
        x=torch.tensor([]).to(self.device)
        for rgn_sqn in batch_rgn_sqn:
            x=torch.cat((x, torch.Tensor([self.word_to_index[word] for word in rgn_sqn]).to(self.device)))
        # x=torch.tensor([[self.word_to_index[word] for word in rgn_sqn]]).to(self.device)
        indices = x.view(len(batch_rgn_sqn), len(batch_rgn_sqn[0])).long()

        # self.encoding --> [pos_enc_len = 512, pos_enc_dim = 512]
        # [batch_size = 128, ps_dim = 30]
        batch_size, seq_len = indices.size()
        
        # [ps_dim = 30, pos_enc_dim = 512]
        # it will add with tok_emb : [128, 30, 512]
        out=torch.tensor([]).to(self.device)
        for indice in indices:
            encoding_result=self.encoding[:seq_len, :self.ps_dim]
            result=torch.index_select(encoding_result, dim=0, index=indice)
            out=torch.cat((out, result.unsqueeze(0)), dim=0)
        return out

    def visualize_positional_encoding(self, pos_enc):
        plt.figure(figsize=(10, 6))
        plt.imshow(pos_enc, cmap='viridis', aspect='auto')
        plt.xlabel('Dimension')
        plt.ylabel('Position')
        plt.title('Positional Encoding Visualization')
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    word_to_index = {"AgNO3": 0, "H2O2": 1, "Sodium citrate": 2, "H2O": 3, "NaBH4": 4}
    reagent_sequence=["Au seed","AgNO3","H2O2","Sodium citrate","H2O","NaBH4"]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    # device="cpu"
    
    vector=PositionalEncoding(reagent_sequence, ps_dim=6, device=device)

    # ["AgNO3","H2O2","Citrate","H2O","NaBH4"]
    rgn_seq=[
        ["H2O2","Sodium citrate","Au seed","AgNO3","NaBH4","H2O"],
        ["AgNO3","NaBH4","H2O","H2O2","Sodium citrate","Au seed"]]
    result=vector(rgn_seq)
    print(result)

    # import time
    # start_test=time.time()
    # for i in range(10000):
    #     result=vector(rgn_seq)
    # finish_test=time.time()
    # print("CUDA : ",finish_test-start_test)