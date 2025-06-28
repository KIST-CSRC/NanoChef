"""
@author : Hyuk Jun Yoo
@when : 2023-12-12
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import sys
sys.path.append("./")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
warnings.filterwarnings("ignore")

import torch 
import torch.nn as nn

from Sequence.embedding.material_embedding import MaterialWordEmbedding
from Sequence.embedding.positional_encoding import PositionalEncoding

torch.set_printoptions(sci_mode=False)

class SeqEmbeddingBlockClass(nn.Module):

    def __init__(self, reagent_list:list, ps_dim:int, device:str, matbert_hidden_size=768, pos_enc_dim=512, pos_enc_len=512, seed_num=None, drop_prob_output=0.0, drop_prob_mat_we=0.0, rgn_vec_onoff=False):
        """
        Purpose
        ----------
        constructor of material word vector & positional information
        token embedding + positional encoding (sinusoid)
            1. MaterialWordEmbedding can give word vector from MatBERT
                - generate 768 dimension word vector from MatBERT 
                - reduce dimension of word vector from nn.Linear
            2. positional encoding can give positional information to network

        Parameters
        ----------
        - reagent_list: : list
            - include reagent type. 
            - ex) ["Ag seed","NaBH4","AgNO3","H2O2","citrate","H2O"]
        - ps_dim : int
            - Size of word embeddings
            - (임베딩 차원 크기)
        - device : str
            - "cuda" or "cpu"
        - matbert_hidden_size=768 : int 
            - hidden layer size of MatBERT
            - if you trained MatBERT with different hidden layer size then you input that numbers
        - pos_enc_dim=512 : int
            - dimension of model, only even number due to denominator 
            - (positional encoding 하나의 차원, Transformer에서는 512)
        - pos_enc_len=512 : int
            - max sequence length
            - 입력 문장의 길이 (Transformer에서는 512, 512 넘어가는 문장을 넣으면 짤림)
        - seed_num=None : int 
            - seed number of torch
        - drop_prob_output=0.0 : float
            - the probability of dropout for final output vector
        - drop_prob_mat_we=0.0 : float
            - the probability of dropout for material embedding vector
        - rgn_vec_onoff : Boolean
            - True -> on (MatBERT-embedding+positional encoding), False -> off (only positional encoding)
        """
        super(SeqEmbeddingBlockClass, self).__init__()
        self.tok_emb = MaterialWordEmbedding(reagent_list, ps_dim, device, matbert_hidden_size, seed_num, drop_prob_mat_we)
        self.pos_emb = PositionalEncoding(reagent_list, ps_dim, device, pos_enc_dim, pos_enc_len, seed_num)
        self.device=device
        self.rgn_vec_onoff=rgn_vec_onoff
        self.drop_out = nn.Dropout(p=drop_prob_output).to(self.device)

    def forward(self, x):
        # print("pos_emb",pos_emb.shape)
        # print("rgn_emb",rgn_emb.shape)
        pos_emb = self.pos_emb(x)

        if self.rgn_vec_onoff == True:
            rgn_emb = self.tok_emb(x)
            # out=torch.concat((rgn_emb,pos_emb),dim=1).to(self.device) # 잘 되는지 확인 완료, concat
            out=torch.add(rgn_emb,pos_emb).to(self.device) # 잘 되는지 확인 완료, summation
            # print("rgn_emb+pos_emb",out)
            # print("rgn_emb+pos_emb.shape",out.shape)
            out = self.drop_out(out)
        else:
            out = self.drop_out(pos_emb)
        
        return out
if __name__ == "__main__":
    reagent_lists=["Ag seed","AgNO3","H2O2","Citrate","H2O","NaBH4"]
    ps_dim=4
    seed_num=1
    
    embed_block=SeqEmbeddingBlockClass(["Ag seed","AgNO3","H2O2","Citrate","H2O","NaBH4"], ps_dim, device="cpu", seed_num=seed_num)

    # ["AgNO3","H2O2","Citrate","H2O","NaBH4"]
    # result=embed_block([["Ag seed","AgNO3","H2O2","Citrate","H2O","NaBH4"], ["Ag seed","AgNO3","H2O2","Citrate","H2O","NaBH4"]])
    result=embed_block([["Ag seed","AgNO3","H2O2","Citrate","H2O","NaBH4"]])

    print(result)
    print(result.shape)