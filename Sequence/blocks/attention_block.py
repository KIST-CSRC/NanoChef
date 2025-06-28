"""
@author : Hyuk Jun Yoo
@when : 2023-12-12
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import sys
sys.path.append("./")

import torch
import torch.nn as nn

from Sequence.attention.attention_layer import AttentionLayer

class AttentionBlockClass(nn.Module):

    def __init__(self, device, att_n_blocks, d_model, ffn_hidden, att_n_head, drop_prob_attention=0.0, seed_num=None):
        """
        Purpose
        ----------
        constructor of attention block class (include AttentionLayer)

        Parameters
        ----------
        - device : object
            - "cuda" or "cpu"
        - att_n_blocks : int 
            - the number of multi-head attention layer
        - d_model : int
            - ***(Caution)*** the hidden size of multi-head attention layer --> d_model
        - ffn_hidden : int
            - ***(Caution)*** the hidden size of PositionwiseFeedForward layer --> ffn_hidden
        - att_n_head : int
            - the number of head in multi-head attention
            - 시약 종류의 갯수를 파악할 만큼 넣어주면 됨.
            - ex) metal precursor, metal seed, surfactant, reductant ...
        - drop_prob_attention : int
            - the probability of dropout
        - seed_num=None : int 
            - seed number of torch
        """
        super(AttentionBlockClass, self).__init__()
        if type(seed_num) == int:
            torch.manual_seed(seed_num)
        self.device=device
        self.layers = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                ffn_hidden=ffn_hidden,
                                                att_n_head=att_n_head,
                                                drop_prob=drop_prob_attention)
                                    for _ in range(int(att_n_blocks))]).to(self.device)
        output_dim=int(d_model/2)
        self.fc=nn.Linear(d_model, output_dim).to(device)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)

        out=self.fc(x)

        return out

if __name__ == "__main__":
    from Sequence.blocks.seq_embedding_block import SeqEmbeddingBlock
    # reagent_sequence=["Au seed","AgNO3","H2O2","Citrate","H2O","NaBH4"]
    reagent_sequence=["AgNO3","H2O2","Citrate","H2O","NaBH4"]
    output_dim=6 # len(reagent_sequence) or others
    att_n_head=4
    seed_num=1
    
    embed_block=SeqEmbeddingBlock(reagent_sequence, output_dim, seed_num=seed_num)

    # ["AgNO3","H2O2","Citrate","H2O","NaBH4"]
    result=embed_block(reagent_sequence)

    attention_block = AttentionBlockClass(att_n_blocks=2,d_model=2*output_dim,ffn_hidden=2*output_dim,att_n_head=att_n_head,seed_num=1)
    result2 = attention_block(result)
    
    print(result2)
    print(result2.shape)