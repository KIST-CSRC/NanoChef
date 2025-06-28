import sys
sys.path.append("./")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizerFast,AutoModel,AutoConfig
from Sequence.blocks.seq_embedding_block import SeqEmbeddingBlockClass
from Sequence.blocks.attention_block import AttentionBlockClass
from Sequence.blocks.NN_block_gamma import BNN


class SeqOpt(nn.Module):

    def __init__(self, 
                reagent_list: list,
                ps_dim: int,
                output_dim: int,
                num_variables: list,
                device: str,
                # att_n_blocks: int,
                # att_n_head: int,
                nn_n_hidden: int,
                seed_num=None,
                emb_matbert_hidden_size=768,
                emb_pos_enc_dim=512,
                emb_pos_enc_len=512,
                emb_drop_prob_mat_we=0,
                emb_drop_prob_output=0,
                # att_drop_prob_attention=0,
                rgn_vec_onoff=False):
        super(SeqOpt, self).__init__()

        self.device=device

        self.emb_block=SeqEmbeddingBlockClass(
            reagent_list=reagent_list,
            ps_dim=ps_dim,
            device=device,
            matbert_hidden_size=emb_matbert_hidden_size,
            pos_enc_dim=emb_pos_enc_dim,
            pos_enc_len=emb_pos_enc_len,
            drop_prob_mat_we=emb_drop_prob_mat_we,
            drop_prob_output=emb_drop_prob_output,
            seed_num=seed_num, 
            rgn_vec_onoff=rgn_vec_onoff)
        
        # self.att_block=AttentionBlockClass(
        #     device=device,
        #     att_n_blocks=att_n_blocks,
        #     d_model=2*output_dim,
        #     ffn_hidden=2*output_dim,
        #     att_n_head=att_n_head,
        #     drop_prob_attention=att_drop_prob_attention,
        #     seed_num=seed_num
        #     )
        
        input_dim=len(reagent_list)*ps_dim+num_variables
        
        self.nn_block=BNN(
            input_dim=input_dim,
            output_dim=output_dim,
            nn_n_hidden=nn_n_hidden,
            seed_num=seed_num,
            device=device
            ).to(device)
        
        self.model_name='matbert-base-cased' 
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=False)
    
    def forward(self, rgn_seq, rgn_cond, src_mask=None) -> tuple:
        embedded_rgn = self.emb_block(rgn_seq).to(self.device)
        # attention_rgn = self.att_block(embedded_rgn, src_mask)

        batch_size,seq_len,ps_dim=embedded_rgn.size()
        embedded_rgn=torch.reshape(embedded_rgn, (batch_size, seq_len*ps_dim))
        rgn_cond_array=np.array(rgn_cond)
        convert_rgn_cond=torch.tensor(rgn_cond_array).to(self.device)
        rgn_vec=torch.cat((embedded_rgn, convert_rgn_cond), dim=1).to(self.device)
        flattened_tensor = rgn_vec.float()

        out = self.nn_block(flattened_tensor)

        return out


if __name__ == "__main__":
    from torchinfo import summary
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    import numpy as np

    rgn_seq=[
        ["AgNO3","H2O2","Citrate","H2O","NaBH4"],
        ["AgNO3","H2O2","Citrate","H2O","NaBH4"]
    ]
    rgn_cond=[
        [0.09,0.23,0.35,0.11,0.83],
        [0.09,0.23,0.35,0.11,0.83]
    ]

    # total_mat_vec_gpu=generateMatVec(rgn_seq).to(device)
    # total_rgn_vec_gpu=torch.tensor(rgn_cond).to(device)

    seq_opt_obj=SeqOpt(
        reagent_list=["AgNO3","Citrate","H2O","H2O2","NaBH4"],
        num_variables=5,
        ps_dim=4,
        output_dim=1, # hyperparameter
        device=device, 
        # att_n_blocks=2, # hyperparameter
        # att_n_head=4, # 시약 종류 갯수 ex) metal precursor, metal seed, surfactant, reductant ...
        nn_n_hidden=20, # hyperparameter
        seed_num=1)
    result=seq_opt_obj(rgn_seq, rgn_cond)
    # seq_opt_obj=nn.DataParallel(seq_opt_obj)
    # summary(seq_opt_obj)
    # print(seq_opt_obj)
    result=seq_opt_obj(rgn_seq, rgn_cond)
    print(result) # type(result) --> tuple

    """
    GPU vs CPU --> 지금 GPU가 약간 느린 것은 training 과정이 아니기 때문. 작은 모델에서는 단순 prediction의 경우 CPU가 더 빠를 수도?

    import time
    start_test=time.time()
    for i in range(10): 
        seq_opt_obj(rgn_seq, rgn_cond)
    finish_test=time.time()
    print("CUDA : ",finish_test-start_test) # CUDA :  38.008267402648926
    print("CPU : ",finish_test-start_test) # CPU :  34.61173415184021
    """
