"""
@author : Hyuk Jun Yoo
@when : 2023-12-12
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
warnings.filterwarnings("ignore")
from transformers.utils import logging
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast,AutoModel,AutoConfig
import numpy as np

torch.set_printoptions(sci_mode=False)

class MaterialWordEmbedding(nn.Module):

    def __init__(self, reagent_list:list, output_dim:int, device:object, matbert_hidden_size=768, seed_num=None, drop_prob_mat_we=0.0):
        """
        Purpose
        ----------
        constructor of material word vector class
            1. MatBERT에서 768차원의 word vector 생성
            2. nn.Linear 로 시스템에 맞게 word vector 차원 축소

        Parameters
        ----------
        - reagent_list: : list
            - include reagent type. 
            - ex) ["Ag seed","NaBH4","AgNO3","H2O2","citrate","H2O"]
        - output_dim : int
            - Size of word embeddings (임베딩 차원 크기)
        - device : object
            - "cuda" or "cpu"
        - matbert_hidden_size=768 : int 
            - hidden layer size of MatBERT
            - if you trained MatBERT with different hidden layer size then you input that numbers
        - seed_num=None : int 
            - seed number of torch
        - drop_prob_mat_we=0.0 : float
            - the probability of dropout for material embedding vector
        """
        super(MaterialWordEmbedding, self).__init__()

        # cased : 대소문자 구별
        # uncased : 대소문자 구별X
        self.model_name='matbert-base-cased' 
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=False)
        self.device=device

        if type(seed_num) == int:
            torch.manual_seed(seed_num)

        self.fc = nn.Linear(matbert_hidden_size, output_dim).to(self.device)
        self.drop_out = nn.Dropout(p=drop_prob_mat_we).to(self.device)

    def _tokenize(self, word_lists):
        """
        word_list (list) : sequence list ex) [["NaBH4","AgNO3","H2O2","citrate","H2O"], ...]
        """
        after_word_lists=[]
        # print("word_list", word_list)
        for word_list in word_lists:
            # print(word_list)
            after_word_list=[]
            for word in word_list:
                input_ids=self.tokenizer(word)["input_ids"]
                # after_word_list.append(input_ids)
                after_word_list.append(input_ids[1:-1]) # delete padding
            # print("after_word_list", after_word_list)
            after_word_lists.append(after_word_list)
        return after_word_lists
        
    # def forward(self, reagent_list):
    def forward(self, input_reagent_lists):
        
        input_ids_tokens=self._tokenize(input_reagent_lists)
        # print("reagent_list",input_ids_tokens)
        # call MatBERT for embedding vector
        config=AutoConfig.from_pretrained(self.model_name, add_pooling_layer=False)
        matbert_model = AutoModel.from_pretrained(self.model_name, config=config).to(self.device)
        # print(len(self.tokenizer))
        matbert_model.resize_token_embeddings(len(self.tokenizer))

        total_embedding_vectors=torch.tensor([]).to(self.device)
        #embedded = [sent len, batch size, emb dim]
        # print("input_ids_tokens", input_ids_tokens)
        for input_ids_token in input_ids_tokens:
            # list convert ot tensor
            # print("input_ids_token", input_ids_token)
            # input_ids_tensor=torch.tensor(input_ids_token).to(self.device)
            # input_ids_tensor=torch.tensor([np.array(input_ids_token).flatten()]).to(self.device)
            word_embedding_vectors=torch.tensor([]).to(self.device)
            for input_id_token in input_ids_token:
                input_id_tensor=torch.tensor([input_id_token]).to(self.device)

                # calculate embedding vector            
                with torch.no_grad():
                    # print(input_id_tensor)
                    outputs = matbert_model(input_id_tensor)
                last_hidden_states = outputs.last_hidden_state

                # 마지막 레이어에서의 각 토큰에 대한 embedding vector 추출
                embedding_vectors = last_hidden_states[0]
                # print("embedding_vectors.shape", embedding_vectors.shape)
                # print("embedding_vectors", embedding_vectors)
                embedding_vectors = embedding_vectors.unsqueeze(0).to(self.device)  # shape[0]에 1차원 확장 // unsqueeze(-1)이면 shape[-1]에 1차원 확장
                # print("embedding_vectors.shape", embedding_vectors.shape)
                embedding_vectors=F.avg_pool2d(embedding_vectors, (embedding_vectors.shape[1],1)).squeeze(1).to(self.device) # Ag seed 같은 vector들을 avg_pool2d 로 average 값 사용
                # print("embedding_vectors.shape", embedding_vectors.shape)
                # print(embedding_vectors)
                word_embedding_vectors=torch.concat((word_embedding_vectors,embedding_vectors)).to(self.device) # 잘 되는지 확인 완료
            # print("[word_embedding_vectors]", torch.tensor(word_embedding_vectors).unsqueeze(0))
            total_embedding_vectors=torch.concat((total_embedding_vectors,torch.tensor(word_embedding_vectors).unsqueeze(0))).to(self.device) # 잘 되는지 확인 완료

            # input_ids_tensor=torch.tensor(input_ids_token).to(self.device)
            # print(input_ids_tensor)

            # # calculate embedding vector            
            # with torch.no_grad():
            #     outputs = matbert_model(input_ids_tensor)
            # last_hidden_states = outputs.last_hidden_state

            # # 마지막 레이어에서의 각 토큰에 대한 embedding vector 추출
            # embedding_vectors = last_hidden_states[0]
            # # print("embedding_vectors.shape", embedding_vectors.shape)
            # # print("embedding_vectors", embedding_vectors)
            # embedding_vectors = embedding_vectors.unsqueeze(0).to(self.device)  # shape[0]에 1차원 확장 // unsqueeze(-1)이면 shape[-1]에 1차원 확장
            # # print("embedding_vectors.shape", embedding_vectors.shape)
            # embedding_vectors=F.avg_pool2d(embedding_vectors, (embedding_vectors.shape[1],1)).squeeze(1).to(self.device) # Ag seed 같은 vector들을 avg_pool2d 로 average 값 사용
            # # print("embedding_vectors.shape", embedding_vectors.shape)
            # # print(embedding_vectors)
            # total_embedding_vectors=torch.concat((total_embedding_vectors,embedding_vectors)).to(self.device) # 잘 되는지 확인 완료
        # print("total_embedding_vectors",total_embedding_vectors.shape)
        # print("total_embedding_vectors",total_embedding_vectors.shape)
        total_embedding_vectors=torch.tensor(total_embedding_vectors).clone().detach().requires_grad_(True).to(self.device)
        # print("total_embedding_vectors", total_embedding_vectors)
        # print("total_embedding_vectors", total_embedding_vectors.shape)
        # print(total_embedding_vectors.device)

        out = self.fc(total_embedding_vectors)
        # print("out",out.shape)
        out = self.drop_out(out)
                
        return out
    

if __name__ == "__main__":
    # 각 단어를 고유한 정수로 매핑하는 사전을 생성
    # word_to_index = {"AgNO3": 0, "H2O2": 1, "Citrate": 2, "H2O": 3, "NaBH4": 4}
    # reagent_list=[
    #     ["AgNO3"],
    #     ["H2O2"],
    #     ['Citrate'],
    #     ["H2O"],
    #     ["NaBH4"]
    # ]
    reagent_list=[["NaBH4","AgNO3","H2O2","Citrate","H2O"]]
    
    # ["NaBH4","AgNO3","H2O2","Citrate","H2O"]
    # 입력 데이터를 정수로 변환
    # reagent_list = torch.tensor(reagent_list)
    # reagent_list = torch.tensor([[word_to_index[word] for word in sequence] for sequence in reagent_list])
        # [["AgNO3"],["H2O2",],["Citrate"],["H2O"],["NaBH4"]]

    # reagent_list=torch.asarray([
    #     # [0,1,2,3,4],
    #     # [0,1,4,3,2],
    #     # [1,0,2,4,3]
    #     ["AgNO3","H2O2","Citrate","H2O","NaBH4"]
    # ])

    # reagent_list="AgNO3 H2O2 Citrate H2O NaBH4"
    # print(reagent_list)
    # print(len(reagent_list))
    device="cpu"
    torch.set_printoptions(sci_mode=False)
    model=MaterialWordEmbedding(reagent_list=reagent_list, output_dim=6, seed_num=1, device=device)
    # for reagent in reagent_list:
    #     result=model(reagent)
    #     print(result)
    result=model(reagent_list)
    print(result)
    print(result.shape)


