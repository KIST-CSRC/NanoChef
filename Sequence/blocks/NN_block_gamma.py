"""
@author : Hyuk Jun Yoo
@when : 2023-12-15
@email : yoohj9475@naver.com, yoohj9475@gmail.com
"""
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
import torch.distributions as dist
import torch.nn.init as init
import math

class ScaledSigmoid(nn.Module):
    def __init__(self, scale=2):
        super(ScaledSigmoid, self).__init__()
        self.scale = scale

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.scale * x))

class BNN(nn.Module):
    def __init__(self, input_dim, output_dim, nn_n_hidden, seed_num=None, device='cpu'):
        """
        Purpose
        ----------
        constructor of neural network

        Parameters
        ----------
        - input_dim : int
            - the dimension of sequence embedding vector 
            - total vector --> concat((material_word_vector,position_encoding_vector))
        - nn_n_hidden : int
            - hidden size of BNN 
        - seed_num=None : int 
            - seed number of torch
        """
        super(BNN, self).__init__()
        self.device=device
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.nn_n_hidden=nn_n_hidden
        # self.scaled_sigmoid = ScaledSigmoid(scale=2)
        
        if type(seed_num) == int:
            torch.manual_seed(seed_num)
        self.z_h = nn.Sequential(
            nn.Linear(input_dim, nn_n_hidden),
            nn.Tanh(),
            nn.Linear(nn_n_hidden, nn_n_hidden),
            nn.Tanh()
        )
        self.z_mu = nn.Sequential(
            nn.Linear(nn_n_hidden, self.output_dim),
            nn.Tanh()
        )
        self.n_observation=0

        # sigma_tensor=torch.zeros(input_dim, moutput_dim).to(self.device)
        # for batch_tensor in sigma_tensor:
        #     gamma_sampling_tensor=torch.tensor([dist.Gamma(shape, rate=1).sample() for shape in batch_tensor]).to(self.device)
        #     gamma_sampling_tensor = torch.unsqueeze(gamma_sampling_tensor, dim=0)
        #     sigma_tensor=torch.cat((sigma_tensor, gamma_sampling_tensor), dim=0)

    def forward(self, x):
        """
        return
        - mu : the mean of gaussian distribution in BNN 
            --> grad_fn=<AddBackward0>, shape=(batch_size, 1)
        - sigma : the deviation of gaussian distribution in BNN 
            --> grad_fn=<ExpBackward0>, shape=(batch_size, 1)
        """
        z_h = self.z_h(x)
        mu_tensor = self.z_mu(z_h)

        shape=12/math.sqrt(self.n_observation)
        sigma_tensor=self.samplingSigma(mu_tensor.shape[0], shape=shape)
        
        return mu_tensor, sigma_tensor
        # return pi_tensor, mu_tensor
    
    def samplingSigma(self, batch_size, shape=1, rate=1):
        sigma_tensor=torch.zeros(batch_size, self.output_dim).to(self.device)
        for i, row in enumerate(sigma_tensor):
            for j, value in enumerate(row):
                # concentration = shape parameter, alpha
                # rate = rate parameter=1/beta (beta=scale parameter)
                sigma_tensor[i, j]=torch.tensor([dist.Gamma(shape, rate=rate).sample()]).to(self.device)

        return sigma_tensor
    
    def initialize_weights(self):
        # 가중치 초기화를 위한 함수
        def init_layer_weights(layer):
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier 초기화
                if layer.bias is not None:
                    init.zeros_(layer.bias)  # 편향 0으로 초기화

        # 모델에 있는 모든 레이어에 초기화 적용
        self.apply(init_layer_weights)