import functools
from typing import Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    def __init__(self,
                 # *,
                 in_dim: int,
                 mlp_dim: int,
                 out_dim: Optional[int]=None,
                 dropout_rate: float=0.1,
                 **kwargs):
        super(MlpBlock, self).__init__()
        self.hidden_layer = nn.Linear(in_dim, mlp_dim)
        self.activate = nn.GELU(approximate='tanh')
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        if out_dim is None:
            self.out_layer = nn.Linear(mlp_dim, in_dim)
        self.out_layer = nn.Linear(mlp_dim, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activate(x)
        x = self.drop1(x)
        x = self.out_layer(x)
        x = self.drop2(x)
        return x 
    

class TokenLearnerModule(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_tokens: int,
                 bottleneck_dim: int=64,
                 dropout_rate: float=0.):
        super(TokenLearnerModule, self).__init__()
        self.mlp = MlpBlock(in_dim=in_dim, mlp_dim=bottleneck_dim, out_dim=num_tokens, dropout_rate=dropout_rate)
        self.layernorm = nn.LayerNorm(in_dim, eps=1e-6)
    
    def forward(self, x):
        if len(x.shape) == 4:
            b, c, h, w = x.shape
            x = x.reshape(b, h*w, c)
        
        selected = self.layernorm(x)
        selected = self.mlp(selected) # shape: [b, h*w, n_token]
        
        selected = selected.transpose(2,1)
        selected = nn.functional.softmax(selected, dim=-1)

        feat = torch.einsum("...si,...id->...sd", [selected, x])   
        return feat # shape: [b, n_token, c]     


if __name__ == '__main__':
    tokenLearner = TokenLearnerModule(512, num_tokens=8)
    input_vec = torch.ones(2, 81, 512)
    tokenLearner.eval()
    output = tokenLearner(input_vec) #[2,8,512]
    import pdb
    pdb.set_trace()
    