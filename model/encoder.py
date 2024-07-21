'''
 This is the encoder model of VQ-VAE. It is based on the convolution architecture!
'''

import torch
import torch.nn as nn
from .residual_block import Residual

class VQ_Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=256, down_sample=4, residual_num=2):
        super().__init__()
        
        # down sample part
        down_module_list = []
        down_module_list.append(nn.Conv2d(in_channels, hidden_size, kernel_size=4, stride=2, padding=1))
        for _ in range(down_sample//2 - 1):
            down_module_list.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1))
        self.down_module_seq = nn.Sequential(*down_module_list)
        
        # residual part
        self.res_seq = nn.Sequential(*[Residual(hidden_size) for _ in range(residual_num)])
            
    
    def forward(self, x):
        x = self.down_module_seq(x)
        x = self.res_seq(x)
        return x




if __name__ == '__main__':
    x = torch.randn(3,128,128)
    model = VQ_Encoder()
    y = model(x)
    print(y.shape)