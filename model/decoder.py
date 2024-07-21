import torch
import torch.nn as nn
from .residual_block import Residual


class VQ_Decoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=256, down_sample=4, residual_num=2) -> None:
        super().__init__()

        self.res_seq = nn.Sequential(*[Residual(hidden_size) for _ in range(residual_num)])

        up_module_list = []
        for _ in range(down_sample//2 - 1):
            up_module_list.append(nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1))
        up_module_list.append(nn.ConvTranspose2d(hidden_size, in_channels, kernel_size=4, stride=2, padding=1))
        self.up_module_seq = nn.Sequential(*up_module_list)
        

        
    def forward(self, x):
        x = self.res_seq(x)
        x = self.up_module_seq(x)
        return x
    
    
    
    
    
    
if __name__ == '__main__':
    x = torch.randn(256,32,32)
    model = VQ_Decoder()
    y = model(x)
    print(y.shape)
    