import torch

import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, hidden_size = 256) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_size,hidden_size,kernel_size=1,stride=1,padding=0)
        self.seq = nn.Sequential(
            self.relu1,
            self.conv1,
            self.relu2,
            self.conv2,
        )
    
    
    def forward(self,x):
        return x + self.seq(x)


if __name__ == '__main__':
    res_block = Residual()
    input = torch.randn((256,128,128))
    output = res_block(input)
    print(output.shape)