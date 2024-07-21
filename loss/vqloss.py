import torch
import torch.nn as nn


class VQ_Loss(nn.Module):
    def __init__(self, beta=0.25, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def forward(self,x, output):
        
        loss_predict = torch.mean((x - output["x"])**2)
        loss_codebook = torch.mean((output["z"].detach() - output["zq"])**2)
        loss_commit = torch.mean((output["z"] - output["zq"].detach())**2)
        
        return loss_predict + loss_codebook + self.beta* loss_commit