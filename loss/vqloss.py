import torch
import torch.nn as nn


class VQ_Loss(nn.Module):
    def __init__(self, beta=0.25, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def forward(self,x, output):
        
        loss_predict = torch.sum((x - output["x"])**2)
        loss_codebook = torch.sum((output["z"].detach() - output["zq"])**2)
        loss_commit = torch.sum((output["z"] - output["zq"].detach())**2)
        
        return loss_predict + loss_codebook + self.beta* loss_commit