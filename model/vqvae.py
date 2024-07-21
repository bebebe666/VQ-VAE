import torch
import torch.nn as nn

from .encoder import VQ_Encoder
from .decoder import VQ_Decoder


def _init_weight(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
            
            
class VQ_VAE(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        hidden_size=256, 
        input_size=128, 
        latent_size=32, 
        residual_num=2, 
        K=512,
        *args, 
        **kwargs
    ) -> None:
    
        super().__init__()
        
        if input_size%latent_size != 0:
            print("The input size and the latent size are not compatibile! ")
            exit()
            
        self.latent_size = latent_size
        
        down_sample = input_size // latent_size
        self.encoder = VQ_Encoder(in_channels, hidden_size, down_sample, residual_num)
        self.decoder = VQ_Decoder(in_channels, hidden_size, down_sample, residual_num)
        
        self.code_books = nn.Parameter(torch.rand(K, hidden_size))

        self.apply(_init_weight)
        self.code_books.data.uniform_(-1/K, 1/K)
        
        
        
    def find_codebook(self,z):
        z = z.flatten(2) # z[N, 256, 1024]
        z = z.permute(0, 2, 1) #z[N, 1024, 256]
        
        code_books = self.code_books
        
        _z = z[:, :, None, :]
        _code_books = code_books[None, None, :, :] #[N, 1024,K,256]
        # print(_z.shape, _code_books.shape)
        distance = torch.sum((_z - _code_books)**2, dim = -1) #[N, 1024,K]
        index = torch.argmin(distance, dim=2) #[N, 1024]
        
        zq = code_books[index, :]
        zq = zq.reshape(z.shape[0], self.latent_size, self.latent_size, -1)
        zq = zq.permute(0, 3, 1, 2)
        
        return zq
            
        
        
    
    def forward(self, x):
        
        z = self.encoder(x) 
        zq = self.find_codebook(z.detach())
        
        z = zq + (z - zq).detach()
        x_predict = self.decoder(zq)
        
        output = {}
        output["x"] = x_predict
        output["z"] = z
        output["zq"] = zq
        return output
    


if __name__ == "__main__":
    x = torch.rand(6, 3,128,128)
    model = VQ_VAE()
    y = model(x)
    print(y["x"].shape)