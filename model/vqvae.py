import torch
import torch.nn as nn

from .encoder import VQ_Encoder
from .decoder import VQ_Decoder



class VQ_VAE(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        hidden_size=256, 
        input_size=128, 
        latent_size=32, 
        residual_num=2, 
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
        
        self.code_books = nn.Parameter(torch.rand(hidden_size,latent_size,latent_size))
    
    def find_codebook(self,z):
        z = z.flatten(1) # z[256, 1024]
        z = z.permute(1, 0) #z[1024, 256]
        
        code_books = self.code_books.flatten(1)
        code_books = code_books.permute(1, 0)
        
        _z = z[:, None, :]
        _code_books = code_books[None, :, :] #[1024,1024,256]
        distance = torch.sum((_z - _code_books)**2, dim = -1) #[1024,1024]
        index = torch.argmin(distance, dim=1) #[1024]
        
        zq = code_books[index, :]
        zq = zq.reshape(self.latent_size, self.latent_size, -1)
        zq = zq.permute(2, 0, 1)
        
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
    x = torch.rand(3,128,128)
    model = VQ_VAE()
    y = model(x)
    print(y["x"].shape)