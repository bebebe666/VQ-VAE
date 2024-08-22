import torch
import torch.nn as nn
# from model.vqvae import VQ_VAE
from .mingpt import GPT

class Transformer(nn.Module):
    def __init__(self, 
                vqvae,
                sos_token=0,
                pkeep=0.5,
                ):
        
        super().__init__()
        self.vqvae = vqvae
        self.sos_token = sos_token
        
        block_size = self.vqvae.latent_size**2 + 2
        self.vocab_size = self.vqvae.latent_size**2
        
        self.pkeep = pkeep
        self.gpt = GPT(self.vocab_size, block_size)
        
    
    def forward(self, x):
        z = self.vqvae.encoder(x)
        _, index = self.vqvae.find_codebook(z)
        N = index.shape[0]
        
        mask = torch.bernoulli(self.pkeep * torch.ones_like(index))
        mask = mask.round().to(dtype=index.dtype)
        random_index = torch.randint_like(index, self.vocab_size)
        m_index = mask*index + (1-mask)*random_index
        
        sos_token = self.sos_token*torch.ones(N,1)
        sos_token = sos_token.long().to(index.device)
        
        
        in_index = torch.cat((sos_token, m_index), dim = 1)
        logit = self.gpt(in_index[:,:-1])
        
        return logit, index
    
    