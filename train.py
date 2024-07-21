from model import VQ_VAE
from loss import VQ_Loss
from torch.optim import Adam
from dataset import imagenet_train_dataloader

device = "cuda"

model = VQ_VAE()
model.to(device)
loss = VQ_Loss()

optimizer = Adam(model.parameters(), lr=2e-4)

