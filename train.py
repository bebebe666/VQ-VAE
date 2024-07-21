from model import VQ_VAE
from loss import VQ_Loss
from torch.optim import Adam
from dataset import imagenet_train_dataloader
import torch

device = "cuda"
save_path = "./checkpoint"
num_steps = 250000


model = VQ_VAE()
model.to(device)
loss_func = VQ_Loss()

optimizer = Adam(model.parameters(), lr=2e-4)




def train():
    step = 0
    while step < num_steps:
        for image,label in imagenet_train_dataloader:
            optimizer.zero_grad()
            image = image.cuda()
            output = model(image)
            loss = loss_func(output)
            
            loss.backward()
            optimizer.step()
        step += 1
        
        if step%100 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)



train()

        