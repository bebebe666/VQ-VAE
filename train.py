from model import VQ_VAE
from loss import VQ_Loss
from torch.optim import Adam
from dataset import imagenet_train_dataloader
import torch
from torch.utils.tensorboard import SummaryWriter


device = "cuda"
num_steps = 250000


model = VQ_VAE()
model.to(device)
loss_func = VQ_Loss()

optimizer = Adam(model.parameters(), lr=2e-4)
writer = SummaryWriter("logs/log4")



def train():
    step = 0
    while step <= num_steps:
        for image,label in imagenet_train_dataloader:
            if step > num_steps:
                break
            optimizer.zero_grad()
            image = image.cuda()
            output = model(image)
            loss = loss_func(image, output)
            
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar("loss", loss.detach(), step)
            print(f"step: {step}, loss: {loss:.2f}")
            
            save_path = f"./checkpoint_2/checkpoint{step}.pth"
            if step%1000 == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
    writer.close()


model = torch.compile(model)

train()

        