from model import VQ_VAE
# from loss import VQ_Loss
from torch.optim import Adam
from dataset import imagenet_train_dataloader
import torch
import torch.nn as nn
from transformer import Transformer
from torch.utils.tensorboard import SummaryWriter


def checkpoint_edit(prev):
    new = dict()
    for key,val in prev.items():
        key_new = key.replace("_orig_mod.","vqvae.")
        new[key_new] = val
    return new


device = "cuda"
num_steps = 250000


vqvae = VQ_VAE()
model = Transformer(vqvae)
model.to(device)

# print(model)
loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-4)
writer = SummaryWriter("logs/logtransformer_1")


checkpoint = torch.load('./checkpoint/checkpoint150000.pth')
new_checkpoint = checkpoint_edit(checkpoint['model_state_dict'])
model.load_state_dict(new_checkpoint, strict=False)

for name, param in model.named_parameters():
    if "vqvae" in name:
        param.requires_grad = False

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        
def train():
    step = 0
    while step <= num_steps:
        for image,label in imagenet_train_dataloader:
            if step > num_steps:
                break
            optimizer.zero_grad()
            image = image.cuda()
            logit, target = model(image)
            # print(logit,target,len(logit))
            # print(logit.shape, target.shape)
            loss = loss_func(logit,target)
            
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar("loss", loss.detach(), step)
            print(f"step: {step}, loss: {loss:.2f}")
            
            save_path = f"./checkpoint_2/checkpoint_transformer{step}.pth"
            if step%1000 == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
    writer.close()


model = torch.compile(model)

train()

        