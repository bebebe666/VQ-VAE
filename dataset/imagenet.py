from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    transforms.Resize(size = (128,128)),
    
])

dataset_root = '/root/autodl-tmp/imagenet/train'

train_dataset = ImageFolder(root=dataset_root, transform=data_transform)

imagenet_train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
