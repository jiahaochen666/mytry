import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load training and testing datasets from torchvision.datasets.Kitti
# Data structure of training is a list of 7481 * (image, label)
# Data structure of each image is 3-channel, 370 * 1224

train_data = datasets.Kitti(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.Kitti(root="data", train=False, download=True, transform=ToTensor())
trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)
