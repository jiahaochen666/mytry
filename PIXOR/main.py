from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.Kitti(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.Kitti(root="data", train=False, download=True, transform=ToTensor())