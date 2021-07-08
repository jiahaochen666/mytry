import os
from torch.utils.data import Dataset, DataLoader


class PointCloudDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, "training")
        self.lidar_dir = os.path.join(self.train_dir, "velodyne")

    def __len__(self):
        return 7481

    def __getitem__(self, index):
        lidar_file = os.path.join(self.lidar_dir, "%06d.bin" % index)


if __name__ == '__main__':
    path = "../../Data/training/velodyne/000000.bin"
    with open(path) as f:
        s = f.readlines()
        print(s)