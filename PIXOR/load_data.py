import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def my_collate_train(batch):
    """
    Collate function for training dataset. How to concatenate individual samples to a batch.
    Point Clouds and labels will be stacked along first dimension
    :param batch: list containing a tuple of items for each sample
    :return: batch data in desired form
    """

    point_clouds = []
    labels = []
    for tuple_id, tuple in enumerate(batch):
        point_clouds.append(tuple[0])
        labels.append(tuple[1])

    point_clouds = torch.stack(point_clouds)
    labels = torch.stack(labels)
    return point_clouds, labels

class Object3D(object):
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.height = data[8]  # box height
        self.width = data[9]  # box width
        self.length = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

def load_velo_pc(velo_filename):
    point_cloud = np.fromfile(velo_filename, dtype=np.float32)
    point_cloud = point_cloud.reshape((-1, 4))
    return point_cloud

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    lines = [line for line in lines if line.split(' ')[0] != 'DontCare']
    objects = [Object3D(line) for line in lines]
    return objects

def load_data(root_dir: str):
    batches = []
    train_dir = os.path.join(root_dir, "training")
    lidar_dir = os.path.join(train_dir, "velodyne")
    label_dir = os.path.join(train_dir, "label_2")
    for i in range(2):
        lidar_file = os.path.join(lidar_dir, "%06d.bin" % i)
        lidar_data = load_velo_pc(lidar_file)
        label_file = os.path.join(label_dir, "%06d.txt" % i)
        label_data = read_label(label_file)
        batches.append((lidar_data, label_data))
    train_data = DataLoader(batches, shuffle=True, batch_size=1, num_workers=0, collate_fn=my_collate_train)
    return train_data

if __name__ == '__main__':
    a = load_data("../../Data")
    for x, (data, label) in enumerate(a):
        print(data)