import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


def voxelize(point_cloud: np.ndarray):

    y_dim, x_dim, z_dim = 800, 700, 36
    point_x = point_cloud[:, 0]
    point_y = point_cloud[:, 1]
    point_z = point_cloud[:, 2]
    point_r = point_cloud[:, 3]

    qx = ((point_x + 80) * 10 / 2 / 8 * 7).astype(int)
    qy = ((point_y + 80) * 10 / 2).astype(int)
    qz = (point_z + 32).astype(int)

    q_point = np.dstack((qx, qy, qz, point_r)).squeeze()

    voxel_grid = np.zeros(shape=(x_dim, y_dim, z_dim), dtype=np.float32)
    reflect_value = np.zeros(shape=(x_dim, y_dim), dtype=np.float32)
    reflect_count = np.ones(shape=(x_dim, y_dim), dtype=int)

    for i, point in enumerate(q_point):
        point = point.astype(int)
        voxel_grid[point[0], point[1], point[2]] = 1
        reflect_value[point[0], point[1]] += point[3]
        reflect_count[point[0], point[1]] += 1

    reflect_value /= reflect_count

    voxel = np.dstack((voxel_grid, reflect_value))

    return voxel


def load_velo_pc(velo_filename: str):
    point_cloud = np.fromfile(velo_filename, dtype=np.float32)
    point_cloud = point_cloud.reshape((-1, 4))
    return point_cloud


def load_data(root_dir: str):
    batches = []
    train_dir = os.path.join(root_dir, "training")
    lidar_dir = os.path.join(train_dir, "velodyne")
    label_dir = os.path.join(train_dir, "label_2")
    for i in range(1):
        lidar_file = os.path.join(lidar_dir, "%06d.bin" % i)
        raw_point_cloud = load_velo_pc(lidar_file)
        voxel_point_cloud = voxelize(raw_point_cloud)

    return raw_point_cloud


if __name__ == '__main__':
    a = load_data("../../Data")
