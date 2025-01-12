import numpy as np
import random
import os
import torch
import numpy as np
import h5py
import torch.utils

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from torchvision.utils import save_image
from plot import plot_pcd_multi_rows


class ParaTrainLoader_ShapeNetCore(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, rotation_mode=None):
        self.dataset_folder = dataset_folder
        self.rotation_mode = rotation_mode
        assert rotation_mode in ['rot_z', 'rot_so3', None]
        self.class_list = parse_list_file(os.path.join(dataset_folder, 'class_list.txt'))
        self.total_list = parse_list_file(os.path.join(dataset_folder, 'total_list.txt'))
        self.num_models_total = len(self.total_list)

    def __getitem__(self, model_index):
        np.random.seed()
        model_name = self.total_list[model_index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        model_path = os.path.join(self.dataset_folder, '256', class_name, model_name + '.npy')
        pts = bounding_box_normalization(np.load(model_path).astype(np.float32))
        pts = bounding_box_normalization(random_anisotropic_scaling(pts, 2/3, 3/2))
        if self.rotation_mode == 'rot_z':
            pts = random_axis_rotation(pts, 'z')
        elif self.rotation_mode == 'rot_so3':
            pts = random_rotation(pts)
        pts = bounding_box_normalization(pts)
        return pts, model_name

    def __len__(self):
        return self.num_models_total


class ParaTrainLoader_PatchCollection(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.h5_file = os.path.join(dataset_folder, 'points_100000_100_3.h5')
        self.num_patches_total = 100_000

    def __getitem__(self, patch_index):
        np.random.seed()
        fid = h5py.File(self.h5_file, 'r')
        pts = bounding_box_normalization(fid['points'][patch_index].astype(np.float32))
        fid.close()
        pts = bounding_box_normalization(random_rotation(bounding_box_normalization(pts)))
        return pts

    def __len__(self):
        return self.num_patches_total


class ParaLoader_ModelNet40_Cls(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, mode):
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.data_file = os.path.join(dataset_folder, 'modelnet40_para_from_5000_' + mode + '.h5')
        f = h5py.File(self.data_file, 'r')
        self.num_models = f['data'].shape[0]
        f.close()

    def __getitem__(self, index):
        np.random.seed()
        f = h5py.File(self.data_file, 'r')
        pgi = bounding_box_normalization(f['data'][index].astype(np.float32))
        cid = f['labels'][index].astype(np.int64)
        f.close()
        if self.mode == 'train':
            pgi = random_anisotropic_scaling(pgi, 2/3, 3/2)
            pgi = random_translation(pgi, 0.20)
        print(pgi.shape)
        return pgi, cid

    def __len__(self):
        return self.num_models


class ModelNet_PC(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.points = []
        self.labels = []
        with h5py.File(filename, "r") as f:
            self.cate_count = f["cate_count"][()]
            self.cate_name = f["cate_name"][:]
            self.models_count = f["models_count"][()]
            self.count_per_cate = f["count_per_cate"][:]
            self.data_group = f["data"]
            for idx, cate in enumerate(self.data_group):
                cate_group = self.data_group[cate]
                for point_name in cate_group:
                    point_data = cate_group[point_name][:]
                    self.points.append(point_data)
                    self.labels.append(idx)

    def __getitem__(self, index):
        pts = bounding_box_normalization(self.points[index])
        return pts, self.labels[index]

    def __len__(self):
        return len(self.points)


class ModelNet_REC(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.filename = filename
        with h5py.File(filename, "r") as f:
            self.models_count = f["models_count"][()]
            self.pgis = f["pgis"][:]
            self.pcds = f["pcds"][:]

    def __getitem__(self, index):
        pgi = bounding_box_normalization(self.pgis[index])
        return pgi, self.pcds[index]

    def __len__(self):
        return self.models_count


class ModelNet_CLS(torch.utils.data.Dataset):
    def __init__(self, filename, mode="train"):
        assert mode in ["train", "test"]
        self.filename = filename
        self.mode = mode
        with h5py.File(filename, "r") as f:
            self.models_count = f["models_count"][()]
            self.pgis = f["pgis"][:]
            self.labels = f["labels"][:]
        np.random.seed()

    def __getitem__(self, index):
        pgi = bounding_box_normalization(self.pgis[index])
        if self.mode == 'train':
            pgi = random_anisotropic_scaling(pgi, 2/3, 3/2)
            pgi = random_translation(pgi, 0.20)
        return pgi, self.labels[index]

    def __len__(self):
        return self.models_count


if __name__ == "__main__":
    pass

    # modelnet40 = ModelNet_PC("../data/ModelNet10/PC_2048/modelnet10_2048_train.h5")
    # modelnet40_loader = DataLoader(modelnet40, batch_size=5, shuffle=True)
    # data, label = next(iter(modelnet40_loader))
    # print(data.shape)
    # print(label.shape)

    modelnet_rec = ModelNet_REC("../data/ModelNet40/PC_2048/modelnet40_2048_256_49_train_112.h5")
    # modelnet_rec_loader = DataLoader(modelnet_rec, 32, shuffle=True)
    # pgi, pcd = modelnet_rec[20]
    # print(modelnet_rec.models_count)
    # print(pgi.shape, pcd.shape)
    SAMPLE_TIMES = 10
    for i in tqdm(range(SAMPLE_TIMES)):
        rand_index = random.randint(0, modelnet_rec.models_count - 1)
        pgi, pcd = modelnet_rec[rand_index]
        # Render PGI
        pgi: torch.Tensor = torch.from_numpy(pgi).cuda()
        visualize_pgi(pgi.unsqueeze(0))[0].save(f"./examples/sample_pgi_{i:02}.png")
        # Render PCD
        pcd: np.ndarray = np.expand_dims(pcd, axis=0)
        plot_pcd_multi_rows(f"./examples/sample_pcd_{i:02}.png", pcd, cmap="viridis")

    # modelNet_cls = ModelNet_CLS("../data/ModelNet10/PC_2048/modelnet10_2048_256_49_train_112.h5")
    # pgi, label = modelNet_cls[20]
    # print(pgi.shape, label)
    # print(pgi)
