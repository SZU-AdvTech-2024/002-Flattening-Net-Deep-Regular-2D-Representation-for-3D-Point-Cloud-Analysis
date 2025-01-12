import numpy as np
import torch
import h5py

from math import sqrt
from tqdm import tqdm
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# from dataset import ModelNet_PC_2048
from parametrization_comps import G2SD, S2PF
from utils import *


class FlatteningNet(nn.Module):
    def __init__(self, para_g_params, para_l_params, N_G, N_C, K):
        super(FlatteningNet, self).__init__()
        self.N_G = N_G
        self.N_C = N_C
        self.K = K
        is_square_number(N_G)
        is_square_number(K)
        n_G = int(N_G ** 0.5)
        k = int(K ** 0.5)
        M = N_G * K
        m = n_G * k
        self.n_G = n_G
        self.k = k
        self.M = M
        self.m = m
        self.para_g_module = G2SD(num_grids=N_G)
        self.para_g_module.load_state_dict(torch.load(para_g_params))
        self.para_g_module.eval()
        self.para_l_module = S2PF()
        self.para_l_module.load_state_dict(torch.load(para_l_params))
        self.para_l_module.eval()

    def forward(self, pts):
        B, _, device = pts.size(0), pts.size(1), pts.device
        N_G = self.N_G
        N_C = self.N_C
        K = self.K
        M = self.M
        self.para_g_module.to(device)
        self.para_l_module.to(device)
        with torch.no_grad():
            pts_g = index_points(pts, fps(pts, N_G))
            # 将 2D 晶格作为 Bottleneck 的 AE 所重建出来的点云数组隐含了 2D 晶格顺序排列的位置信息
            rec_g = self.para_g_module(pts_g)
            # 若以重建之后的稀疏点云作为 Guidance 去查询相应的 Context Points
            # 那么所有的 Context Points 在排列上必然同样以 2D 晶格为序的
            # 所以说之后的构建 PGI 的过程理论上应当依赖于此顺序，同时也不可以随意修改此顺序 
            pts_c = index_points(pts, knn_search(pts, rec_g, N_C))  # [B,Ng,Nc,3]
            pts_c_n = normalize_anchor_patches(pts_c)  # 归一化，所以最终的图像也是归一化的
            # 将所有的 Context Points 都转换为对应的二维坐标 [B,Ng,Nc,2]
            ebd_c = rescale_pe(self.para_l_module(pts_c_n.view(B * N_G, N_C, 3)), 0+1e-6, 1-1e-6).view(B, N_G, N_C, 2)
            # 将所有 Patches 重采样到大小为 K{k*k}(K>N_C) 的小网格中，对应的像素值是对应点的三维坐标
            pgi_local = seperately_grid_resample_patches(pts_c, ebd_c, K)  # [B,Ng,3,k,k]
            # 再将每个 Guidance 相关的所有 Patches 以特定顺序收集起来就是最终结果
            pgi_global = assemble_separate_patch_parameterizations(pgi_local)
            # 形状大小可以解释为 [B,3,M] <==> [B,3,m,m] <==> [B,3,n_G*k,n_G*k]
            pgi = pgi_global.view(B, 3, M).permute(0, 2, 1).contiguous()  # [B,M,3]{M=N_G*K}
        return pgi


if __name__ == "__main__":
    g2sd_path = "../ckpt/comps/g2sd.pth"
    s2pf_path = "../ckpt/comps/s2pf.pth"
    MODE = "train"
    # N = 1024
    # K = 49
    # N_G = 256
    # N_C = 40
    N = 1024
    K = 25
    N_G = 256
    N_C = 20
    net = FlatteningNet(g2sd_path, s2pf_path, N_G, N_C, K).cuda().eval()

    """ =========================================== ModelNet 系列 ======================================== """
    # NOTE: Resolusion is 112 >>> 16 * 7 = 112, 16 * 16 = 256, 7 * 7 = 49
    # NOTE: Resolution is  80 >>> 16 * 5 =  80, 16 * 16 = 256, 5 * 5 = 25
    src_path = f"../data/ModelNet40/PC_{N}/modelnet40_{N}_{MODE}.h5"
    """
    源数据集格式说明：
    cate_count [dataset]: 类别数量
    cate_name [dataset]: 类别名称
    models_count [dataset]: 点云总数
    count_per_cate [dataset]: 逐类别点云数量
    data [group]: 类别存储的点云数据（字典）
    """
    dataset = {}
    with h5py.File(src_path, "r") as fsrc:
        dataset["cate_count"] = fsrc["cate_count"][()]
        dataset["cate_name"] = [name.decode("utf-8") for name in fsrc["cate_name"][:]]
        dataset["models_count"] = fsrc["models_count"][()]
        dataset["count_per_cate"] = fsrc["count_per_cate"][:]
        dataset["pgis"] = []
        dataset["pcds"] = []
        dataset["labels"] = []
        data_group = fsrc["data"]
        for idx, cate in enumerate(data_group, 0):
            cate_group = data_group[cate]
            for point_name in tqdm(cate_group, desc=cate, ncols=150):
                point_data = cate_group[point_name][:].astype(np.float32)
                point_data = bounding_box_normalization(point_data)
                pts = torch.from_numpy(point_data).unsqueeze(0).cuda()
                pgi: torch.Tensor = net(pts)
                dataset["pgis"].append(pgi.squeeze(0).cpu().numpy())
                dataset["pcds"].append(point_data)
                dataset["labels"].append(idx)

    dst_path = f"../data/ModelNet40/PC_{N}/modelnet40_{N}_{N_G}_{K}_{MODE}_{int(sqrt(N_G*K))}.h5"
    """
    目标数据集格式说明：
    cate_count [dataset]: 类别数量
    cate_name [dataset]: 类别名称
    models_count [dataset]: 点云总数
    count_per_cate [dataset]: 逐类别点云数量
    pgis [dataset]: 以类别顺序排列的 PGI 数组
    pcds [dataset]: 以类别顺序排列的点云数组
    labels [dataset]: 以类别顺序排列的标签数组
    """
    with h5py.File(dst_path, "w") as fdst:
        fdst.create_dataset("cate_count", data=dataset["cate_count"])
        fdst.create_dataset("cate_name", data=dataset["cate_name"])
        fdst.create_dataset("models_count", data=dataset["models_count"])
        fdst.create_dataset("count_per_cate", data=dataset["count_per_cate"])
        fdst.create_dataset("pgis", data=np.array(dataset["pgis"]))
        fdst.create_dataset("pcds", data=np.array(dataset["pcds"]))
        fdst.create_dataset("labels", data=np.array(dataset["labels"]))
    """ =========================================== ModelNet 系列 ======================================== """
