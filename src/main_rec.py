import numpy as np
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from dataset import ModelNet_REC

from parametrization_comps import G2SD
from utils import *
from networks import FlatNetRec


N = 2048
N_G = 256
N_C = 40
k = 7
n_G = int(N_G ** 0.5)
K = (k ** 2)
M = N_G * K
m = n_G * k

train_bs = 64
train_set = ModelNet_REC("../data/ModelNet40/PC_2048/modelnet40_2048_256_49_train_112.h5")
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, drop_last=True)
test_bs = 64
test_set = ModelNet_REC(f"../data/ModelNet40/PC_2048/modelnet40_2048_256_49_test_112.h5")
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, drop_last=False)

# training
g2sd = G2SD(N_G)
g2sd.load_state_dict(torch.load("../ckpt/comps/g2sd.pth"))
g2sd.eval().cuda()
net = FlatNetRec(N_G, N_C, K).cuda()
max_lr = 1e-4
min_lr = 1e-5
num_epc = 100
optimizer = optim.AdamW(net.parameters(), lr=max_lr, betas=(0.5, 0.9))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)


for epc in range(1, num_epc + 1):
    net.train()
    loop = tqdm(train_loader, desc=f"Rec{epc:04}", leave=True, ncols=150)
    for (pgi, pcd) in loop:
        batch_size = pgi.size(0)
        optimizer.zero_grad()
        pgi, pcd = pgi.cuda(), pcd.cuda()
        coarse_gt = g2sd(pcd)
        coarse_rec = net(pgi)
        loss = chamfer_distance_cuda(coarse_rec, coarse_gt) + F.l1_loss(coarse_rec, coarse_gt)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    scheduler.step()
    if epc % 10 == 0:
        torch.save(net.state_dict(), f"../ckpt/tasks/rec/flatnet_rec_{N}_modelnet40.pth")


# testing
net = FlatNetRec(N_G, N_C, K).cuda().eval()
net.load_state_dict(torch.load(f"../ckpt/tasks/rec/flatnet_rec_{N}_modelnet40.pth"))
loop = tqdm(test_loader, leave=True, ncols=150)
total_cd = 0
count = 0
for (pgi, pcd) in loop:
    count += pgi.size(0)
    pgi, pcd = pgi.cuda(), pcd.cuda()
    coarse_rec = net(pgi)
    pgi_as_img = pgi.permute(0, 2, 1).contiguous().view(-1, 3, m, m)
    blocks_for_sample, _ = get_pgi_blocks(pgi_as_img, N_G, N_C, 7)  # [B,256,7,7,3]
    complete_pcd = sample_points_from_blocks_topk(coarse_rec, blocks_for_sample, 8)
    print(complete_pcd.shape)
    total_cd += chamfer_distance_cuda(complete_pcd, pcd)
print(f"Average CD in test is {total_cd.item() * 1000 / count}")
