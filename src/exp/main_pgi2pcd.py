import numpy as np
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from dataset import ModelNet_REC

from parametrization_comps import G2SD
from networks import FlatNet_PGI2PC_FIT_PGI
from utils import earth_mover_distance_cuda, chamfer_distance_cuda


N = 2048
N_G = 256
N_C = 40
k = 7
n_G = int(N_G ** 0.5)
K = (k ** 2)
M = N_G * K
m = n_G * k

train_bs = 32
train_set = ModelNet_REC("../data/ModelNet40/PC_2048/modelnet40_2048_256_49_train_112.h5")
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, drop_last=True)


# training
g2sd = G2SD(N_G)
g2sd.load_state_dict(torch.load("../ckpt/comps/g2sd.pth"))
g2sd.eval().cuda()
net = FlatNet_PGI2PC_FIT_PGI(N_G, N_C, K).cuda()
max_lr = 1e-4
min_lr = 1e-5
num_epc = 100
optimizer = optim.AdamW(net.parameters(), lr=max_lr, betas=(0.5, 0.9))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)
loss_l1 = torch.nn.L1Loss()


for epc in range(1, num_epc + 1):
    net.train()
    loop = tqdm(train_loader, desc=f"Rec{epc:04}", leave=True, ncols=150)
    for (pgi, pcd) in loop:
        batch_size = pgi.size(0)
        optimizer.zero_grad()
        pgi, pcd = pgi.cuda(), pcd.cuda()
        coarse_gt = g2sd(pcd)
        coarse, complete, rec_pgi = net(pgi)
        rec_pgi_loss = loss_l1(rec_pgi, pgi)
        coarse_loss = 0.2 * earth_mover_distance_cuda(coarse, coarse_gt)
        complete_loss = earth_mover_distance_cuda(complete, pcd)
        loss = coarse_loss + complete_loss + rec_pgi_loss
        loss.backward()
        optimizer.step()
        loop.set_postfix(LossCoarse=coarse_loss.item(), LossComplete=complete_loss.item(),
                         LossRecPGI=rec_pgi_loss.item(), LossTotal=loss.item())
    scheduler.step()
    if epc % 10 == 0:
        torch.save(net.state_dict(), f"../temp/ckpt_rec/flatnet_rec_{epc:04}_modelnet40.pth")
