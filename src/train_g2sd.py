import numpy as np
import torch

from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from dataset import ParaTrainLoader_ShapeNetCore
from parametrization_comps import G2SD
from utils import chamfer_distance_cuda


CONFIG = {
    "data_dir": "../data/ShapeNetCore256",
    "batch_size": 128,
    "lr_max": 5e-4,
    "lr_min": 5e-6,
    "weight_decay": 1e-8,
    "nepochs": 500,
    "ckpt_dir": "../ckpt/comps/g2sd.pth",
}


# Loader
loader = DataLoader(ParaTrainLoader_ShapeNetCore(CONFIG["data_dir"], "rot_so3"),
    batch_size=CONFIG["batch_size"], drop_last=True)
# Model
model = G2SD(num_grids=256)  # 16 by 16
# Optimizer & Scheduler
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["lr_max"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
    T_max=CONFIG["nepochs"], eta_min=CONFIG["lr_min"])
# Loop
for epoch in range(1, CONFIG["nepochs"] + 1):
    loop = tqdm(loader, total=len(loader), leave=True, ncols=150, desc=f"EPOCH-{epoch:06}")
    total_loss = 0
    nprocessed = 0
    model.cuda().train()
    for points, _ in loop:
        optimizer.zero_grad()
        batch_size = points.size(0)
        pts = points.float().cuda()
        rec_pts = model(pts)
        loss = chamfer_distance_cuda(pts, rec_pts)        
        nprocessed += batch_size
        total_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"epoch: {epoch:05}, cd loss: {np.around(total_loss/nprocessed, 8)}")
    torch.save(model.cpu().state_dict(), CONFIG["ckpt_dir"])
