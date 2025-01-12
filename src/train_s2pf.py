import numpy as np
import torch

from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from dataset import ParaTrainLoader_PatchCollection
from parametrization_comps import S2PF
from utils import repulsion_loss, index_points, fps


CONFIG = {
    "data_dir": "../data/PatchCollection100",
    "batch_size": 64,
    "lr_max": 1e-4,
    "lr_min": 1e-4,
    "weight_decay": 1e-8,
    "nepochs": 100,
    "ckpt_dir": "../ckpt/comps/s2pf.pth",
}


""" Train patch with size of 100 """
# Loader
loader = DataLoader(ParaTrainLoader_PatchCollection(CONFIG["data_dir"]),
    batch_size=CONFIG["batch_size"], drop_last=True)
# Model
model = S2PF(rescale_ebd=False)
# Optimizer & Scheduler
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["lr_max"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
    T_max=CONFIG["nepochs"], eta_min=CONFIG["lr_min"])
# Loop
for epoch in range(1, CONFIG["nepochs"] + 1):
    model.cuda().train()
    total_loss = 0
    nprocessed = 0
    loop = tqdm(loader, total=len(loader), leave=True, ncols=150, desc=f"EPOCH-{epoch:06}")
    for points in loop:
        optimizer.zero_grad()
        batch_size, num_points = points.size(0), points.size(1)
        pts = points.cuda()
        ebd = model(pts)
        loss = repulsion_loss(ebd, 1 / (np.sqrt(num_points) - 1))
        nprocessed += batch_size
        total_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"epoch: {epoch:05}, rep loss: {np.around(total_loss/nprocessed, 8)}")
    torch.save(model.cpu().state_dict(), CONFIG["ckpt_dir"])

""" Train patch with size from 8 to 64 """
# Loader
loader = DataLoader(ParaTrainLoader_PatchCollection(CONFIG["data_dir"]),
    batch_size=CONFIG["batch_size"], drop_last=True)
# Model
model = S2PF(rescale_ebd=False)
model.load_state_dict(torch.load(CONFIG["ckpt_dir"]))
# Optimizer & Scheduler
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["lr_max"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
    T_max=CONFIG["nepochs"], eta_min=CONFIG["lr_min"])
# Loop
for epoch in range(1, CONFIG["nepochs"] + 1):
    model.cuda().train()
    total_loss = 0
    nprocessed = 0
    loop = tqdm(loader, total=len(loader), leave=True, ncols=150, desc=f"EPOCH-{epoch:06}")
    for points in loop:
        optimizer.zero_grad()
        batch_size, num_points = points.size(0), points.size(1)
        pts = points.cuda()
        np.random.seed()
        K = np.random.randint(8, 64 + 1)
        with torch.no_grad():
            pts = index_points(pts, fps(pts, K))
        ebd = model(pts)
        loss = repulsion_loss(ebd, (1 / (np.sqrt(num_points) - 1) * 0.5))
        nprocessed += batch_size
        total_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"epoch: {epoch:05}, rep loss: {np.around(total_loss/nprocessed, 8)}")
    torch.save(model.cpu().state_dict(), CONFIG["ckpt_dir"])
