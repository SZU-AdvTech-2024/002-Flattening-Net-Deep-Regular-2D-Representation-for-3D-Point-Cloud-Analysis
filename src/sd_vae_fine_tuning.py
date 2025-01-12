import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from torchvision import transforms
from torch import optim

from dataset import ModelNet_REC


CONFIG = {
    "batch_size": 32,
    "num_epoch": 100,
    "learning_rate": 1e-4,
    "betas": (0.9, 0.999),
    "save_interval": 10,
}

N = 2048
N_G = 256
N_C = 40
k = 7
n_G = int(N_G ** 0.5)
K = (k ** 2)
M = N_G * K
m = n_G * k


set = ModelNet_REC("../data/ModelNet40/PC_2048/modelnet40_2048_256_49_train_112.h5")
loader = DataLoader(set, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
sd_vae = AutoencoderKL.from_pretrained("../pretrain/sd-vae-ft-mse").cuda()
optimizer = optim.Adam(sd_vae.parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["betas"])


def encode_then_decode(vae, images):
    scaling_factor =vae.config.scaling_factor
    # images = transforms.Normalize(mean=.5, std=.5)(images)
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * scaling_factor
    images = vae.decode(latents / scaling_factor, return_dict=False)[0]
    # images = (images / 2 + .5).clamp(0, 1)
    return images


for epoch in range(1, CONFIG["num_epoch"] + 1):
    loop = tqdm(loader, desc=f"EPOCH: {epoch:06}", leave=True, ncols=150)
    for pgi, _ in loop:
        optimizer.zero_grad()
        pgi_as_img = pgi.permute(0, 2, 1).contiguous().view(-1, 3, m, m).cuda()  # [B,3,H,W]
        rec_img = encode_then_decode(sd_vae, pgi_as_img)
        loss = F.mse_loss(rec_img, pgi_as_img)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    if epoch % CONFIG["save_interval"] == 0:
        torch.save(sd_vae.state_dict(), f"../temp/ckpt_rec/sd_vae_fine_tuning_{epoch:04}.pth")
