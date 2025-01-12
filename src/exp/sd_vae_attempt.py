import torch

from typing import Union
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from torchvision.utils import save_image

vae = AutoencoderKL.from_pretrained("../pretrain/sd-vae-ft-mse").to("cuda")
vae.load_state_dict(torch.load("../temp/ckpt_rec/sd_vae_fine_tuning_0020.pth"))
img = Image.open("../temp/visualization/pgi_gt_2.png")

@torch.no_grad()
def image2latent(vae, image: Union[Image.Image, torch.Tensor]):
    if isinstance(image, Image.Image):
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=.5, std=.5)
        ])(image).unsqueeze(0).to("cuda")

    latent = vae.encode(image).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    return latent

@torch.no_grad()
def latent2image(vae, latents):
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    images = (images / 2 + .5).clamp(0, 1)
    return images

img_rec = latent2image(vae, image2latent(vae, img))
save_image(img_rec, "../temp/visualization/pgi_rec_sd_vae_ft_20.png")
