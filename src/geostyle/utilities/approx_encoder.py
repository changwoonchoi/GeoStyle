
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from .camera import CameraBatch
from .helpers_nvdiffrec import create_scene
from nvdiffrec.src import mesh
from nvdiffrec.src import render
from nvdiffrec.src import util

SDXL_MODEL_LIB = {
    "1.0": "stabilityai/stable-diffusion-xl-base-1.0",
}

@torch.no_grad()
def generate_img_latent_dataset(cfg, base_mesh, n_vert, lgt, mat, output_path):
    glctx = dr.RasterizeGLContext()
    device = torch.device(f"cuda:{cfg.gpu}")

    os.makedirs(output_path, exist_ok=True)

    cams_data = CameraBatch(
        1024,
        [cfg.dist_min, cfg.dist_max],
        [cfg.azim_min, cfg.azim_max],
        [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
        [cfg.fov_min, cfg.fov_max],
        cfg.aug_loc,
        cfg.aug_light,
        cfg.aug_bkg,
        1, 
        rand_solid=True,
        bkg_color=cfg.bkg_color,
    )
    cams = torch.utils.data.DataLoader(cams_data, 1, num_workers=0, pin_memory=True)
    sdxl_version = cfg.sd_model.split("_")[-1]

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_LIB[sdxl_version],
        resume_download=True,
        add_watermark=False,
    ).to(device)
    vae = pipeline.vae.eval()
    halfres = False

    m = mesh.Mesh(
        n_vert,
        base_mesh.t_pos_idx,
        material=mat,
        base=base_mesh,
    )
    base_mesh = create_scene([m.eval()], sz=1024)
    base_mesh = mesh.auto_normals(base_mesh)
    base_mesh = mesh.compute_tangents(base_mesh)
    
    for i in tqdm(range(cfg.num_latent_img_dataset), leave=False):
        params_camera = next(iter(cams))
        for key in params_camera:
            params_camera[key] = params_camera[key].to(device)
        
        if isinstance(lgt, list):
            _lgt = lgt[i % len(lgt)]
        else:
            _lgt = lgt
        out_buffer = render.render_mesh(
            glctx,
            base_mesh.eval(params_camera),
            params_camera['mvp'],
            params_camera['campos'],
            _lgt,
            [1024, 1024],
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs'],
        )
        rendered_img = out_buffer['shaded']
        rendered_img = util.rgb_to_srgb(rendered_img)
        rendered_img = rendered_img[..., :3]  # Remove alpha channel
        rendered_img = rendered_img.permute(0, 3, 1, 2)
        rendered_img = rendered_img.clamp(0, 1)
        
        _img = rendered_img * 2. - 1.
        if halfres:
            _img = _img.to(torch.float16)
        posterior = vae.encode(_img).latent_dist
        latent = posterior.sample()
        latent = latent * vae.config.scaling_factor

        assert latent.shape[0] == 1, "Batch size must be 1"

        rendered_img_128 = F.interpolate(
            rendered_img,
            (128, 128),
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )

        img_np = rendered_img[0].mul(255).permute(1, 2, 0).to('cpu', dtype=torch.uint8).numpy()
        img_128_np = rendered_img_128[0].mul(255).permute(1, 2, 0).to('cpu', dtype=torch.uint8).numpy()

        im = Image.fromarray(img_np)
        im_128 = Image.fromarray(img_128_np)

        torch.save(latent.data[0].cpu().permute(1, 2, 0), os.path.join(output_path, f'latent_{i:06d}.pt'))
        im.save(os.path.join(output_path, f'img_{i:06d}.png'))
        im_128.save(os.path.join(output_path, f'img_128_{i:06d}.png'))


def fit_approx_encoder(cfg, img_latent_path, output_path):
    """Fit an approximate encoder using least squares.
    Args:
        cfg: Configuration object.
        img_latent_path: Path to the image and latent files.
        output_path: Path to save the fitted encoder.
    Returns:
        path to the fitted encoder.
    """
    img_paths = sorted(glob.glob(img_latent_path + '/img_128_*.png'))
    latent_paths = sorted(glob.glob(img_latent_path + '/*.pt'))

    imgs = []
    latents = []

    for img_path, latent_path in zip(img_paths, latent_paths):
        # Load image and latent
        img = torch.from_numpy(np.array(Image.open(img_path)))
        img = img / 255.
        latent = torch.load(latent_path)

        imgs.append(img.view(-1, 3))  # (64*64, 3)
        latents.append(latent.view(-1, 4))  # (64*64, 4)
    
    imgs = torch.cat(imgs, dim=0)  # (1024*64*64, 3)
    imgs_aug = torch.cat([imgs, torch.ones_like(imgs[:, :1])], dim=1)  # (1024*64*64, 4)
    latents = torch.cat(latents, dim=0)  # (1024*64*64, 4)
    latents = latents.to(imgs.dtype)

    # Least square, solve Imgs @ X = Latents
    X = torch.linalg.lstsq(imgs, latents).solution  # (3, 4)
    X_aug = torch.linalg.lstsq(imgs_aug, latents).solution  # (4, 4)
    torch.save(X.clone(), os.path.join(output_path, "img2latent.pt"))
    print("image to latent matrix, linear-only")
    print(X)
    print(f"error: {(imgs @ X - latents).norm(dim=1).mean()}")

    approx_encoder_path = os.path.join(output_path, "img2latent_aug.pt")
    torch.save(X_aug.clone(), approx_encoder_path)
    print("image to latent matrix, linear + constant")
    print(X_aug)
    print(f"error: {(imgs_aug @ X_aug - latents).norm(dim=1).mean()}")
    return approx_encoder_path

