import os
import yaml
import torch
import random
import argparse
import numpy as np

from loop import loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', type=str, required=True)
    parser.add_argument('--output_path', help='Output directory (will be created)', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--gpu', help='GPU index', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--seed', help='Random seed', type=int, default=argparse.SUPPRESS)

    parser.add_argument('--sd_model', help='Stable Diffusion model card', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--text_prompt', help='Target text prompt', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--grad_clamp_val', help='gradieng clamping value for guidance', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--cfg_scale', help='CFG scale', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--use_half_precision', help='Use half precision', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    
    parser.add_argument('--approx_encoder', help='use approximated encoder instead of vae', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--approx_encoder_type', help='type of approximated encoder', type=str, default=argparse.SUPPRESS, choices=['linear', 'linear+constant'])
    parser.add_argument('--approx_encoder_path', help='path of approximated encoder', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--num_latent_img_dataset', help='number of latent images in dataset', type=int, default=argparse.SUPPRESS)

    parser.add_argument('--lora_scale', help='LoRA scale', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--lora_model', help='path of lora weight', type=str, default=argparse.SUPPRESS)

    parser.add_argument('--mesh', help='Path to input mesh', type=str, default=argparse.SUPPRESS)
    
    parser.add_argument('--envlight', help='Path to environment light', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--mtl', help='Path to material file', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--bkg_color', help='background color', type=str, default=argparse.SUPPRESS, choices=["white", "gray", "black"])
    parser.add_argument('--random_textures', help='Randomize textures', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--optimize_textures', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--random_light_rotation_latent_img_dataset', help='Random rotation for environment lighting during approximated encoder fitting', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--random_light_rotation', help='Random rotation for environment lighting', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--num_envlight_rotations', help='Number of random rotations for environment lighting', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--texture-res', nargs=2, type=int, default=[1024, 1024])

    parser.add_argument('--lr', help='Learning rate', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--epochs', help='Number of optimization steps', type=int, default=argparse.SUPPRESS)

    parser.add_argument('--train_res', help='Resolution of render', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--batch_size', help='Number of images rendered at the same time', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--fov_min', help='Minimum camera field of view angle during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--fov_max', help='Maximum camera field of view angle during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dist_min', help='Minimum distance of camera from mesh during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dist_max', help='Maximum distance of camera from mesh during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_min', help='Minimum elevation angle in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_max', help='Maximum elevation angle in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_alpha', help='Alpha parameter for Beta distribution for elevation sampling', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_beta', help='Beta parameter for Beta distribution for elevation sampling', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--azim_min', help='Minimum azimuth angle in degree',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--azim_max', help='Maximum azimuth angle in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aug_loc', help='Offset mesh from center of image?', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--aug_light', help='Augment the direction of light around the camera', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--aug_bkg', help='Augment the background', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--adapt_dist', help='Adjust camera distance to account for scale of shape', type=int, default=argparse.SUPPRESS, choices=[0, 1])

    parser.add_argument('--log_interval', help='Interval for logging, every X epochs',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_interval_im', help='Interval for logging renders image, every X epochs',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_elev', help='Logging elevation angle',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_rot', help='Logging azimuth angle in degree',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_dist', help='Logging distance from object',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_res', help='Logging render resolution',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_fov', help='Logging field of view',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_vertex_color', help='Log vertex color', type=int, default=argparse.SUPPRESS, choices=[0, 1])

    parser.add_argument('--sds_weight', help='Weight for SDS loss', type=float, default=argparse.SUPPRESS)

    parser.add_argument('--regularize_jacobians_weight', help='Weight for jacobian regularization', type=float, default=argparse.SUPPRESS)

    parser.add_argument('--use_cage_loss', help='Use cage loss (coarse stage) during deformation', type=int, default=argparse.SUPPRESS, choices=[0, 1])
    parser.add_argument('--cage_loss_weight', help='Weight for cage loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--segmented_label_path', help='path of segmented label', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--cage_loss_epoch', help='Number of coarse stage steps', type=int, default=argparse.SUPPRESS)

    parser.add_argument('--symmetry_loss_weight', help='Weight for symmetry loss', type=float, default=argparse.SUPPRESS)

    parser.add_argument('--aux_lr', help='Learning rate for optimizing an auxiliary mesh', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aux_sds_weight', help='Weight for SDS loss during auxiliary mesh optimization', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aux_symmetry_loss_weight', help='Weight for symmetry loss during auxiliary mesh optimization', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aux_sampling_method', help='Vertex sampling strategy for constructing an auxiliary mesh', type=str, default=argparse.SUPPRESS, choices=["random", "fps"])
    parser.add_argument('--aux_sampling_ratio', help='Vertex sampling ratio for constructing an auxiliary mesh', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aux_sphere_radius', help='Radius of sphere in auxiliary mesh', type=float, default=argparse.SUPPRESS)


    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
    
    for key in vars(args):
        cfg[key] = vars(args)[key]

    print(yaml.dump(cfg, default_flow_style=False))
    random.seed(cfg['seed'])
    os.environ['PYTHONHASHSEED'] = str(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True

    if cfg['approx_encoder']:
        if cfg['approx_encoder_path'] is None:
            approx_encoder_path = loop(cfg, train_approx_encoder=True)
            cfg['approx_encoder_path'] = approx_encoder_path
            torch.cuda.empty_cache()
            loop(cfg)
        else:
            loop(cfg)
    else:
        loop(cfg)
    print('Done')

if __name__ == '__main__':
    main()

