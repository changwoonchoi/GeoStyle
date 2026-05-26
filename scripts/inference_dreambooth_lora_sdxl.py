import argparse
from pathlib import Path
import torch
import os
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from geometric_stylization.guidance.blora_utils import BLOCKS, filter_lora, scale_lora


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, required=True, help="prompt"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--LoRA", type=str, default=None, help="path for the LoRA"
    )
    parser.add_argument(
        "--lora_scale", type=float, default=1., help="lora scale"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")

    lora_path = Path(args.LoRA)
    assert lora_path.exists(), f"LoRA path {lora_path} does not exist"

    pipeline.load_lora_weights(str(lora_path))

    cross_attention_kwargs = {"scale": args.lora_scale}

    # Generate
    images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt, cross_attention_kwargs=cross_attention_kwargs).images

    os.makedirs(args.output_path, exist_ok=True)
    # Save
    for i, img in enumerate(images):
        img.save(f'{args.output_path}/{args.prompt}_{i}.jpg')