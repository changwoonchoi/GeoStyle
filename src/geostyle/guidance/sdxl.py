"""
sdxl.py

A wrapper around Stable Diffusion XL for computing SDS loss and its variants.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Literal, Optional, Union
from math import sqrt
from diffusers import (
    DDIMScheduler,
    StableDiffusionXLPipeline,
)
from jaxtyping import Float, Int64, jaxtyped
from typeguard import typechecked
from .base_guidance import Guidance

SDXL_MODEL_LIB = {
    "1.0": "stabilityai/stable-diffusion-xl-base-1.0",
}

class StableDiffusionXLGuidance(Guidance):
    
    model_type: Literal["stabilityai/stable-diffusion-xl-base-1.0"]
    """Type of the Stable Diffusion model to use."""
    min_step_t: float
    """Minimum diffusion timestep."""
    max_step_t: float
    """Maximum diffusion timestep."""
    use_half_precision: bool
    """Whether to use half-precision floating point."""
    device: torch.device
    """Device to use for computation."""
    lora_path: Optional[Union[str, Path]]
    """Path to the pre-trained LoRA weights."""
    lora_scale: float
    """Scaling factor for the LoRA weights."""
    grad_clamp_val: Optional[float]
    """Value to clamp the gradients to."""
    approx_encoder: Optional[int]
    """Approximate SD's vae encoder with a linear layer."""
    approx_encoder_type: Optional[str]
    approx_encoder_path: Optional[Union[str, Path]]
    """Path to the approximated encoder weights."""

    
    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        model_type: str = "1.0",
        min_step_t: float = 0.02,
        max_step_t: float = 0.98,
        use_half_precision: bool = False,
        device: torch.device = torch.device("cuda"),
        lora_path: Optional[Union[str, Path]] = None,
        lora_scale: float = 1.0,
        grad_clamp_val: Optional[float] = None,
        approx_encoder = None,
        approx_encoder_type = None,
        approx_encoder_path = None
    ) -> None:
        """Constructor"""
        super().__init__()

        self.model_type = SDXL_MODEL_LIB[model_type]
        self.min_step_t = min_step_t
        self.max_step_t = max_step_t
        self.use_half_precision = use_half_precision
        self.device = device
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.grad_clamp_val = grad_clamp_val
        self.approx_encoder = approx_encoder and approx_encoder_path is not None
        self.approx_encoder_type = approx_encoder_type
        self.approx_encoder_path = approx_encoder_path

        self._build_model()

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        prompt: str,
        image: Float[torch.Tensor, "B C H W"] = None,
        latent: Optional[Float[torch.Tensor, "B 4 128 128"]] = None,
        cfg_scale: float = 7.5,
        step: Optional[Int64[torch.Tensor, "1"]] = None
    ):
        if latent is None:
            assert image is not None, "Either image or latent must be specified."
            image = F.interpolate(
                image,
                (1024, 1024),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            latent = self.encode_images(image)
        batch_size = latent.shape[0]

        # sample diffusion timestep ~ U(t_min, t_max)
        if step is None:
            step = torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                dtype=torch.long,
                device=self.device,
            )
        assert step.min() >= self.min_step and step.max() <= self.max_step, (
            f"step must be in [{self.min_step}, {self.max_step}]: {step.min()}, {step.max()}"
        )

        # predict gradient
        prompt = [prompt]
        grad = self.compute_img_grad(latent, prompt, cfg_scale, step)
        grad = torch.nan_to_num(grad)
        if not self.grad_clamp_val is None:
            grad = torch.clamp(
                grad,
                -self.grad_clamp_val,
                self.grad_clamp_val,
            )
        
        # compute loss
        target = (latent - grad).detach()
        loss = (0.5 / batch_size) * F.mse_loss(
            latent,
            target,
            reduction="mean",
        )

        return loss

    @jaxtyped(typechecker=typechecked)
    def compute_img_grad(
        self,
        latent: Float[torch.Tensor, "B 4 128 128"],
        prompt: List[str],
        cfg_scale: float,
        step: Int64[torch.Tensor, "B"],
    ) -> Float[torch.Tensor, "B 4 128 128"]:
        """
        Computes Score Distillation Sampling gradient.
        """
        # encode prompts
        batch_size = latent.shape[0]
        text_embeddings, negative_text_embeddings, pooled_text_embeddings, negative_pooled_text_embeddings = self.encode_prompts(prompt, num_images_per_prompt=batch_size)

        # predict noise
        with torch.no_grad():
            noise = torch.randn_like(latent)
            latents_noisy = self.scheduler.add_noise(latent, noise, step)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            eps = self.forward_unet(
                latent_model_input,
                torch.cat([step] * 2),
                text_embeddings=text_embeddings,
                negative_text_embeddings=negative_text_embeddings,
                pooled_text_embeddings=pooled_text_embeddings,
                negative_pooled_text_embeddings=negative_pooled_text_embeddings
            )

        # compute the Classifier-Free Guidance
        eps_uncond, eps_text = eps.chunk(2)
        eps = eps_text + cfg_scale * (eps_text - eps_uncond)

        # compute SDS gradient
        weight = (1 - self.alphas[step]).view(-1, 1, 1, 1)
        grad = weight * (eps - noise)

        return grad

    @jaxtyped(typechecker=typechecked)
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self,
        image: Float[torch.Tensor, "B C H W"]
    ) -> Float[torch.Tensor, "B 4 128 128"]:
        """
        Encodes RGB images into latents using pre-trained VAE.
        """
        if self.approx_encoder:
            input_dtype = image.dtype
            image = F.interpolate(
                image,
                (128, 128),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).permute(0, 2, 3, 1)
            if self.approx_encoder_type == "linear":
                latent = torch.matmul(image, self.encoder).permute(0, 3, 1, 2)
            elif self.approx_encoder_type == "linear+constant":
                latent = torch.matmul(torch.cat([image, torch.ones_like(image[..., :1])], dim=-1), self.encoder).permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Invalid approx_encoder_type: {self.approx_encoder_type}")
            return latent.to(input_dtype)
        else:
            input_dtype = image.dtype
            image = image * 2.0 - 1.0
            posterior = self.vae.encode(image.to(self.weight_dtype)).latent_dist
            latent = posterior.sample() * self.vae.config.scaling_factor
            return latent.to(input_dtype)

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def encode_prompts(
        self,
        prompt: List[str],
        num_images_per_prompt: int = 1,
    ):
        """
        Encodes the input text prompt.
        """
        lora_scale = self.lora_scale
        if self.lora_path is None:
            lora_scale = None
        text_embedding, negative_text_embeddings, pooled_text_embedding, negative_pooled_text_embeddings = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=[""] * len(prompt),
            lora_scale=lora_scale,
        )

        return text_embedding, negative_text_embeddings, pooled_text_embedding, negative_pooled_text_embeddings
    
    
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim):
        # adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


    @jaxtyped(typechecker=typechecked)
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latent: Float[torch.Tensor, "* 4 128 128"],
        step: Int64[torch.Tensor, "*"],
        text_embeddings,
        negative_text_embeddings,
        pooled_text_embeddings,
        negative_pooled_text_embeddings,
    ) -> Float[torch.Tensor, "* 4 128 128"]:
        """
        Forward pass of Stable Diffusion U-Net.
        """
        input_dtype = latent.dtype
        cross_attention_kwargs = None
        if self.lora_path is not None:
            cross_attention_kwargs = {
                "scale": self.lora_scale,
            }
        add_time_ids = self._get_add_time_ids(
            (1024, 1024),
            (0, 0),
            (1024, 1024),
            latent.dtype,
            pooled_text_embeddings.shape[-1]
        ).to(self.device)

        add_time_ids = add_time_ids.repeat(len(step), 1)
        add_text_embeds = torch.cat([negative_pooled_text_embeddings, pooled_text_embeddings], dim=0).to(self.device)

        unet_added_conditions = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }
        unet_output = self.unet(
            latent.to(self.weight_dtype),
            step.to(self.weight_dtype),
            encoder_hidden_states=torch.cat([negative_text_embeddings, text_embeddings], dim=0).to(self.device).to(self.weight_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=unet_added_conditions
        ).sample.to(input_dtype)

        return unet_output

    @jaxtyped(typechecker=typechecked)
    def _build_model(self) -> None:
        # set precision
        self.weight_dtype = torch.float32
        if self.use_half_precision:
            self.weight_dtype = torch.float16

        # load Stable Diffusion
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_type,
            torch_dtype=self.weight_dtype,
            resume_download=True,
            add_watermark=False,
        ).to(self.device)

        # configure scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_type,
            subfolder="scheduler",
            torch_dtype=self.weight_dtype,
        )

        # configure timestep scheduling
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.min_step_t * self.num_train_timesteps)
        self.max_step = int(self.max_step_t * self.num_train_timesteps)
        self.alphas: Float[torch.Tensor, "num_train_timesteps"] = (
            self.scheduler.alphas_cumprod.to(self.device)
        )

        # load pre-trained LoRA weights if specified
        if self.lora_path is not None:
            assert self.lora_scale >= 0.0 and self.lora_scale <= 1.5, (
                f"LoRA scale must be in [0, 1.5]: {self.lora_scale}"
            )
            self._load_lora_weights()
        
        # extract modules from Stable Diffusion pipeline
        if self.approx_encoder:
            self.encoder = torch.load(self.approx_encoder_path).to(self.device)  # (3, 4)
        else:
            self.vae = self.pipeline.vae.eval()
        self.unet = self.pipeline.unet.eval()

        # freeze the modules
        if not self.approx_encoder:
            for param in self.vae.parameters():
                param.requires_grad_(False)
        for param in self.unet.parameters():
            param.requires_grad_(False)

        print("="*80)
        print("[!] Built Stable Diffusion XL Guidance")
        print("="*80)


    @jaxtyped(typechecker=typechecked)
    def _load_lora_weights(self) -> None:
        if isinstance(self.lora_path, str):
            self.lora_path = Path(self.lora_path)
        assert self.lora_path.exists(), (
            f"LoRA weights not found: {str(self.lora_path)}"
        )
        assert self.pipeline is not None, (
            "Stable Diffusion pipeline must be loaded before loading LoRA weights."
        )
        assert isinstance(self.pipeline, StableDiffusionXLPipeline), (
            "Stable Diffusion pipeline must be of type StableDiffusionXLPipeline."
        )

        # load an attach LoRA to the pipeline
        self.pipeline.load_lora_weights(str(self.lora_path))
        self.pipeline = self.pipeline.to(self.device)

        print("="*80)
        print("[!] Loaded LoRA weights")
        print("="*80)
    
