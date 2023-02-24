import math
from typing import Callable, Dict, List, Optional, Tuple,Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel, PNDMScheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T

from .paint_with_words import (
    pww_load_tools, preprocess, _pil_from_latents, _encode_text_color_inputs)


def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        if isinstance(mask, Image.Image):
            mask = np.array(mask.convert("L"))
            mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image    


def prepare_mask_latents(
        vae, mask, masked_image, batch_size, height, width, 
        dtype, device, generator, do_classifier_free_guidance):
    # resize the mask to latents shape as we concatenate the mask to the latents
    # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
    # and half precision
    mask = F.interpolate(mask, size=(height // 8, width // 8))
    mask = mask.to(device=device, dtype=dtype)
    masked_image = masked_image.to(device=device, dtype=dtype)

    # encode the mask image into latents space so we can concatenate it to the latents
    masked_image_latents = vae.encode(masked_image).latent_dist.sample(generator=generator)
    masked_image_latents = 0.18215 * masked_image_latents

    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    mask = mask.repeat(batch_size, 1, 1, 1)
    masked_image_latents = masked_image_latents.repeat(batch_size, 1, 1, 1)

    mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
    masked_image_latents = (
        torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
    )

    # aligning device to prevent device errors when concating it with the latent model input
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
    return mask, masked_image_latents


@torch.no_grad()
@torch.autocast("cuda")
def paint_with_words_inpaint(
    color_context: Dict[Tuple[int, int, int], str] = {},
    color_map_image: Optional[Image.Image] = None,
    mask_image: Optional[Image.Image] = None,
    init_image: Image.Image = None,
    input_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 0,
    scheduler_type = LMSDiscreteScheduler,
    device: str = "cuda:0",
    weight_function: Callable = lambda w, sigma, qk: 0.1
    * w
    * math.log(sigma + 1)
    * qk.max(),
    local_model_path: Optional[str] = None,
    hf_model_path: Optional[str] = "runwayml/stable-diffusion-inpainting",
    preloaded_utils: Optional[Tuple] = None,
    unconditional_input_prompt: str = "",
    model_token: Optional[str] = None,
    strength: float = 0.5,
):

    vae, unet, text_encoder, tokenizer, scheduler = (
        pww_load_tools(
            device,
            scheduler_type,
            local_model_path=local_model_path,
            hf_model_path=hf_model_path,
            model_token=model_token,
        )
        if preloaded_utils is None
        else preloaded_utils
    )

    width, height = init_image.size
    _, _, encoder_hidden_states, uncond_encoder_hidden_states = \
        _encode_text_color_inputs(text_encoder, tokenizer, device, color_map_image, color_context, input_prompt, unconditional_input_prompt)

    mask, masked_image = prepare_mask_and_masked_image(init_image, mask_image)
    
    scheduler.set_timesteps(num_inference_steps)
    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    t_start = max(num_inference_steps - init_timestep + offset, 0)
    timesteps = scheduler.timesteps[t_start:]
    num_inference_steps = num_inference_steps - t_start
    latent_timestep = timesteps[:1]

    # Latent
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    generator_cpu = torch.manual_seed(seed)
    init_image = preprocess(init_image)
    image = init_image.to(device=device)
    init_latent_dist = vae.encode(image).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)
    init_latents = 0.18215 * init_latents
    noise = torch.randn(init_latents.shape, generator=generator_cpu).to(device)
    init_latents = scheduler.add_noise(init_latents, noise, latent_timestep)
    latents = init_latents

    # Mask image
    mask, masked_image_latents = prepare_mask_latents(
        vae,
        mask,
        masked_image,
        1,
        height,
        width,
        latents.dtype,
        device,
        generator=generator,
        do_classifier_free_guidance=False,
    )

    # Check that sizes of mask, masked image and latents match
    num_channels_latents = latents.shape[1]
    num_channels_mask = mask.shape[1]
    num_channels_masked_image = masked_image_latents.shape[1]
    if num_channels_latents + num_channels_mask + num_channels_masked_image != unet.in_channels:
        raise ValueError(
            f"Incorrect configuration settings! The config of `pipeline.unet`: {unet.config} expects"
            f" {unet.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
            f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
            f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
            " `pipeline.unet` or your `mask_image` or `image` input."
        )

    is_mps = device == "mps"
    for t in tqdm(timesteps):
        # sigma for pww
        step_index = (scheduler.timesteps == t).nonzero().item()
        sigma = scheduler.sigmas[step_index]

        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
        _t = t if not is_mps else t.float()
        encoder_hidden_states.update({
                "SIGMA": sigma,
                "WEIGHT_FUNCTION": weight_function,
            })
        noise_pred_text = unet(
            latent_model_input,
            _t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
        uncond_encoder_hidden_states.update({
                "SIGMA": sigma,
                "WEIGHT_FUNCTION": lambda w, sigma, qk: 0.0,
            })
        noise_pred_uncond = unet(
            latent_model_input,
            _t,
            encoder_hidden_states=uncond_encoder_hidden_states,
        ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    ret_pil_images = _pil_from_latents(vae, latents)

    return ret_pil_images[0]