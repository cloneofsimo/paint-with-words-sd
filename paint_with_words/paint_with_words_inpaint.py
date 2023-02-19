import math
from typing import Callable, Dict, List, Optional, Tuple,Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T

from .paint_with_words import (
    # pww_load_tools, 
    _extract_seed_and_sigma_from_context, 
    _image_context_seperator,_blur_image_mask,_tokens_img_attention_weight,
    _get_binary_mask, preprocess, _pil_from_latents)


def pww_load_tools(
    device: str = "cuda:0",
    scheduler_type=LMSDiscreteScheduler,
    local_model_path: Optional[str] = None,
    hf_model_path: Optional[str] = None,
    model_token: Optional[str] = None,
) -> Tuple[
    UNet2DConditionModel,
    CLIPTextModel,
    CLIPTokenizer,
    AutoencoderKL,
    LMSDiscreteScheduler,
]:

    assert (
        local_model_path or hf_model_path
    ), "either local_model_path or hf_model_path must be provided"

    is_mps = device == 'mps'
    dtype = torch.float16 if not is_mps else torch.float32
    model_path = local_model_path if local_model_path is not None else hf_model_path
    local_path_only = local_model_path is not None
    print(model_path)
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        use_auth_token=model_token,
        torch_dtype=dtype,
        local_files_only=local_path_only,
    )

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        use_auth_token=model_token,
        torch_dtype=dtype,
        local_files_only=local_path_only,
    )

    vae.to(device), unet.to(device), text_encoder.to(device)

    for _module in unet.modules():
        if _module.__class__.__name__ == "CrossAttention":
            _module.__class__.__call__ = inj_forward

    scheduler = scheduler_type(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    return vae, unet, text_encoder, tokenizer, scheduler


def inj_forward(self, hidden_states, context=None, mask=None):

    is_dict_format = True
    if context is not None:
        try:
            context_tensor = context["CONTEXT_TENSOR"]
        except:
            context_tensor = context
            is_dict_format = False

    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)

    key = self.to_k(context_tensor)
    value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))

    attention_size_of_img = attention_scores.shape[-2]
    if context is not None:
        if is_dict_format:
            f: Callable = context["WEIGHT_FUNCTION"]
            w = context[f"CROSS_ATTENTION_WEIGHT_{attention_size_of_img}"]
            sigma = context["SIGMA"]

            cross_attention_weight = f(w, sigma, attention_scores)
        else:
            cross_attention_weight = 0.0
    else:
        cross_attention_weight = 0.0

    attention_scores = (attention_scores + cross_attention_weight) * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)

    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states


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
    # generator = generator.to(device=device)

    # encode the mask image into latents space so we can concatenate it to the latents
    # masked_image_latents = vae.encode(masked_image).latent_dist.sample(generator=generator)
    masked_image_latents = vae.encode(masked_image).latent_dist.sample()
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
    scheduler_type=LMSDiscreteScheduler,
    device: str = "cuda:0",
    weight_function: Callable = lambda w, sigma, qk: 0.1
    * w
    * math.log(sigma + 1)
    * qk.max(),
    local_model_path: Optional[str] = None,
    # hf_model_path: Optional[str] = "CompVis/stable-diffusion-v1-4",
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

    text_input = tokenizer(
        [input_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    color_context, extra_seeds, extra_sigmas = _extract_seed_and_sigma_from_context(color_context)
    is_extra_seed, is_extra_sigma = len(extra_seeds) > 0, len(extra_sigmas) > 0
    seperated_word_contexts, width, height = _image_context_seperator(
        color_map_image, color_context, tokenizer
    )
    if is_extra_sigma:
        print('Use extra sigma to smooth mask', extra_sigmas)
        seperated_word_contexts = _blur_image_mask(seperated_word_contexts, extra_sigmas)

    cross_attention_weight_8 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=8
    ).to(device)
    cross_attention_weight_16 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=16
    ).to(device)
    cross_attention_weight_32 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=32
    ).to(device)
    cross_attention_weight_64 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=64
    ).to(device)

    cond_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [unconditional_input_prompt],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
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
    init_image = preprocess(init_image)
    image = init_image.to(device=device)
    init_latent_dist = vae.encode(image).latent_dist
    init_latents = init_latent_dist.sample()
    init_latents = 0.18215 * init_latents
    noise = torch.randn(init_latents.shape).to(device)
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
        cond_embeddings.dtype,
        device,
        generator=torch.manual_seed(seed),
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
        noise_pred_text = unet(
            latent_model_input,
            _t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": cond_embeddings,
                f"CROSS_ATTENTION_WEIGHT_{height * width // (8 * 8)}": cross_attention_weight_8,
                f"CROSS_ATTENTION_WEIGHT_{height * width // (16 * 16)}": cross_attention_weight_16,
                f"CROSS_ATTENTION_WEIGHT_{height * width // (32 * 32)}": cross_attention_weight_32,
                f"CROSS_ATTENTION_WEIGHT_{height * width // (64 * 64)}": cross_attention_weight_64,
                "SIGMA": sigma,
                "WEIGHT_FUNCTION": weight_function,
            },
        ).sample

        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
        noise_pred_uncond = unet(
            latent_model_input,
            _t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": uncond_embeddings,
                "CROSS_ATTENTION_WEIGHT_4096": 0,
                "CROSS_ATTENTION_WEIGHT_1024": 0,
                "CROSS_ATTENTION_WEIGHT_256": 0,
                "CROSS_ATTENTION_WEIGHT_64": 0,
                "SIGMA": sigma,
                "WEIGHT_FUNCTION": lambda w, sigma, qk: 0.0,
            },
        ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    ret_pil_images = _pil_from_latents(vae, latents)

    return ret_pil_images[0]