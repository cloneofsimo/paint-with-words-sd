import math
from typing import Callable, Dict, List, Optional, Tuple,Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel, PNDMScheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
import torchvision.transforms as T
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput


from .paint_with_words import (
    PaintWithWord_StableDiffusionPipeline,
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


class PaintWithWord_StableDiffusionInpaintPipeline(PaintWithWord_StableDiffusionPipeline):
    def __init__(self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: LMSDiscreteScheduler,
        safety_checker: None,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker
        )
    
    @classmethod
    def from_pretrained(self, save_dir, **kwargs):
        sd = super().from_pretrained(save_dir, **kwargs)
        self = PaintWithWord_StableDiffusionInpaintPipeline(
            vae=sd.vae,
            text_encoder=sd.text_encoder,
            tokenizer=sd.tokenizer,
            unet=sd.unet,
            scheduler=sd.scheduler,
            safety_checker=sd.safety_checker,
            feature_extractor=sd.feature_extractor,
            requires_safety_checker=sd.requires_safety_checker
        )
        return self

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
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
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Image.Image = None,
        mask_image: Optional[Image.Image] = None,
        color_map_image: Optional[Image.Image] = None,
        color_context: Dict[Tuple[int, int, int], str] = {},
        weight_function: Callable = lambda w, sigma, qk: 0.1 * w * math.log(sigma + 1) * qk.max(),
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 1.0,
        seed: Optional[int] = 0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # text_embeddings = self._encode_prompt(
        #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        # )
        _, text_embeddings_type, seperated_word_contexts, encoder_hidden_states, uncond_encoder_hidden_states = \
            self._encode_text_color_inputs(device, color_map_image, color_context, prompt, negative_prompt)
        
        # 4. Preprocess mask and image
        mask, masked_image = prepare_mask_and_masked_image(image, mask_image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * eta) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        num_inference_steps = num_inference_steps - t_start
        latent_timestep = timesteps[:1]

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     text_embeddings_type,
        #     device,
        #     generator,
        #     latents,
        # )
        # Latent
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generator_cpu = torch.manual_seed(seed)
        image = preprocess(image)
        image = image.to(device=device)
        init_latent_dist = self.vae.encode(image).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents
        noise = torch.randn(init_latents.shape, generator=generator_cpu).to(device)
        init_latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)
        latents = init_latents

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            text_embeddings_type,
            device,
            generator,
            do_classifier_free_guidance=False,
        )

        # 8. Check that sizes of mask, masked image and latents match
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                step_index = (self.scheduler.timesteps == t).nonzero().item()
                sigma = self.scheduler.sigmas[step_index]

                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                # _t = t if not is_mps else t.float()
                encoder_hidden_states.update({
                        "SIGMA": sigma,
                        "WEIGHT_FUNCTION": weight_function,
                    })
                noise_pred_text = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                uncond_encoder_hidden_states.update({
                        "SIGMA": sigma,
                        "WEIGHT_FUNCTION": lambda w, sigma, qk: 0.0,
                    })
                noise_pred_uncond = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=uncond_encoder_hidden_states,
                ).sample

                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 11. Post-processing
        image = self.decode_latents(latents)

        # 12. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings_type)
        has_nsfw_concept = False

        # 13. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
