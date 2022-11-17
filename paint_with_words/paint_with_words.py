import functools
import inspect
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from dotenv import load_dotenv
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def _img_importance_flatten(img: torch.tensor, ratio: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        scale_factor=1 / ratio,
        mode="bilinear",
        align_corners=True,
    ).squeeze()


def _pil_from_latents(vae, latents):
    _latents = 1 / 0.18215 * latents.clone()
    image = vae.decode(_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    ret_pil_images = [Image.fromarray(image) for image in images]

    return ret_pil_images


def inj_forward(self, hidden_states, context=None, mask=None):

    if context is not None:
        context_tensor = context["CONTEXT_TENSOR"]

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
        f: Callable = context["WEIGHT_FUNCTION"]
        w = context[f"CROSS_ATTENTION_WEIGHT_{attention_size_of_img}"]
        sigma = context["SIGMA"]

        cross_attention_weight = f(w, sigma, attention_scores)

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

    model_path = local_model_path if local_model_path is not None else hf_model_path
    local_path_only = local_model_path is not None
    print(model_path)
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only,
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
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


def _image_context_seperator(
    img: Image.Image, color_context: dict, _tokenizer
) -> List[Tuple[List[int], torch.Tensor]]:

    ret_lists = []

    if img is not None:
        w, h = img.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        img = img.resize((w, h), resample=PIL.Image.LANCZOS)

        for color, v in color_context.items():
            v, f = v.split(",")
            f = float(f)
            v_input = _tokenizer(
                v,
                max_length=_tokenizer.model_max_length,
                truncation=True,
            )
            v_as_tokens = v_input["input_ids"][1:-1]
            if isinstance(color, str):
                r, g, b = color[1:3], color[3:5], color[5:7]
                color = (int(r, 16), int(g, 16), int(b, 16))

            img_where_color = (np.array(img) == color).all(axis=-1)

            if not img_where_color.sum() > 0:
                print(f"Warning : not a single color {color} not found in image")

            img_where_color = torch.tensor(img_where_color, dtype=torch.float32) * f

            ret_lists.append((v_as_tokens, img_where_color))
    else:
        w, h = 512, 512

    if len(ret_lists) == 0:
        ret_lists.append(([-1], torch.zeros((w, h), dtype=torch.float32)))
    return ret_lists, w, h


def _tokens_img_attention_weight(
    img_context_seperated, tokenized_texts, ratio: int = 8
):

    token_lis = tokenized_texts["input_ids"][0].tolist()
    w, h = img_context_seperated[0][1].shape

    w_r, h_r = w // ratio, h // ratio

    ret_tensor = torch.zeros((w_r * h_r, len(token_lis)), dtype=torch.float32)

    for v_as_tokens, img_where_color in img_context_seperated:
        is_in = 0

        for idx, tok in enumerate(token_lis):
            if tok in v_as_tokens:
                is_in = 1
                ret_tensor[:, idx] += _img_importance_flatten(
                    img_where_color, ratio
                ).reshape(-1)

        if not is_in == 1:
            print(f"Warning ratio {ratio} : token {v_as_tokens} not found in text")

    return ret_tensor


@torch.no_grad()
@torch.autocast("cuda")
def paint_with_words(
    color_context: Dict[Tuple[int, int, int], str] = {},
    color_map_image: Optional[Image.Image] = None,
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
    hf_model_path: Optional[str] = "CompVis/stable-diffusion-v1-4",
    preloaded_utils: Optional[Tuple] = None,
    unconditional_input_prompt: str = "",
    model_token: Optional[str] = os.environ.get("HF_TOKEN"),
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

    generator = torch.manual_seed(seed)

    text_input = tokenizer(
        [input_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    seperated_word_contexts, width, height = _image_context_seperator(
        color_map_image, color_context, tokenizer
    )

    cross_attention_weight_4096 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=8
    ).to(device)
    cross_attention_weight_1024 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=16
    ).to(device)
    cross_attention_weight_256 = _tokens_img_attention_weight(
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

    latents = torch.randn(
        (1, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):
        step_index = (scheduler.timesteps == t).nonzero().item()
        sigma = scheduler.sigmas[step_index]

        latent_model_input = scheduler.scale_model_input(latents, t)

        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": cond_embeddings,
                "CROSS_ATTENTION_WEIGHT_4096": cross_attention_weight_4096,
                "CROSS_ATTENTION_WEIGHT_1024": cross_attention_weight_1024,
                "CROSS_ATTENTION_WEIGHT_256": cross_attention_weight_256,
                "CROSS_ATTENTION_WEIGHT_64": cross_attention_weight_64,
                "SIGMA": sigma,
                "WEIGHT_FUNCTION": weight_function,
            },
        ).sample

        latent_model_input = scheduler.scale_model_input(latents, t)

        noise_pred_uncond = unet(
            latent_model_input,
            t,
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
