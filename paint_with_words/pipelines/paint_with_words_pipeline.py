from dataclasses import dataclass
from typing import List, Union

import torch as th
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


@dataclass
class SeparatedImageContext(object):
    word: str
    token_ids: List[int]
    color_map_th: th.Tensor


class PaintWithWordsPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

    def replace_cross_attention(
        self, cross_attention_name: str = "CrossAttention"
    ) -> None:
        for _module in unet.modules():
            if _module.__class__.__name__ == "CrossAttention":
                _module.__class__.__call__ = inj_forward

    def _image_context_seperator(
        img: Image.Image, color_context: dict, _tokenizer
    ) -> List[Tuple[List[int], torch.Tensor]]:

        ret_lists = []

        if img is not None:
            w, h = img.size
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
            img = img.resize((w, h), resample=PIL.Image.LANCZOS)

            for color, v in color_context.items():
                f = v.split(",")[-1]
                v = ",".join(v.split(",")[:-1])
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
                if token_lis[idx : idx + len(v_as_tokens)] == v_as_tokens:
                    is_in = 1

                    # print(token_lis[idx : idx + len(v_as_tokens)], v_as_tokens)
                    ret_tensor[:, idx : idx + len(v_as_tokens)] += (
                        _img_importance_flatten(img_where_color, ratio)
                        .reshape(-1, 1)
                        .repeat(1, len(v_as_tokens))
                    )

            if not is_in == 1:
                print(f"Warning ratio {ratio} : tokens {v_as_tokens} not found in text")

        return ret_tensor
