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
