from typing import Dict

import pytest
import torch
from diffusers.schedulers import LMSDiscreteScheduler
from PIL import Image

from paint_with_words.helper.aliases import RGB
from paint_with_words.pipelines import PaintWithWordsPipeline
from paint_with_words.pipelines.paint_with_words_pipeline import SeparatedImageContext


@pytest.fixture
def model_name() -> str:
    return "CompVis/stable-diffusion-v1-4"


@pytest.fixture
def gpu_device() -> str:
    return "cuda"


EXAMPLE_SETTING_1 = {
    "color_context": {
        (0, 0, 0): "cat,1.0",
        (255, 255, 255): "dog,1.0",
        (13, 255, 0): "tree,1.5",
        (90, 206, 255): "sky,0.2",
        (74, 18, 1): "ground,0.2",
    },
    "color_map_image_path": "contents/example_input.png",
    "input_prompt": "realistic photo of a dog, cat, tree, with beautiful sky, on sandy ground",
    "output_img_path": "contents/output_cat_dog.png",
}

EXAMPLE_SETTING_2 = {
    "color_context": {
        (0, 0, 0): "dog,1.0",
        (255, 255, 255): "cat,1.0",
        (13, 255, 0): "tree,1.5",
        (90, 206, 255): "sky,0.2",
        (74, 18, 1): "ground,0.2",
    },
    "color_map_image_path": "contents/example_input.png",
    "input_prompt": "realistic photo of a dog, cat, tree, with beautiful sky, on sandy ground",
    "output_img_path": "contents/output_dog_cat.png",
}


EXAMPLE_SETTING_3 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_image_path": "contents/aurora_2.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
    "output_img_path": "contents/aurora_2_output.png",
}

EXAMPLE_SETTING_4 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_image_path": "contents/aurora_1.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
    "output_img_path": "contents/aurora_1_output.png",
}

EXAMPLES = [
    (
        EXAMPLE["color_context"],
        EXAMPLE["color_map_image_path"],
        EXAMPLE["input_prompt"],
    )
    for EXAMPLE in [
        EXAMPLE_SETTING_1,
        EXAMPLE_SETTING_2,
        EXAMPLE_SETTING_3,
        EXAMPLE_SETTING_4,
    ]
]


@pytest.mark.parametrize(
    argnames="color_context, color_map_image_path, input_prompt,",
    argvalues=EXAMPLES,
)
def test_pipeline(
    model_name: str,
    color_context: Dict[RGB, str],
    color_map_image_path: str,
    input_prompt: str,
    gpu_device: str,
):

    pipe = PaintWithWordsPipeline.from_pretrained(
        model_name,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    assert isinstance(pipe.scheduler, LMSDiscreteScheduler), type(pipe.scheduler)
    pipe.safety_checker = None  # disable the safety checker
    pipe.to(gpu_device)

    # generate latents with seed-fixed generator
    generator = torch.manual_seed(0)
    latents = torch.randn((1, 4, 64, 64), generator=generator)

    # load color map image
    color_map_image = Image.open(color_map_image_path).convert("RGB")

    with torch.autocast("cuda"):
        image = pipe(
            prompt=input_prompt,
            color_context=color_context,
            color_map_image=color_map_image,
            latents=latents,
            num_inference_steps=30,
        ).images[0]

    image.save("generated_image.png")
    color_map_image.save("color_map.png")


@pytest.mark.parametrize(
    argnames="color_context, color_map_image_path, input_prompt,",
    argvalues=EXAMPLES,
)
def test_separate_image_context(
    model_name: str,
    color_context: Dict[RGB, str],
    color_map_image_path: str,
    input_prompt: str,
):
    pipe = PaintWithWordsPipeline.from_pretrained(model_name)

    color_map_image = Image.open(color_map_image_path).convert("RGB")

    ret_list = pipe.separate_image_context(
        img=color_map_image, color_context=color_context
    )

    for ret in ret_list:
        assert isinstance(ret, SeparatedImageContext)
        assert isinstance(ret.word, str)
        assert isinstance(ret.token_ids, list)
        assert isinstance(ret.color_map_th, torch.Tensor)

        token_ids = pipe.tokenizer(
            ret.word,
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        ).input_ids
        assert ret.token_ids == token_ids


@pytest.mark.parametrize(
    argnames="color_context, color_map_image_path, input_prompt,",
    argvalues=EXAMPLES,
)
def test_calculate_tokens_image_attention_weight(
    model_name: str,
    color_context: Dict[RGB, str],
    color_map_image_path: str,
    input_prompt: str,
):
    pipe = PaintWithWordsPipeline.from_pretrained(model_name)

    color_map_image = Image.open(color_map_image_path).convert("RGB")
    w, h = color_map_image.size

    separated_image_context_list = pipe.separate_image_context(
        img=color_map_image, color_context=color_context
    )

    cross_attention_weight_8 = pipe.calculate_tokens_image_attention_weight(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=8,
    )
    assert cross_attention_weight_8.size() == (
        int((w * 1 / 8) * (h * 1 / 8)),
        pipe.tokenizer.model_max_length,
    )

    cross_attention_weight_16 = pipe.calculate_tokens_image_attention_weight(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=16,
    )
    assert cross_attention_weight_16.size() == (
        int((w * 1 / 16) * (h * 1 / 16)),
        pipe.tokenizer.model_max_length,
    )

    cross_attention_weight_32 = pipe.calculate_tokens_image_attention_weight(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=32,
    )
    assert cross_attention_weight_32.size() == (
        int((w * 1 / 32) * (h * 1 / 32)),
        pipe.tokenizer.model_max_length,
    )

    cross_attention_weight_64 = pipe.calculate_tokens_image_attention_weight(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=64,
    )
    assert cross_attention_weight_64.size() == (
        int((w * 1 / 64) * (h * 1 / 64)),
        pipe.tokenizer.model_max_length,
    )
