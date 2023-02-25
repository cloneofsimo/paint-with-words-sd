import math

import dotenv
from PIL import Image

from paint_with_words import PaintWithWord_StableDiffusionRepaintPipeline
import torch


EXAMPLE_SETTING_1 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_img_path": "contents/aurora_1.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
    "output_img_path": "contents/aurora_3_output.png",
    "img_path": "contents/aurora_1_output.png",
    "mask_path": "contents/moon_mask.png",
}

EXAMPLE_SETTING_2 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_img_path": "contents/aurora_3.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
    "output_img_path": "contents/aurora_4_output.png",
    "img_path": "contents/aurora_1_output.png",
    "mask_path": "contents/moon_mask.png",
}

if __name__ == "__main__":

    dotenv.load_dotenv()
    
    settings = EXAMPLE_SETTING_2

    color_map_image = Image.open(settings["color_map_img_path"]).convert("RGB")
    color_context = settings["color_context"]
    input_prompt = settings["input_prompt"]
    init_image = Image.open(settings["img_path"]).convert("RGB")
    mask_image = Image.open(settings["mask_path"])

    use_pipeline = True
    if use_pipeline:
        pipe = PaintWithWord_StableDiffusionRepaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        pipe = pipe.to("cuda")
        generator = torch.Generator(device="cuda")
        generator.manual_seed(81)
        img = pipe(
                prompt=input_prompt,
                image=init_image,
                color_context=color_context,
                color_map_image=color_map_image,
                mask_image=mask_image,
                num_inference_steps=150,
                guidance_scale=7.5,
                seed=81,           
                weight_function=lambda w, sigma, qk: 0.15 * w * math.log(1 + sigma) * qk.max(),
                eta=1.0,
                generator=generator
        ).images[0]
    else:
        img = paint_with_words_inpaint(
            color_context=color_context,
            color_map_image=color_map_image,
            init_image=init_image,
            mask_image=mask_image,
            input_prompt=input_prompt,
            num_inference_steps=150,
            guidance_scale=7.5,
            device="cuda:0",
            seed=81,
            weight_function=lambda w, sigma, qk: 0.15 * w * math.log(1 + sigma) * qk.max(),
            strength = 1.0,
        )


    img.save(settings["output_img_path"])
