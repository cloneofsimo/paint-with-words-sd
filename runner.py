import math

import dotenv
from PIL import Image
import torch 
from paint_with_words import paint_with_words, PaintWithWord_StableDiffusionPipeline


EXAMPLE_SETTING_1 = {
    "color_context": {
        (0, 0, 0): "cat,1.0",
        (255, 255, 255): "dog,1.0",
        (13, 255, 0): "tree,1.5",
        (90, 206, 255): "sky,0.2",
        (74, 18, 1): "ground,0.2",
    },
    "color_map_img_path": "contents/example_input.png",
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
    "color_map_img_path": "contents/example_input.png",
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
    "color_map_img_path": "contents/aurora_2.png",
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
    "color_map_img_path": "contents/aurora_1.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
    "output_img_path": "contents/aurora_1_output.png",
}

EXAMPLE_SETTING_4_seed = {
    "color_context": {
        (7, 9, 182): "aurora,0.5,-1",
        (136, 178, 92): "full moon,1.5,-1",
        (51, 193, 217): "mountains,0.4,-1",
        (61, 163, 35): "a half-frozen lake,0.3,-1",
        (89, 102, 255): "boat,2.0,2077",
    },
    "color_map_img_path": "contents/aurora_1.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
    "output_img_path": "contents/aurora_1_seed_output.png",
}


if __name__ == "__main__":

    dotenv.load_dotenv()

    settings = EXAMPLE_SETTING_4_seed

    color_map_image = Image.open(settings["color_map_img_path"]).convert("RGB")
    color_context = settings["color_context"]
    input_prompt = settings["input_prompt"]
    
    use_pipeline = True
    if use_pipeline:
        pipe = PaintWithWord_StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe = pipe.to("cuda")
        img = pipe(color_context=color_context,
                color_map_image=color_map_image,
                prompt=input_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                weight_function=lambda w, sigma, qk: 0.4 * w * math.log(1 + sigma) * qk.max(),
        ).images[0]
    else:
        img = paint_with_words(
            color_context=color_context,
            color_map_image=color_map_image,
            input_prompt=input_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            device="cuda:0",
            weight_function=lambda w, sigma, qk: 0.4 * w * math.log(1 + sigma) * qk.max(),
        )

    img.save(settings["output_img_path"])
