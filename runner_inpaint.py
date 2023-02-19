import math

import dotenv
from PIL import Image

from paint_with_words import paint_with_words_inpaint


EXAMPLE_SETTING_1 = {
    "color_context": {
        (51, 193, 217): "yellow cat,2.0,-1",
    },
    "color_map_img_path": "contents/inpaint_examples/dog_2.png",
    "input_prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
    "output_img_path": "contents/inpaint_examples/cat_output.png",
    "img_path": "contents/inpaint_examples/dog_on_bench.png",
    "mask_path": "contents/inpaint_examples/dog_mask.png",
}

EXAMPLE_SETTING_2 = {
    "color_context": {
        (51, 193, 217): "red parrot,2.0,-1",
    },
    "color_map_img_path": "contents/inpaint_examples/dog_1.png",
    "input_prompt": "a red parrot, high resolution, flying on a park bench",
    "output_img_path": "contents/inpaint_examples/bird_output.png",
    "img_path": "contents/inpaint_examples/dog_on_bench.png",
    "mask_path": "contents/inpaint_examples/dog_mask.png",
}

EXAMPLE_SETTING_3 = {
    "color_context": {
        (51, 193, 217): "flower,2.0,-1",
    },
    "color_map_img_path": "contents/inpaint_examples/dog_2.png",
    "input_prompt": "a flower, high resolution, flying on a park bench",
    "output_img_path": "contents/inpaint_examples/leaf_output.png",
    "img_path": "contents/inpaint_examples/dog_on_bench.png",
    "mask_path": "contents/inpaint_examples/dog_mask.png",
}

if __name__ == "__main__":

    dotenv.load_dotenv()
    
    settings = EXAMPLE_SETTING_3

    color_map_image = Image.open(settings["color_map_img_path"]).convert("RGB")
    color_context = settings["color_context"]
    input_prompt = settings["input_prompt"]
    init_image = Image.open(settings["img_path"]).convert("RGB")
    mask_image = Image.open(settings["mask_path"])
    img = paint_with_words_inpaint(
        color_context=color_context,
        color_map_image=color_map_image,
        init_image=init_image,
        mask_image=mask_image,
        input_prompt=input_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        device="cuda:0",
        seed=81,
        # weight_function=lambda w, sigma, qk: 0.4 * w * math.log(1 + sigma) * qk.max(),
        weight_function=lambda w, sigma, qk: 0.2 * w * math.log(1 + sigma) * qk.max(),
        strength = 0.7,
    )

    img.save(settings["output_img_path"])
