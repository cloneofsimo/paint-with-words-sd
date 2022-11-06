from paint_with_words import paint_with_words
from PIL import Image
import dotenv
import os

EXAMPLE_SETTING_1 = {
    "color_context": {
        (0, 0, 0): "cat,2.0",
        (255, 255, 255): "dog,2.0",
        (13, 255, 0): "tree,3.0",
        (90, 206, 255): "sky,0.5",
        (74, 18, 1): "ground,0.5",
    },
    "color_map_img_path": "contents/example_input.png",
    "input_prompt": "photo of dog, cat, tree, sky, ground",
}

EXAMPLE_SETTING_2 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_img_path": "contents/aurora_2.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
}

if __name__ == "__main__":

    dotenv.load_dotenv()

    settings = EXAMPLE_SETTING_2

    color_map_image = Image.open(settings["color_map_img_path"]).convert("RGB")
    color_context = settings["color_context"]
    input_prompt = settings["input_prompt"]

    img = paint_with_words(
        color_context=color_context,
        color_map_image=color_map_image,
        input_prompt=input_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        device="cuda:0",
    )

    img.save("contents/output.png")
