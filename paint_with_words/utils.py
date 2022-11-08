import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def fig_from_settings(settings: Dict[str, Any]):

    # create before-after figure from settings
    color_map_image = Image.open(settings["color_map_img_path"]).convert("RGB")
    color_context = settings["color_context"]
    input_prompt = settings["input_prompt"]

    draw = ImageDraw.Draw(color_map_image)

    output_img = Image.open(settings["output_img_path"]).convert("RGB")

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        print("Could not load font. Using default font.")
        print(
            "Run $ sudo apt install ttf-mscorefonts-installer \n\
                $ sudo fc-cache -fv\n\
                to install the font."
        )
        font = ImageFont.load_default()

    # on color map image, write captions.
    for color, caption in color_context.items():

        is_color_region = (np.array(color_map_image) == color).all(axis=-1)
        color_region_indices = np.argwhere(is_color_region == True)
        yxs = color_region_indices[:, 0:2]
        top_left_corner = min(yxs.tolist())

        draw.text(
            (top_left_corner[1] + 5, top_left_corner[0] + 5),
            caption,
            (0, 0, 0),
            font=font,
        )

    # merge color map image and output image
    fig = Image.new(
        "RGB",
        (color_map_image.width + output_img.width, color_map_image.height + 30),
        (255, 255, 255),
    )
    fig.paste(color_map_image, (0, 0))
    fig.paste(output_img, (color_map_image.width, 0))

    # write input prompt
    draw = ImageDraw.Draw(fig)
    draw.text((5, color_map_image.height + 5), input_prompt, (0, 0, 0), font=font)

    return fig
