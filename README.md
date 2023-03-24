# Paint-with-Words, Implemented with Stable diffusion

## Subtle Control of the Image Generation

<!-- #region -->
<p align="center">
<img  src="contents/rabbit_mage.jpg">
</p>
<!-- #endregion -->

> Notice how without PwW the cloud is missing.

<!-- #region -->
<p align="center">
<img  src="contents/road.jpg">
</p>
<!-- #endregion -->

> Notice how without PwW, abandoned city is missing, and road becomes purple as well.

## Shift the object : Same seed, just the segmentation map's positional difference

<!-- #region -->
<p align="center">
<img  src="contents/aurora_1_merged.jpg">
</p>
<!-- #endregion -->

<!-- #region -->
<p align="center">
<img  src="contents/aurora_2_merged.jpg">
</p>
<!-- #endregion -->

> "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed."

> Notice how nearly all of the composition remains the same, other than the position of the moon.

---

Recently, researchers from NVIDIA proposed [eDiffi](https://arxiv.org/abs/2211.01324). In the paper, they suggested method that allows "painting with word". Basically, this is like make-a-scene, but with just using adjusted cross-attention score. You can see the results and detailed method in the paper.

Their paper and their method was not open-sourced. Yet, paint-with-words can be implemented with Stable Diffusion since they share common Cross Attention module. So, I implemented it with Stable Diffusion.

<!-- #region -->
<p align="center">
<img  src="contents/paint_with_words_figure.png">
</p>
<!-- #endregion -->

# Installation

```bash
pip install git+https://github.com/cloneofsimo/paint-with-words-sd.git
```

# Basic Usage

Before running, fill in the variable `HF_TOKEN` in `.env` file with Huggingface token for Stable Diffusion, and load_dotenv().

Prepare segmentation map, and map-color : tag label such as below. keys are (R, G, B) format, and values are tag label.

```python
{
    (0, 0, 0): "cat,1.0",
    (255, 255, 255): "dog,1.0",
    (13, 255, 0): "tree,1.5",
    (90, 206, 255): "sky,0.2",
    (74, 18, 1): "ground,0.2",
}
```

You neeed to have them so that they are in format "{label},{strength}", where strength is additional weight of the attention score you will give during generation, i.e., it will have more effect.

```python

import dotenv
from PIL import Image

from paint_with_words import paint_with_words

settings = {
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


dotenv.load_dotenv()

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

img.save(settings["output_img_path"])

```

There is minimal working example in `runner.py` that is self contained. Please have a look!

---

# Weight Scaling

In the paper, they used $w \log (1 + \sigma)  \max (Q^T K)$ to scale appropriate attention weight. However, this wasn't optimal after few tests, found by [CookiePPP](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/4406). You can check out the effect of the functions below:

<!-- #region -->
<p align="center">
<img  src="contents/compare_std.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma)  std (Q^T K)$

<!-- #region -->
<p align="center">
<img  src="contents/compare_max.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma)  \max (Q^T K)$

<!-- #region -->
<p align="center">
<img  src="contents/compare_log2_std.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma^2)  std (Q^T K)$

You can define your own weight function and further tweak the configurations by defining `weight_function` argument in `paint_with_words`.

Example:

```python
w_f = lambda w, sigma, qk: 0.4 * w * math.log(sigma**2 + 1) * qk.std()

img = paint_with_words(
    color_context=color_context,
    color_map_image=color_map_image,
    input_prompt=input_prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    device="cuda:0",
    preloaded_utils=loaded,
    weight_function=w_f
)
```

## More on the weight function, (but higher)

<!-- #region -->
<p align="center">
<img  src="contents/compare_4_std.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma)  std (Q^T K)$

<!-- #region -->
<p align="center">
<img  src="contents/compare_4_max.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma)  \max (Q^T K)$

<!-- #region -->
<p align="center">
<img  src="contents/compare_4_log2_std.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma^2)  std (Q^T K)$

# Regional-based seeding

Following this example, where the random seed for whole image is 0,
<!-- #region -->
<p align="center">
<img  src="contents/aurora_1_merged.jpg">
</p>
<!-- #endregion -->

> "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed."

the random seed for 'boat', 'moon', and 'mountain' are set to various values show in the top row.

<!-- #region -->
<p align="center">
<img  src="contents/cmp_regional_based_seeing.png">
</p>
<!-- #endregion -->

Example:

```python

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
```
where the 3rd item of context are random seed for the object. Use -1 to follow the seed set in paint_with_words function. In this example the random seed of boat is set to 2077.

# Image inpainting
Following the previous example, the figure below shows the results of image inpainting with paint-with-word
<!-- #region -->
<p align="center">
<img  src="contents/pww_inpainting.jpg">
</p>
<!-- #endregion -->

where the top row shows the example of editing moon size by inpainting.
The bottom row shows the example of re-synthesize the moon by inpainting with the original "input color map" for text-image paint-with-word.


Example

```python
from paint_with_words import paint_with_words_inpaint


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
```

To run inpainting

```bash
python runner_inpaint.py
```

# Using other Fine-tuned models

If you are from Automatic1111 community, you maybe used to using native LDM checkpoint formats, not diffuser-checkpoint format. Luckily, there is a quick script that allows conversion.
[this](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py).

```bash
python change_model_path.py --checkpoint_path custom_model.ckpt --scheduler_type ddim --dump_path custom_model_diffusion_format
```

Now, use the converted model in `paint_with_words` function.

```python
from paint_with_words import paint_with_words, pww_load_tools

loaded = pww_load_tools(
    "cuda:0",
    scheduler_type=LMSDiscreteScheduler,
    local_model_path="./custom_model_diffusion_format"
)
#...
img = paint_with_words(
    color_context=color_context,
    color_map_image=color_map_image,
    input_prompt=input_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    device="cuda:0",
    weight_function=lambda w, sigma, qk: 0.4 * w * math.log(1 + sigma) * qk.max(),
    preloaded_utils=loaded
)
```

# Example Notebooks

You can view the minimal working notebook [here](./contents/notebooks/paint_with_words.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MZfGaY3aQQn5_T-6bkXFE1rI59A2nJlU?usp=sharing)

- [Painting with words](./contents/notebooks/paint_with_words.ipynb)

- [Painting with words + Textual Inversion](./contents/notebooks/paint_with_words_textual_inversion.ipynb)

---

# Gradio interface
## Paint-with-word
To launch gradio api

```bash
python gradio_pww.py
```

<!-- #region -->
<p align="center">
<img  src="contents/gradio_demo.png">
</p>
<!-- #endregion -->

Noting that the "Color context" should follows the format defined as the example in runner.py. 
For example, 
> {(7, 9, 182): "aurora,0.5,-1",(136, 178, 92): "full moon,1.5,-1",(51, 193, 217): "mountains,0.4,-1",(61, 163, 35): "a half-frozen lake,0.3,-1",(89, 102, 255): "boat,2.0,2077",}

### Color contenet extraction
One can extract the color content from "Segmentation map" by expanding the "Color content option". 
Press the button "Extract color content" to extract the unique color of images.

<!-- #region -->
<p align="center">
<img  src="contents/gradio_color_content_demo_0.png">
</p>
<!-- #endregion -->

In "Color content option", the extracted colors are shown respectively for each item. One can then replace "obj" with the object appear in the prompt. Importantly, don't use "," in the object, as this is the separator of the color content.

Click the button "Generate color content" to collect all the contents into "Color content" the textbox as the formal input of Paint-with-word.

<!-- #region -->
<p align="center">
<img  src="contents/gradio_color_content_demo.png">
</p>
<!-- #endregion -->

The same function is supported for Paint-with-word for image inpainting as shown below

## Paint-with-word for image inpainting
To launch gradio api

```bash
python gradio_pww_inpaint.py
```

<!-- #region -->
<p align="center">
<img  src="contents/gradio_inpaint_demo.png">
</p>
<!-- #endregion -->


# Paint with Word (PwW) + ControlNet Extension for [AUTOMATIC1111(A1111) stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

This extension provide additional PwW control to ControlNet. See [sd-webui-controlnet-pww
](https://github.com/lwchen6309/sd-webui-controlnet-pww) for the repo of this module.

The demo is shown below.

![screencapture-127-0-0-1-7860-2023-03-13-10_56_34](https://user-images.githubusercontent.com/42672685/225545442-bdb481ec-e234-475e-900d-e9340c0c7deb.png)

The implementation is based on the great [controlnet extension for A1111](https://github.com/Mikubill/sd-webui-controlnet)

## Benchmark of ControlNet + PwW

The following figure shows the comparison between the ControlNet results and the ControlNet+PwW results for the boat examples. 

<!-- #region -->
<p align="center">
<img  src="contents/cn_pww/cn_pww_boat.jpg">
</p>
<!-- #endregion -->

Noting that the PwW make the background, e.g. aurora and mountains, more realistic as weight function scales increases. 

The setups are detailed as follows

Scribble and Segmentation map:

<p float="middle">
  <img src="contents/cn_pww/user1.png" width="200" />
  <img src="contents/cn_pww/seg_map1.png" width="200" /> 
</p>

Prompts:

> "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed."

Color contents: 

> "{(7, 9, 182): "aurora@0.5@-1",(136, 178, 92): "full moon@1.5@-1",(51, 193, 217): "mountains@0.4@-1",(61, 163, 35): "a half-frozen lake@0.3@-1",(89, 102, 255): "boat@2.0@-1",}"

Note that A1111 extension now use "@" as separator instead of ",".

## Assign the material for the specific region in scribble

One can use PwW to assign the material upon scribble, see the results comparing ControlNet and ControlNet+PwW below.

<!-- #region -->
<p align="center">
<img  src="contents/cn_pww/cn_pww_turtle.jpg">
</p>
<!-- #endregion -->

<!-- #region -->
<p align="center">
<img  src="contents/cn_pww/cn_pww_ballon.jpg">
</p>
<!-- #endregion -->

Noting that the material of turtle shell specified by PwW is significantly improved showns in the right blocks.
Please see [sd-webui-controlnet-pww
](https://github.com/lwchen6309/sd-webui-controlnet-pww#assign-the-material-for-the-specific-region-in-scribble) for the experimental setups.

## Installation

### (1) Clone the source code to A1111 webui extensions
one can install by cloning the 'pww_controlnet" directory into the extensions directory of A1111 webui

```bash
cp -rf pww_controlnet path/stable-diffusion-webui/extensions/
```

or simply

```bash
cd path/stable-diffusion-webui/extensions/
git clone git@github.com:lwchen6309/sd-webui-controlnet-pww.git
```

where path is the location of A1111 webui.

### (2) Setup pretrained model of ControlNet
Please follow the instruction of [controlnet extension](https://github.com/Mikubill/sd-webui-controlnet) to get the pretrained models. 

#### IMPORTANT: This extension is currently NOT compatible with [ControlNet extension](https://github.com/Mikubill/sd-webui-controlnet) as reported at [this issue](https://github.com/cloneofsimo/paint-with-words-sd/issues/38). Hence, please disable the ControlNet extension before you install ControlNet+PwW.

However, one can still make them compatible by following [the instruction of installation](https://github.com/lwchen6309/sd-webui-controlnet-pww/tree/fc7b0e4471f1da491d12a2f12f3f0487bb671696#important-this-extension-is-currently-not-compatible-with-controlnet-extension-as-reported-at-this-issue-hence-please-disable-the-controlnet-extension-before-you-install-controlnetpww-this-repo-will-sync-the-latest-controlnet-extension-and-should-therefore-includes-its-original-function).


# TODO

- [ ] Make extensive comparisons for different weight scaling functions.
- [ ] Create word latent-based cross-attention generations.
- [ ] Check if statement "making background weight smaller is better" is justifiable, by using some standard metrics
- [x] Create AUTOMATIC1111's interface
- [x] Create Gradio interface
- [x] Create tutorial
- [ ] See if starting with some "known image latent" is helpful. If it is, we might as well hard-code some initial latent.
- [x] Region based seeding, where we set seed for each regions. Can be simply implemented with extra argument in `COLOR_CONTEXT`
- [ ] sentence wise text seperation. Currently token is the smallest unit that influences cross-attention. This needs to be fixed. (Can be done pretty trivially)
- [x] Allow different models to be used. use [this](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py).
- [ ] "negative region", where we can set some region to "not" have some semantics. can be done with classifier-free guidance.
- [x] Img2ImgPaintWithWords -> Img2Img, but with extra text segmentation map for better control
- [x] InpaintPaintwithWords -> inpaint, but with extra text segmentation map for better control
- [x] Support for other schedulers

# Acknowledgement
Thanks for the inspiring gradio interface from [ControlNet](https://github.com/lllyasviel/ControlNet)

Thanks for the wonderful [A1111 extension of controlnet](https://github.com/Mikubill/sd-webui-controlnet) as the baseline of our implementation
