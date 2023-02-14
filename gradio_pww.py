from PIL import Image
import math
import torch
from random import randint
import ast
import gradio as gr
import dotenv

from paint_with_words import paint_with_words


dotenv.load_dotenv()


def run_pww(input_image, color_context, input_prompt, a_prompt, n_prompt, num_samples, ddim_steps, scale, seed, eta, device):
    
    color_map_image = Image.fromarray(input_image['image'])
    color_context = ast.literal_eval(color_context)
    if device == 'cuda':
        device += ':0'

    gen_seed = [seed]
    if num_samples > 1:
        gen = torch.Generator()
        gen.manual_seed(seed)
        gen_seed.extend([int(i) for i in torch.randint(0, 2147483647, (1, num_samples-1), generator=gen).ravel()])   

    images = []
    for _seed in gen_seed:
        img = paint_with_words(
            color_context=color_context,
            color_map_image=color_map_image,
            input_prompt='%s,%s'%(input_prompt,a_prompt),
            unconditional_input_prompt=n_prompt,
            num_inference_steps=ddim_steps,
            guidance_scale=scale,
            device=device,
            seed=_seed,
            weight_function=lambda w, sigma, qk: 0.4 * w * math.log(1 + sigma) * qk.max(),
            strength=eta
        )
        images.append(img)

    return images


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Paint-with-word")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type='numpy', tool='sketch')
            prompt = gr.Textbox(label="Prompt")
            color_context = gr.Textbox(label="Color context", value='')
            device = gr.inputs.Dropdown(label='Device', default='cuda', choices=['cuda', 'mps'])
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                # seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=0)
                eta = gr.Number(label="eta (DDIM)", value=0.5)
                a_prompt = gr.Textbox(label="Added Prompt", value='')
                n_prompt = gr.Textbox(label="Negative Prompt", value='')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, color_context, prompt, a_prompt, n_prompt, num_samples, ddim_steps, scale, seed, eta, device]
    run_button.click(fn=run_pww, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
