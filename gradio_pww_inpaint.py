from PIL import Image, ImageDraw
import numpy as np
import math
import torch
import ast
import gradio as gr
import dotenv
from paint_with_words import paint_with_words_inpaint
from gradio_pww import collect_color_content, create_canvas, extract_color_textboxes


dotenv.load_dotenv()

MAX_NUM_COLORS = 8


def run_pww(color_map_image, init_image, mask_image, color_context, input_prompt, a_prompt, n_prompt, num_samples, ddim_steps, scale, seed, eta, device, width, height):
    
    mask_image = mask_image['image'].convert('L')
    color_map_image = color_map_image.resize((width, height), Image.Resampling.NEAREST)
    if init_image is not None:
        init_image = init_image.resize((width, height), Image.Resampling.BILINEAR)
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
        img = paint_with_words_inpaint(
            color_context=color_context,
            color_map_image=color_map_image,
            init_image=init_image,
            mask_image=mask_image,
            input_prompt='%s,%s'%(input_prompt,a_prompt),
            unconditional_input_prompt=n_prompt,
            num_inference_steps=ddim_steps,
            guidance_scale=scale,
            device=device,
            seed=_seed,
            weight_function=lambda w, sigma, qk: 0.15 * w * math.log(1 + sigma) * qk.max(),
            strength = eta,
        )

        images.append(img)

    return images


if __name__ == '__main__':
    
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## Paint-with-word for Image inpainting")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    color_map_image = gr.Image(label='Segmentation map', source='upload', type='pil', tool='color-sketch', id='segmentation-map')
                    init_image = gr.Image(label='Initial image', source='upload', type='pil')
                with gr.Row():
                    mask_image = gr.Image(label='Mask image', source='upload', type='pil', tool='sketch')
                    gr.Column()
                prompt = gr.Textbox(label="Prompt")
                color_context = gr.Textbox(label="Color context", value='')
                run_button = gr.Button(value="Run Paint-with-Word")            

            with gr.Column():
                with gr.Accordion("Color content options", open=False):
                    with gr.Row():
                        extract_color_boxes_button = gr.Button(value="Extract color content")
                        generate_color_boxes_button = gr.Button(value="Generate color content")
                    prompts = []
                    strengths = []
                    seeds = []
                    color_maps = []
                    colors = [gr.Textbox(value="", visible=False) for i in range(MAX_NUM_COLORS)]
                    for n in range(MAX_NUM_COLORS):
                        with gr.Accordion('item%d'%n):
                            with gr.Row():
                                color_maps.append(gr.Image(image=create_canvas(15,3), interactive=False, type='numpy'))
                                with gr.Column():
                                    prompts.append(gr.Textbox(label="Prompt", interactive=True))
                                    with gr.Row():
                                        strengths.append(gr.Textbox(label="Strength", interactive=True))
                                        seeds.append(gr.Textbox(label="Random Seed", interactive=True))
                            
                with gr.Accordion("Advanced options", open=False):
                    width = gr.Slider(label="Width", minimum=256, maximum=1024, value=512, step=256)
                    height = gr.Slider(label="Height", minimum=256, maximum=1024, value=512, step=256)
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=300, value=150, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=0)
                    eta = gr.Number(label="eta (DDIM)", value=1.0)
                    a_prompt = gr.Textbox(label="Added Prompt", value='')
                    n_prompt = gr.Textbox(label="Negative Prompt", value='')

                device = gr.inputs.Dropdown(label='Device', default='cuda', choices=['cuda', 'mps'])

                with gr.Row():
                    gr.Markdown("### Results")
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        extract_color_boxes_button.click(fn=extract_color_textboxes, inputs=[color_map_image], outputs=[*color_maps, *prompts, *strengths, *seeds, *colors])
        generate_color_boxes_button.click(fn=collect_color_content, inputs=[*colors, *prompts, *strengths, *seeds], outputs=[color_context])
        ips = [color_map_image, init_image, mask_image, color_context, prompt, a_prompt, n_prompt, num_samples, ddim_steps, scale, seed, eta, device, width, height]
        run_button.click(fn=run_pww, inputs=ips, outputs=[result_gallery])   
        

    block.launch(server_name='0.0.0.0')
