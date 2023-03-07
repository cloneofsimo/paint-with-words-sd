from PIL import Image, ImageDraw
import numpy as np
import math
import torch
import ast
import gradio as gr
import dotenv
from paint_with_words import paint_with_words_inpaint
# from gradio_pww import collect_color_content, create_canvas, extract_color_textboxes


dotenv.load_dotenv()

MAX_NUM_COLORS = 8

def extract_color_textboxes(color_map_image, *color_textbox):
    # Get unique colors in color_map_image
    colors = unique_colors(color_map_image)
    
    # Update color_textbox labels and values based on unique colors
    color_textbox = ["" for _ in range(MAX_NUM_COLORS)]
    for i in range(len(colors)):
        if i == MAX_NUM_COLORS:
            break
        color_textbox[i] = f"{colors[i]}:obj,0.5,-1"
    # # Clear the remaining textboxes
    for i in range(len(colors), MAX_NUM_COLORS):
        color_textbox[i] = ""
    
    colorbar = generate_color_bar(colors)
    return (colorbar, *color_textbox)


def quote_text(s):
    # Find the index of the colon separating the color values and text
    colon_index = s.find(':')
    # Extract the color values
    color_values = s[:colon_index]
    # Extract the text
    text = s[colon_index + 1:]
    # Add quotes around the text
    quoted_text = '"' + text + '"'
    # Combine the color values and quoted text
    result = color_values + ':' + quoted_text
    return result


def unique_colors(image, threshold=0.01):
    colors = image.getcolors(image.size[0] * image.size[1])
    total_pixels = image.size[0] * image.size[1]
    unique_colors = []
    for count, color in colors:
        if count / total_pixels > threshold:
            unique_colors.append(color)
    return unique_colors


def generate_color_bar(colors, width=100, height=5):
    # Create a new image with the specified width and height
    img = Image.new("RGB", (width, height))
    
    # Compute the width of each color stripe based on the number of colors
    stripe_width = width // len(colors)
    
    # Draw a rectangle for each color stripe
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(colors):
        # Compute the x-coordinate of the left edge of this stripe
        x0 = i * stripe_width
        
        # Compute the x-coordinate of the right edge of this stripe
        x1 = x0 + stripe_width
        
        # Draw the rectangle for this color stripe
        draw.rectangle((x0, 0, x1, height), fill=color)
    
    return img


def collect_color_content(*color_textbox):
    content_collection = []
    for color in color_textbox:
        if len(color) > 0:
            content_collection.append(quote_text(color))
    if len(content_collection) > 0:
        return "{%s}"%','.join(content_collection)
    else:
        return ""


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
    # Create a black image with shape (100, 20)
    data = np.zeros((100, 5), dtype=np.uint8)
    blank_image = Image.fromarray(data)
    # Make the image read-only
    blank_image.load = lambda: None
    blank_image.readonly = True

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## Paint-with-word for Image inpainting")
        with gr.Row():
            with gr.Column():
                color_map_image = gr.Image(label='Segmentation map', source='upload', type='pil', tool='color-sketch')
                prompt = gr.Textbox(label="Prompt")
                with gr.Accordion("Color content options", open=False):
                    # Initialize with empty textboxes
                    color_map = gr.Image(image=blank_image, label='color map', type='pil')
                    color_textbox = [gr.Textbox(label=f"Color {i+1}", value="") for i in range(MAX_NUM_COLORS)]
                    with gr.Row():
                        extract_color_boxes_button = gr.Button(value="Extract color content")
                        generate_color_boxes_button = gr.Button(value="Generate color content")
                color_context = gr.Textbox(label="Color context", value='')
                init_image = gr.Image(label='Initial image', source='upload', type='pil')
                mask_image = gr.Image(label='Mask image', source='upload', type='pil', tool='sketch')
                device = gr.inputs.Dropdown(label='Device', default='cuda', choices=['cuda', 'mps'])
                run_button = gr.Button(value="Run Paint-with-Word")
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
            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        extract_color_boxes_button.click(fn=extract_color_textboxes, inputs=[color_map_image, *color_textbox], outputs=[color_map, *color_textbox])
        generate_color_boxes_button.click(fn=collect_color_content, inputs=[*color_textbox], outputs=[color_context])
        ips = [color_map_image, init_image, mask_image, color_context, prompt, a_prompt, n_prompt, num_samples, ddim_steps, scale, seed, eta, device, width, height]
        run_button.click(fn=run_pww, inputs=ips, outputs=[result_gallery])   

    # block = gr.Blocks().queue()
    # with block:
    #     with gr.Row():
    #         gr.Markdown("## Paint-with-word for Image inpainting")
    #     with gr.Row():
    #         with gr.Column():
    #             with gr.Row():
    #                 color_map_image = gr.Image(label='Segmentation map', source='upload', type='pil', tool='color-sketch', id='segmentation-map')
    #                 init_image = gr.Image(label='Initial image', source='upload', type='pil')
    #             mask_image = gr.Image(label='Mask image', source='upload', type='pil', tool='sketch')
    #             prompt = gr.Textbox(label="Prompt")
    #             color_context = gr.Textbox(label="Color context", value='')
    #             run_button = gr.Button(value="Run Paint-with-Word")            

    #         with gr.Column():
    #             with gr.Accordion("Color content options", open=False):
    #                 with gr.Row():
    #                     extract_color_boxes_button = gr.Button(value="Extract color content")
    #                     generate_color_boxes_button = gr.Button(value="Generate color content")
    #                 prompts = []
    #                 strengths = []
    #                 seeds = []
    #                 color_maps = []
    #                 colors = [gr.Textbox(value="", visible=False) for i in range(MAX_NUM_COLORS)]
    #                 for n in range(MAX_NUM_COLORS):
    #                     with gr.Accordion('item%d'%n):
    #                         with gr.Row():
    #                             color_maps.append(gr.Image(image=create_canvas(15,3), interactive=False, type='numpy'))
    #                             with gr.Column():
    #                                 prompts.append(gr.Textbox(label="Prompt", interactive=True))
    #                                 with gr.Row():
    #                                     strengths.append(gr.Textbox(label="Strength", interactive=True))
    #                                     seeds.append(gr.Textbox(label="Random Seed", interactive=True))
                            
    #             with gr.Accordion("Advanced options", open=False):
    #                 width = gr.Slider(label="Width", minimum=256, maximum=1024, value=512, step=256)
    #                 height = gr.Slider(label="Height", minimum=256, maximum=1024, value=512, step=256)
    #                 num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    #                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=300, value=150, step=1)
    #                 scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
    #                 seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=0)
    #                 eta = gr.Number(label="eta (DDIM)", value=1.0)
    #                 a_prompt = gr.Textbox(label="Added Prompt", value='')
    #                 n_prompt = gr.Textbox(label="Negative Prompt", value='')

    #             device = gr.inputs.Dropdown(label='Device', default='cuda', choices=['cuda', 'mps'])

    #             with gr.Row():
    #                 gr.Markdown("### Results")
    #             result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    #     extract_color_boxes_button.click(fn=extract_color_textboxes, inputs=[color_map_image], outputs=[*color_maps, *prompts, *strengths, *seeds, *colors])
    #     generate_color_boxes_button.click(fn=collect_color_content, inputs=[*colors, *prompts, *strengths, *seeds], outputs=[color_context])
    #     ips = [color_map_image, init_image, mask_image, color_context, prompt, a_prompt, n_prompt, num_samples, ddim_steps, scale, seed, eta, device, width, height]
    #     run_button.click(fn=run_pww, inputs=ips, outputs=[result_gallery])   
        

    block.launch(server_name='0.0.0.0')
