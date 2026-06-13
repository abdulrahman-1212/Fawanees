import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

def generate_custom_cartoon(input_image_path, output_image_path, prompt):
    
    # 1. Load the free ControlNet model (Canny Edge Detector)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16 # Use float16 to save GPU memory
    )

    # 2. Load the base Stable Diffusion model (Completely free & open source)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet, 
        safety_checker=None, # Disable NSFW filter since we are generating kids' cartoons
        torch_dtype=torch.float16
    )

    # Speed up the pipeline
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    print("Processing image...")
    image = Image.open(input_image_path).convert("RGB")
    
    # Stable diffusion requires 512x512 images
    image = image.resize((512, 512))
    image_np = np.array(image)

    low_threshold = 100
    high_threshold = 200
    canny_image = cv2.Canny(image_np, low_threshold, high_threshold)
    canny_image = canny_image[:, :, None] # Add channel dimension
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    style_suffix = ", 3d pixar style, disney animation, cute, vibrant colors, children's book illustration, high quality"
    full_prompt = prompt + style_suffix

    print(f"Generating cartoon with prompt: '{full_prompt}'...")
    
    cartoon_image = pipe(
        full_prompt, 
        image=canny_image, 
        num_inference_steps=20,
        controlnet_conditioning_scale=0.8 
    ).images[0]

    cartoon_image.save(output_image_path)
    print(f"Success! Cartoon saved to {output_image_path}")
    return cartoon_image
