import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

class Inpainter:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # use float16 if we have gpu so it doesnt crash
        dt = torch.float16 if self.device == 'cuda' else torch.float32
        
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dt,
            variant="fp16",
        )
        self.pipeline = self.pipeline.to(self.device)
        
        if self.device == 'cuda':
            self.pipeline.enable_attention_slicing()

    def inpaint(self, image, mask, prompt=""):
        # resize to 512 for stable diffusion
        orig_size = image.size
        img512 = image.resize((512, 512))
        mask512 = mask.resize((512, 512))
        
        gen = torch.Generator(device=self.device).manual_seed(42)
        
        out = self.pipeline(
            prompt=prompt,
            image=img512,
            mask_image=mask512,
            generator=gen,
            num_inference_steps=20
        ).images[0]
        
        # put back to normal size
        out = out.resize(orig_size)
        
        # paste original parts back over the generated image
        orig_np = np.array(image).astype(np.float32)
        gen_np = np.array(out).astype(np.float32)
        
        mask_np = np.array(mask).astype(np.float32) / 255.0
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, axis=2)
            
        final = orig_np * (1 - mask_np) + gen_np * mask_np
        return Image.fromarray(final.astype(np.uint8))

