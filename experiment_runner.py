import os
import pandas as pd
from datasets import load_dataset
from PIL import Image
import numpy as np

from src.mask_generator import MaskGenerator
from src.inpainter import Inpainter
from src.evaluator import Evaluator

def setup_dirs():
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('masks', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def main():
    print("setting up...")
    setup_dirs()

    print("loading dataset...")
    # use a small dataset for testing
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train[:5]")
    
    orig_imgs = []
    for i, item in enumerate(dataset):
        img = item['image'].convert('RGB').resize((512, 512))
        img.save(f"dataset/orig_{i}.png")
        orig_imgs.append(img)

    print("initializing...")
    mask_gen = MaskGenerator((512, 512))
    inpainter = Inpainter()
    evaluator = Evaluator()

    results = []

    print("running...")
    for i, img in enumerate(orig_imgs):
        print(f"processing image {i}")
        
        masks = {
            'rect': mask_gen.get_rect_mask(),
            'irreg': mask_gen.get_irreg_mask(),
            'multi': mask_gen.get_multi_mask()
        }

        for m_type, mask in masks.items():
            mask.save(f"masks/mask_{i}_{m_type}.png")
            
            # save masked image
            img_np = np.array(img)
            mask_np = np.array(mask) / 255.0
            masked = img_np * (1 - np.expand_dims(mask_np, axis=2))
            Image.fromarray(masked.astype(np.uint8)).save(f"results/masked_{i}_{m_type}.png")

            # text guidance bonus for the first rect mask
            prompt = ""
            if i == 0 and m_type == 'rect':
                prompt = "a beautiful butterfly, high resolution"
                
            recon = inpainter.inpaint(img, mask, prompt=prompt)
            recon.save(f"results/recon_{i}_{m_type}.png")

            metrics = evaluator.evaluate(img, recon)
            
            results.append({
                'id': i,
                'mask': m_type,
                'prompt': prompt,
                'psnr': metrics['PSNR'],
                'ssim': metrics['SSIM'],
                'lpips': metrics['LPIPS']
            })

    df = pd.DataFrame(results)
    df.to_csv("results/metrics.csv", index=False)
    print("done!")
    print(df.groupby('mask')[['psnr', 'ssim', 'lpips']].mean())

if __name__ == "__main__":
    main()
