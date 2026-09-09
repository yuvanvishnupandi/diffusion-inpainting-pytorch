import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import torchvision.transforms as T

class Evaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # setup metrics
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_metric = lpips.LPIPS(net='vgg').to(self.device)
        
        self.transform = T.ToTensor()

    def evaluate(self, orig, recon):
        # convert images to tensor
        img1 = self.transform(orig).unsqueeze(0).to(self.device)
        img2 = self.transform(recon).unsqueeze(0).to(self.device)

        psnr_val = self.psnr_metric(img2, img1).item()
        ssim_val = self.ssim_metric(img2, img1).item()

        # lpips needs -1 to 1 range
        img1_lpips = img1 * 2 - 1
        img2_lpips = img2 * 2 - 1
        
        with torch.no_grad():
            lpips_val = self.lpips_metric(img1_lpips, img2_lpips).item()

        return {
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'LPIPS': lpips_val
        }

