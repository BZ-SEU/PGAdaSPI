"""MSE/PSNR on unclipped physical intensities."""
import math
import torch

def mse_psnr(image: torch.Tensor, truth: torch.Tensor) -> tuple[float,float]:
    mse=float((image.double()-truth.double()).square().mean())
    return mse, float('inf') if mse==0 else -10*math.log10(mse)
