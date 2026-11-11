"""uint8 complementary quantization and model-coordinate measurement conversion."""
import math
import torch
from .types import PatternBatch

def quantize(p: torch.Tensor, ids: list[str] | None = None, bit_depth: int = 8,
             thresholds: torch.Tensor | None = None) -> PatternBatch:
    """Quantize signed DMD patterns ``p`` [B,d] to complementary uint8 codes.

    ``bit_depth=8`` is the historical rounding rule.  ``bit_depth=1`` is a
    strict binary amplitude DMD: every positive code is either 0 or 255.
    ``thresholds`` is optional per-row signed-DMD threshold(s), used only for
    the binary experimental branch.  In every case the formal measurement
    rows are reconstructed from the actual emitted uint8 codes, never from
    the continuous relaxation.
    """
    if bit_depth not in (1, 8):
        raise ValueError(f'DMD bit_depth must be 1 or 8, got {bit_depth}')
    p = p.clamp(-1,1)
    if bit_depth == 8:
        if thresholds is not None:
            raise ValueError('thresholds are defined only for 1-bit DMD quantization')
        plus = torch.round(255*(p+1)/2).to(torch.uint8)
    else:
        if thresholds is None:
            threshold = torch.zeros((len(p), 1), dtype=p.dtype, device=p.device)
        else:
            threshold = torch.as_tensor(thresholds, dtype=p.dtype, device=p.device)
            if threshold.ndim == 0:
                threshold = threshold.expand(len(p)).reshape(-1, 1)
            elif threshold.ndim == 1 and threshold.numel() == len(p):
                threshold = threshold[:, None]
            elif tuple(threshold.shape) != (len(p), 1):
                raise ValueError(f'binary thresholds must be scalar or [B], got {tuple(threshold.shape)}')
        plus = torch.where(p >= threshold, 255, 0).to(torch.uint8)
    minus = 255-plus
    rows = (plus.to(p.dtype)-minus.to(p.dtype))/255/math.sqrt(p.shape[-1])
    return PatternBatch(p, plus, minus, rows, ids or [f'local:{i}' for i in range(len(p))])


def model_measurements(A: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    """For z=2x-1: A_model=A/2, y_model=y-A1/2. No ad hoc conversions elsewhere."""
    return A/2, y-A.sum(-1)/2
