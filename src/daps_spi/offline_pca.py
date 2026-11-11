"""Immutable training-only PCA DMD bank; no observations or GT enter this module."""
from functools import lru_cache
from pathlib import Path
import torch
from .io import file_hash
from .types import PatternBatch


@lru_cache(maxsize=2)
def load_bank(path: str, expected_hash: str) -> dict:
    """Load CPU uint8 [K,d] physical codes and verify a sealed asset hash."""
    if file_hash(Path(path)) != expected_hash:
        raise ValueError('offline PCA bank hash mismatch')
    bank = torch.load(path, map_location='cpu', weights_only=False)
    codes = bank['plus_uint8']
    if codes.dtype != torch.uint8 or codes.ndim != 2:
        raise ValueError('PCA bank requires uint8 [K,d] positive exposures')
    return bank


def bank_batch(spec: dict, start: int, count: int, optics, device: str) -> PatternBatch:
    """Return predetermined rows, independent of A/y, in descending PCA variance order."""
    from .propagation_policy import mapped
    bank = load_bank(spec['path'], spec['sha256'])
    if bank['metadata']['physics'] != optics.metadata()['parameters']:
        raise ValueError('PCA bank optical parameters differ from runtime')
    codes = bank['plus_uint8'][start:start+count].to(device)
    if codes.shape != (count, optics.height*optics.width):
        raise ValueError('PCA bank has insufficient rows or wrong geometry')
    p=(2*codes.double()-255)/255
    return mapped(PatternBatch(p,codes,255-codes,torch.empty_like(p),
                               [f'offline_pca:{i}' for i in range(start,start+count)]),optics)
