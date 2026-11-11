"""Pattern pools: 2D Walsh (square power-of-two) and cosine-sign (rectangular) bases."""
import math
from functools import lru_cache
import torch
import torch.nn.functional as F
from .physics import quantize
from .types import PatternBatch

@lru_cache(maxsize=8)
def pool_order(size: int=256, count: int=4096) -> list[tuple[int,int]]:
    """DC first; 2D sequency pairs sorted by (u+v,max(u,v),u,v)."""
    if size&(size-1): raise ValueError('power-of-two image side required')
    return sorted([(u,v) for u in range(size) for v in range(size)],key=lambda t:(sum(t),max(t),*t))[:count]

@lru_cache(maxsize=8)
def rectangular_order(height: int, width: int, count: int = 4096) -> list[tuple[int,int]]:
    """Deterministic low-frequency separable cosine-sign basis for rectangular templates."""
    if count > height * width: raise ValueError('rectangular pool exceeds pixel count')
    return sorted([(u,v) for u in range(height) for v in range(width)],
                  key=lambda t:(sum(t),max(t),*t))[:count]


def pool_indices(size: int, count: int, design: dict | None = None) -> list[int]:
    """Canonical full-Walsh IDs in sequency order; the first IDs are the common initial prefix."""
    design = design or {'name': 'legacy_low_order'}
    if not 0 < count <= size*size: raise ValueError('invalid pool size')
    if design['name'] != 'legacy_low_order': raise ValueError('unknown pool design')
    return list(range(count))

def rectangular_pool_indices(height: int, width: int, count: int,
                             design: dict | None = None) -> list[int]:
    design=design or {'name':'rectangular_cosine_v1'}
    if design.get('name') not in ('rectangular_cosine_v1','rect_dct_v1'):
        raise ValueError('rectangular templates require rectangular_cosine_v1')
    if not 0 < count <= height*width: raise ValueError('invalid rectangular pool size')
    return list(range(count))


def hadamard(indices: list[int], size: int=256, device: str='cpu', bit_depth: int=8) -> PatternBatch:
    order=pool_order(size,size*size); bits=int(math.log2(size))
    def row(sequency: int) -> torch.Tensor:
        gray=sequency^(sequency>>1); code=int(f'{gray:0{bits}b}'[::-1],2)
        values=torch.arange(size,device=device,dtype=torch.int64)&code
        parity=torch.zeros(size,device=device,dtype=torch.int64)
        for b in range(bits): parity^=(values>>b)&1
        return (1-2*parity).double()
    p=torch.stack([torch.outer(row(order[i][0]),row(order[i][1])).flatten() for i in indices])
    return quantize(p,[f'walsh:{i}' for i in indices],bit_depth=bit_depth)

def rectangular_patterns(indices: list[int], height: int, width: int, device: str='cpu', bit_depth: int=8) -> PatternBatch:
    """Realizable [-1,1] separable cosine-sign patterns for HxW non-power-of-two grids."""
    order=rectangular_order(height,width,height*width)
    yy=torch.arange(height,device=device,dtype=torch.float64)[:,None]
    xx=torch.arange(width,device=device,dtype=torch.float64)[None,:]
    rows=[]
    for idx in indices:
        u,v=order[int(idx)]
        sy=torch.cos(math.pi*(yy+0.5)*u/height)
        sx=torch.cos(math.pi*(xx+0.5)*v/width)
        rows.append(torch.where(sy*sx >= 0, torch.ones((height,width),device=device,dtype=torch.float64),
                                -torch.ones((height,width),device=device,dtype=torch.float64)).flatten())
    return quantize(torch.stack(rows),[f'rectcos:{i}' for i in indices],bit_depth=bit_depth)

def pattern_batch(indices: list[int], height: int, width: int, device: str='cpu', design: dict | None=None,
                  bit_depth: int=8) -> PatternBatch:
    name=(design or {}).get('name','legacy_low_order')
    if name in ('rectangular_cosine_v1','rect_dct_v1'):
        return rectangular_patterns(indices,height,width,device,bit_depth)
    if height != width or height & (height-1):
        raise ValueError('Walsh patterns require a square power-of-two template')
    return hadamard(indices,height,device,bit_depth)


def constrained(p0: torch.Tensor, control: torch.Tensor, shape: int | tuple[int,int], bound: float) -> torch.Tensor:
    """Coarse control grid -> bilinear DMD perturbation, clipped to [-1,1], relative RMS <= bound."""
    shape=(shape,shape) if isinstance(shape,int) else tuple(shape)
    delta=F.interpolate(control,size=shape,mode='bilinear',align_corners=False).flatten(1)
    raw=p0+delta
    # Strict inequalities preserve inward search derivatives at saturated initialization.
    p=torch.where(raw>1,torch.ones_like(raw),torch.where(raw< -1,-torch.ones_like(raw),raw)); delta=p-p0
    ratio=delta.square().mean(-1).clamp_min(1e-30).sqrt()/p0.square().mean(-1).sqrt()
    scale=(bound/ratio.clamp_min(1e-15)).clamp(max=1)
    return p0+delta*scale[:,None]
