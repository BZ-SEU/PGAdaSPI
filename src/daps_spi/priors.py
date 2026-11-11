"""Core frozen propagation-full-pseudoinverse pipeline."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .io import file_hash

class UBlock(nn.Module):
    """Residual block with sigma-embedding scale/shift, optional channel change."""
    def __init__(self, cin: int, cout: int, embedding: int, dropout: float=0.):
        super().__init__(); g=lambda c: math.gcd(32,c)
        self.norm1=nn.GroupNorm(g(cin),cin); self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.emb=nn.Linear(embedding,2*cout); self.norm2=nn.GroupNorm(g(cout),cout)
        self.drop=nn.Dropout(dropout); self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h=self.conv1(F.silu(self.norm1(x))); scale,shift=self.emb(e)[:,:,None,None].chunk(2,1)
        h=self.norm2(h)*(1+scale)+shift
        return self.skip(x)+self.conv2(self.drop(F.silu(h)))


class Attention(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__(); self.heads=heads
        self.norm=nn.GroupNorm(math.gcd(32,channels),channels); self.qkv=nn.Conv2d(channels,3*channels,1); self.proj=nn.Conv2d(channels,channels,1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b,c,h,w=x.shape; q,k,v=self.qkv(self.norm(x)).reshape(b,3,self.heads,c//self.heads,h*w).unbind(1)
        out=F.scaled_dot_product_attention(q.transpose(-1,-2),k.transpose(-1,-2),v.transpose(-1,-2))
        return x+self.proj(out.transpose(-1,-2).reshape(b,c,h,w))


class UNetDenoiser(nn.Module):
    """EDM-x0 denoiser D(z,sigma) -> x0 in [-1,1].

    Fully convolutional (attention at the lowest resolutions); inputs are replicate-padded
    bottom/right to a multiple of 2^(levels-1) and cropped back to the original input shape.
    """
    def __init__(self, channels: int=64, sigma_data: float=.5, mult: tuple=(1,2,2,4,4),
                 blocks: int=2, attention_levels: tuple=(3,4), heads: int=4, dropout: float=0.):
        super().__init__(); self.sigma_data=sigma_data; self.factor=2**(len(mult)-1); emb=4*channels
        self.embedding=nn.Sequential(nn.Linear(16,emb),nn.SiLU(),nn.Linear(emb,emb))
        self.input=nn.Conv2d(1,channels,3,padding=1)
        self.down=nn.ModuleList(); skips=[channels]; c=channels
        for level,m in enumerate(mult):
            for _ in range(blocks):
                layer=nn.ModuleList([UBlock(c,channels*m,emb,dropout)]); c=channels*m
                if level in attention_levels: layer.append(Attention(c,heads))
                self.down.append(layer); skips.append(c)
            if level<len(mult)-1:
                self.down.append(nn.ModuleList([nn.Conv2d(c,c,3,stride=2,padding=1)])); skips.append(c)
        self.mid=nn.ModuleList([UBlock(c,c,emb,dropout),Attention(c,heads),UBlock(c,c,emb,dropout)])
        self.up=nn.ModuleList()
        for level,m in reversed(list(enumerate(mult))):
            for i in range(blocks+1):
                layer=nn.ModuleList([UBlock(c+skips.pop(),channels*m,emb,dropout)]); c=channels*m
                if level in attention_levels: layer.append(Attention(c,heads))
                if level>0 and i==blocks: layer.append(nn.Upsample(scale_factor=2,mode='nearest')); layer.append(nn.Conv2d(c,c,3,padding=1))
                self.up.append(layer)
        self.out_norm=nn.GroupNorm(math.gcd(32,c),c); self.output=nn.Conv2d(c,1,3,padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    @staticmethod
    def _run(layer, h, e):
        for m in layer: h=m(h,e) if isinstance(m,UBlock) else m(h)
        return h

    def forward(self, z: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        orig_h,orig_w=z.shape[-2:]; f=self.factor
        pad_h=(f-orig_h%f)%f; pad_w=(f-orig_w%f)%f
        if pad_h or pad_w: z=F.pad(z,(0,pad_w,0,pad_h),mode='replicate')
        s=torch.as_tensor(sigma,device=z.device,dtype=z.dtype).reshape(-1).expand(len(z)).clamp_min(1e-8)
        freq=torch.exp(torch.linspace(0,math.log(1000),8,device=z.device,dtype=z.dtype))
        phase=s.log()[:,None]/4*freq[None]; e=self.embedding(torch.cat([phase.sin(),phase.cos()],1))
        sd=self.sigma_data; sb=s[:,None,None,None]; denom=(sb.square()+sd**2).sqrt()
        h=self.input(z/denom); hs=[h]
        for layer in self.down: h=self._run(layer,h,e); hs.append(h)
        h=self._run(self.mid,h,e)
        for layer in self.up: h=self._run(layer,torch.cat([h,hs.pop()],1),e)
        out=sd**2/denom.square()*z+sb*sd/denom*self.output(F.silu(self.out_norm(h)))
        return out[..., :orig_h, :orig_w]


def build_prior(meta: dict) -> nn.Module:
    """Architecture from the frozen UNet checkpoint metadata."""
    arch=meta.get('arch')
    if arch=='unet_v2':
        return UNetDenoiser(meta['channels'],meta['sigma_data'],tuple(meta['mult']),meta['blocks'],
                            tuple(meta['attention_levels']),meta['heads'],meta.get('dropout',0.))
    raise ValueError(f'unknown prior architecture {arch}')


def load_prior(path: str, device: str) -> tuple[nn.Module,dict,str]:
    payload=torch.load(path,map_location='cpu',weights_only=False)
    meta=payload['metadata']
    if meta['domain']!='[-1,1]' or meta['parameterization']!='EDM-x0': raise ValueError('unsupported checkpoint convention')
    net=build_prior(meta).to(device); net.load_state_dict(payload['ema'],strict=True)
    net.eval().requires_grad_(False)
    return net,meta,file_hash(path)
