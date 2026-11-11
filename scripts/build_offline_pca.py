"""Fit a fixed PCA bank using exclusively the frozen MIRFLICKR train split."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import torch
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from daps_spi.data import load_image
from daps_spi.io import file_hash, digest, atomic_save
from daps_spi.runner import load_recipe, resolve_template, ROOT
from daps_spi.propagation_pinv import OpticalConfig, IntensityOptics
from daps_spi.propagation_policy import invert_rows, mapped
from daps_spi.physics import quantize


def streamed_pca(X: torch.Tensor, rank: int, q: int, iterations: int, device: str):
    """Randomized feature-space iteration on CPU X [n,d]; only blocks enter GPU.

    Equivalent to truncated range-finder SVD; no pixel covariance and no full
    training matrix on GPU. Returns unit image-space directions [rank,d].
    """
    def multiply(v, transpose=False):
        if transpose:
            out=torch.zeros((X.shape[1],v.shape[1]),device=device)
        else:
            out=torch.empty((X.shape[0],v.shape[1]),device=device)
        for j in range(0,len(X),512):
            block=X[j:j+512].to(device)
            if transpose: out.addmm_(block.T,v[j:j+512])
            else: out[j:j+512]=block@v
        return out
    Q=torch.linalg.qr(multiply(torch.randn(len(X),q,device=device),True),mode='reduced').Q
    for iteration in range(iterations):
        Y=torch.linalg.qr(multiply(Q),mode='reduced').Q
        del Q
        Q=torch.linalg.qr(multiply(Y,True),mode='reduced').Q
        del Y
        if (iteration+1)%4==0:
            torch.cuda.synchronize(device)
            print(f'PCA subspace iteration {iteration+1}/{iterations}',flush=True)
    B=multiply(Q)
    _,S,Vh=torch.linalg.svd(B,full_matrices=False)
    directions=(Q@Vh[:rank].T).T.contiguous()
    del Q,B,Vh
    av=multiply(directions.T)
    residual=multiply(av,True)-directions.T*S[:rank].square()[None,:]
    relative=residual.norm(dim=0)/S[:rank].square()
    return directions,S[:rank],relative


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--size',type=int,choices=[256,512],default=256)
    p.add_argument('--template', choices=['BSD68_481x321_bsdv2'])
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--rank',type=int)
    p.add_argument('--oversampling',type=int,default=256)
    p.add_argument('--power-iterations',type=int,default=32)
    args=p.parse_args()
    height,width=(321,481) if args.template else (args.size,args.size)
    if args.rank is None: args.rank=int(.01*height*width)
    if args.output is None: args.output=ROOT/('assets/offline_pca_bsd68_321x481.pt' if args.template else f'assets/offline_pca_{args.size}.pt')
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(8); torch.manual_seed(20261104)
    cfg=resolve_template(load_recipe(ROOT/'configs/final.yaml'),args.template or f'SET11_{args.size}')
    manifest=json.loads((ROOT/cfg['manifest']).read_text())
    names=manifest['split']['train']
    assert not set(names)&set(manifest['split']['dev'])
    assert not set(names)&set(manifest['split']['test'])
    begin=time.perf_counter()
    def read(name):
        path=Path(manifest['root'])/name
        if args.template:
            with Image.open(path) as source:
                im=ImageOps.fit(source.convert('L'),(width,height),method=Image.Resampling.LANCZOS,centering=(.5,.5))
                row=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255).flatten()
            return row,file_hash(path)
        return load_image(path,args.size).flatten(),file_hash(path)
    X=torch.empty((len(names),height*width),dtype=torch.float32)
    hashes=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i,(row,sha) in enumerate(pool.map(read,names)):
            X[i]=row; hashes.append(sha)
    print(f'Loaded {len(X)} train-only images in {time.perf_counter()-begin:.1f}s',flush=True)
    # All train images, empirical centering, no evaluation images or d x d covariance.
    q=args.rank+args.oversampling
    t=time.perf_counter()
    if args.size==256 and not args.template:
        X=X.to(args.device); mean=X.mean(0); X-=mean
        U,S,V=torch.pca_lowrank(X,q=q,center=False,niter=args.power_iterations)
        directions=V[:,:args.rank].T.contiguous()
        av=X@directions.T
        residual=X.T@av-directions.T*S[:args.rank].square()[None,:]
        relative_residual=residual.norm(dim=0)/S[:args.rank].square()
        total_variance=float(X.square().sum())
        del U,V,av,residual
    else:
        mean=X.mean(0); X-=mean
        total_variance=sum(float(X[j:j+512].square().double().sum()) for j in range(0,len(X),512))
        directions,S,relative_residual=streamed_pca(X,args.rank,q,args.power_iterations,args.device)
    del X
    # Canonicalize arbitrary SVD signs for repeatable exposure identities.
    directions*=directions.gather(1,directions.abs().argmax(1,keepdim=True)).sign()
    torch.cuda.synchronize(args.device)
    print(f'PCA q={q} completed in {time.perf_counter()-t:.1f}s; max eigenpair residual={float(relative_residual.max()):.6g}',flush=True)
    optics=IntensityOptics(OpticalConfig(**cfg['physics']),args.device)
    codes=[]; alignment=[]; lambdas=[]
    # Same fixed regularizer grid as online eigen_center, selected by physical
    # cosine alignment only (no test image / no current measurements).
    for start in range(0,args.rank,32):
        v=directions[start:start+32].double(); best=None
        for lam in [1e-10,1e-9,1e-8]:
            b=mapped(quantize(invert_rows(v,optics,lam)),optics)
            a=(b.rows*v).sum(1).abs()/b.rows.norm(dim=1)/v.norm(dim=1)
            score=float(a.mean())
            if best is None or score>best[0]: best=(score,lam,b.plus.cpu(),a.cpu().tolist())
        codes.append(best[2]); lambdas.append(best[1]); alignment.extend(best[3])
    meta=dict(definition='Centered natural-image PCA, descending variance; all rows predetermined before test observations',
              training_count=len(names),training_ids=names,training_image_sha256=hashes,
              manifest_sha256=file_hash(ROOT/cfg['manifest']),split='train',
              preprocessing=dict(source=manifest['preprocessing'],target_shape=[height,width],resize=('PIL ImageOps.fit centered aspect crop then LANCZOS resize from 256 source, grayscale /255; no native rectangular training detail' if args.template else ('none' if args.size==256 else 'PIL LANCZOS 256 to 512, uint8 grayscale /255; no native 512 training detail'))),shape=[height,width],rank=args.rank,
              pca_center='empirical mean of all training images; decoder mean unchanged',
              algorithm=('torch.pca_lowrank' if args.size==256 and not args.template else 'CPU-block streamed randomized subspace SVD')+'; approximate top components',
              seed=20261104,oversampling=args.oversampling,power_iterations=args.power_iterations,
              eigenpair_relative_residual=relative_residual.cpu().tolist(),
              captured_training_variance=float(S[:args.rank].square().sum())/total_variance,
              physics=optics.metadata()['parameters'],bit_depth=8,
              wiener_lambda_grid=[1e-10,1e-9,1e-8],wiener_block_rows=32,
              lambda_per_block=lambdas,physical_cosine_alignment=alignment,
              selection='max mean physical-row cosine with PCA direction, per predetermined block; no Q/current measurements',
              seconds=time.perf_counter()-begin,torch_version=torch.__version__)
    atomic_save(dict(plus_uint8=torch.cat(codes),training_mean=mean.cpu(),
                     singular_values=S[:args.rank].cpu(),metadata=meta),args.output)
    args.output.with_suffix('.json').write_text(json.dumps(dict(meta,bank_sha256=file_hash(args.output)),indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ['training_count','rank','captured_training_variance','seconds']}),flush=True)
    print('max eigenpair residual:',float(relative_residual.max()),flush=True)

if __name__=='__main__': main()
