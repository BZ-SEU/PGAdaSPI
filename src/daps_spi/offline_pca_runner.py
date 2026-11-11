"""Fixed-bank evaluation with exact nested QR reuse, only for offline PCA.

The complete measurement schedule is known before any observation. A thin QR
of A.T supplies every prefix row space, without repeatedly factoring the bank.
No future y is used by a prefix reconstruction; no online decisions occur.
"""
import math
from pathlib import Path
import time
import torch
from .offline_pca import load_bank, bank_batch
from .types import MeasurementState
from .propagation_pinv import MeasurementFactor, PseudoinverseDecoder
from .sampling import sample, chain_seeds, sampler_steps
from .compact_output import metrics, save_result
from .io import digest


def prefix_reconstruction(factor: MeasurementFactor, mu: torch.Tensor, count: int) -> torch.Tensor:
    """Physical xhat [d] from first count measured rows/values only, full pinv."""
    q=factor.Q[:,:count]; r=factor.R[:count,:count]
    rhs=factor.y[:count]-factor.A[:count]@mu.double()
    alpha=torch.linalg.solve_triangular(r.T,rhs[:,None],upper=False).flatten()
    return mu.double()+q@alpha


def run_fixed_bank(recipe: dict, cfg: dict, settings: dict, optics, mu: torch.Tensor,
                   truth: torch.Tensor, roi: dict, noise, final_net, final_hash: str,
                   image_id: str, image_index: int, output: Path, device: str,
                   load_seconds: float) -> dict:
    """Simulate a presealed schedule and evaluate prefixes offline; GT is evaluation-only."""
    from .runner import sync
    if cfg['final_decoder']!='daps_mean':
        raise ValueError('fixed QR reuse currently requires the frozen posterior-mean final decoder')
    sync(device); begin=time.perf_counter()
    m=cfg['final_measurements']; d=cfg['image_height']*cfg['image_width']
    bank=load_bank(cfg['offline_pca']['path'],cfg['offline_pca']['sha256'])
    plus=bank['plus_uint8'][:m].to(device)
    # Chunked physical mapping avoids all-bank FFT workspace on the GPU.
    A=torch.empty((m,d),dtype=torch.float64,device=device)
    for j in range(0,m,32):
        batch=bank_batch(cfg['offline_pca'],j,min(32,m-j),optics,device)
        A[j:j+len(batch.rows)]=batch.rows
        del batch
    ids=[f'offline_pca:{j}' for j in range(m)]
    seal=digest(plus,A,ids)
    sync(device); mapped_seconds=time.perf_counter()-begin
    tick=time.perf_counter()
    y=optics.observe(truth,plus)+noise(0,m,device)  # Only after all codes are sealed.
    if seal!=digest(plus,A,ids): raise RuntimeError('fixed bank changed after sealing')
    sync(device); exposure_seconds=time.perf_counter()-tick
    state=MeasurementState(A,y,plus,ids)
    tick=time.perf_counter(); factor=MeasurementFactor(state,recipe['rank_rtol'])
    decoder=PseudoinverseDecoder(state,mu,recipe['rank_rtol'],factor)
    sync(device); factor_seconds=time.perf_counter()-tick
    sizes=list(range(cfg['initial_measurements'],m,cfg['batch_size']))+[m]
    history=[]; max_residual=0.; y0,x0,h,w=(int(roi[k]) for k in ('top','left','height','width'))
    tick=time.perf_counter()
    for count in sizes:
        start=time.perf_counter()
        rec=prefix_reconstruction(factor,mu,count).reshape(cfg['image_height'],cfg['image_width'])
        max_residual=max(max_residual,float((A[:count]@rec.flatten()-y[:count]).abs().max()))
        sync(device)
        history.append(metrics(rec[y0:y0+h,x0:x0+w],truth[y0:y0+h,x0:x0+w],count,time.perf_counter()-start))
    prefix_seconds=time.perf_counter()-tick
    # Same decision count and F seeds as the ordinary B32 acquisition schedule.
    decisions=math.ceil((m-cfg['initial_measurements'])/cfg['batch_size'])
    fcfg=dict(cfg['sampling'],**cfg['final_sampling'])
    fseeds=chain_seeds(cfg['seed'],image_index,decisions,'F',fcfg['chains'])
    start=time.perf_counter()
    ensemble=sample(final_net,state,fcfg,fseeds,'F',final_hash,factor)
    if ensemble.diagnostics['rank']!=m: raise ValueError('final projection truncated rows')
    rec=ensemble.samples.mean(0).reshape(cfg['image_height'],cfg['image_width'])
    sync(device); final_seconds=time.perf_counter()-start
    pinv_final=history[-1].copy()
    history[-1]=dict(metrics(rec[y0:y0+h,x0:x0+w],truth[y0:y0+h,x0:x0+w],m,pinv_final['seconds']+final_seconds),decoder='daps_mean')
    max_residual=max(max_residual,float((A@rec.flatten()-y).abs().max()))
    if max_residual>1e-6*(1+float(y.abs().max())): raise RuntimeError('measurement consistency failed')
    settings['final_pinv']=pinv_final
    sampler,steps=sampler_steps(fcfg)
    settings['final_daps']=dict(sampler=sampler,chains=fcfg['chains'],steps=steps,outer_steps=fcfg['outer_steps'],seeds=fseeds,
                                checkpoint_sha256=final_hash,nfe_per_chain=ensemble.diagnostics['nfe_per_chain'])
    settings['chain_seeds']=[{} for _ in range(decisions)]
    settings['fixed_bank_execution']=dict(rule='all codes presealed; prefix metrics use exact nested unpivoted QR; no future y in prefix reconstructions',seal=seal,
                                           prefix_times='offline evaluation times, not incremental hardware timings')
    settings['numerics']=dict(max_condition=factor.condition,max_measurement_residual=max_residual,final_decoder='daps_mean',
                               rank_policy='full rank; no truncation, regularization or damping',
                               condition_definition='full A condition bounds every prefix condition')
    settings['input_region']=roi
    settings['budget_definition']=dict(rule=recipe['measurement_rule'],roi_pixels=h*w,canvas_pixels=d,final_measurements=m)
    capture=decoder.captured_energy_ratio(truth)
    runtime=dict(adaptive_wall_seconds=time.perf_counter()-begin,model_load_seconds=load_seconds,
                 components=dict(O_seconds=0.,search_seconds=mapped_seconds,exposure_seconds=exposure_seconds,
                                 decode_seconds=factor_seconds+prefix_seconds,final_seconds=final_seconds),
                 device=device,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),GPU=torch.cuda.get_device_name(device),
                 timing_definition='synchronized fixed-bank simulation plus exact prefix evaluation and final sampling; excludes publishing; offline PCA fit separately')
    print(dict(image_id=image_id,final=history[-1],seconds=runtime['adaptive_wall_seconds']),flush=True)
    del ensemble
    return save_result(output/image_id,image_id,plus,ids,history,truth,rec,settings,runtime,roi,information_capture=capture)
