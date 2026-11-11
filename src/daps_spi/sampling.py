"""Measurement-consistent posterior chains: DAPS (default), or noiseless DDRM / DDNM / DiffPIR.

DAPS: deterministic Heun ODE to x0, then exact affine projection, re-noise.
DDRM: the AdaSense/DDRM efficient_generalized_steps update, written in EDM (VE) coordinates.
DDNM / DiffPIR: official noiseless defaults, one network call per step, same exact projection.
Every sampler ends with the exact projection, so each sample satisfies Ax=y.
"""
import math, time
import torch
from .physics import model_measurements
from .reconstruction import AffineProjector
from .types import MeasurementState, ScenarioEnsemble

def sigma_nodes(high: float, low: float, count: int, rho: float=7.) -> list[float]:
    return torch.linspace(high**(1/rho),low**(1/rho),count,dtype=torch.float64).pow(rho).tolist()

def heun(net, z: torch.Tensor, sigma: float, updates: int, low: float) -> tuple[torch.Tensor,int,list[float]]:
    """ode_steps counts updates, including final Euler-to-zero; NFE=2*updates-1.
    Each invocation goes from its own current outer sigma to zero via low, NOT next outer sigma.
    """
    nodes=sigma_nodes(sigma,min(low,sigma),updates)+[0.]; w=z; nfe=0
    for s,t in zip(nodes[:-1],nodes[1:]):
        d1=(w-net(w,s))/s; nfe+=1; predict=w+(t-s)*d1
        if t==0: w=predict
        else:
            d2=(predict-net(predict,t))/t; nfe+=1; w=w+(t-s)*(d1+d2)/2
    return w,nfe,nodes

def ddrm_sigmas(config: dict) -> tuple[list[float], list[float], int]:
    """VE noise levels (current, next) of the DDRM sweep, clipped to the prior's sigma_max.

    ddpm_linear follows AdaSense: 1000-step linear beta in [1e-4, 0.02], seq = range(0, 1000, 1000//steps),
    visited in reverse; sigma_t = sqrt((1-abar_t)/abar_t); the step after t=0 has sigma 0.
    edm uses the same rho schedule as DAPS. Returns the number of clipped levels.
    """
    steps=config['ddrm_steps']; top=config['sigma_max']
    if config.get('ddrm_schedule','ddpm_linear')=='ddpm_linear':
        betas=torch.linspace(1e-4,0.02,1000,dtype=torch.float64); abar=(1-betas).cumprod(0)
        seq=list(range(0,1000,1000//steps))
        level=[float(((1-abar[t])/abar[t]).sqrt()) for t in reversed(seq)]
    elif config['ddrm_schedule']=='edm':
        level=sigma_nodes(top,config['sigma_min'],steps,config['rho'])
    else: raise ValueError(f"unknown ddrm_schedule {config['ddrm_schedule']}")
    clipped=sum(s>top for s in level); level=[min(s,top) for s in level]
    return level,level[1:]+[0.],clipped

def _sample_ddrm(net, config, seeds, height, width, device, project):
    """Noiseless DDRM (sigma_y=0, eta_b=1) in model coordinates z=2x-1, VE form z_t=x0+sigma_t*eps.

    Per step: x0=D(z,s); eps=(z-x0)/s; null=x0+s'*(sqrt(1-eta)*eps+eta*n1) (AdaSense coefficients);
    z'=Proj(null)+s'*(n2-(Proj(n2)-Proj(0))), i.e. the null-space part of null plus the measured
    part pinned to y with fresh noise. The last step returns Proj(x0) exactly. One NFE per step.
    """
    eta=config['ddrm_eta']; cur,nxt,clipped=ddrm_sigmas(config); all_samples=[]; calls=0; cumulative=[]
    begin=time.perf_counter()
    for offset in range(0,len(seeds),config['gpu_batch_size']):
        chosen=seeds[offset:offset+config['gpu_batch_size']]
        generators=[torch.Generator(device=device).manual_seed(seed) for seed in chosen]
        def noise() -> torch.Tensor:
            return torch.stack([torch.randn(1,height,width,generator=g,device=device) for g in generators]).double()
        shape=(len(chosen),1,height,width)
        base=project(torch.zeros(len(chosen),height*width,device=device,dtype=torch.float64)).reshape(shape)
        z=base+cur[0]*noise()  # AdaSense init: H^+ y + sqrt(1-abar_T) eps, in VE scaling
        for s,t in zip(cur,nxt):
            x0=net(z.float(),s).double(); calls+=1
            if t==0:
                z=project(x0.flatten(1)).reshape(shape); break
            eps=(z-x0)/s
            null=x0+t*((1-eta)**0.5*eps+eta*noise())
            n2=noise()
            z=project(null.flatten(1)).reshape(shape)+t*(n2-(project(n2.flatten(1)).reshape(shape)-base))
        all_samples.append((z.flatten(1)+1)/2)
        if device.type=='cuda': torch.cuda.synchronize()
        cumulative.append(time.perf_counter()-begin)
    return torch.cat(all_samples),calls,cur,clipped,cumulative,begin

def _ddpm_ve_levels(timesteps: list[int], top: float) -> tuple[list[float], int]:
    """VE sigma of DDPM linear-beta timesteps (t=-1 -> 0), clipped to the prior's sigma_max."""
    betas=torch.linspace(1e-4,0.02,1000,dtype=torch.float64); abar=(1-betas).cumprod(0)
    level=[0. if t<0 else float(((1-abar[t])/abar[t]).sqrt()) for t in timesteps]
    clipped=sum(s>top for s in level)
    return [min(s,top) for s in level],clipped

def _chain_noise(seeds, height, width, device):
    generators=[torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    return lambda: torch.stack([torch.randn(1,height,width,generator=g,device=device) for g in generators]).double()

def _sample_ddnm(net, config, seeds, height, width, device, project):
    """Noiseless simplified DDNM (Wang et al., ICLR 2023; official DDNM main.py --simplified).

    Official defaults: T_sampling=100 uniform skips of the 1000-step linear schedule, eta=0.85,
    sigma_y=0 (lambda_t=1, gamma_t=sigma_t), no time travel. Per step, in VE form z=x_t/sqrt(abar):
    x0=D(z,s); x0_hat=Proj(x0) (Eq. 17, exact for arbitrary rows); eps=(z-x0)/s;
    z'=x0_hat+s'*(eta*n+sqrt(1-eta^2)*eps) (paper's DDIM update). The last step returns Proj(x0).
    """
    steps=config['ddnm_steps']; eta=config['ddnm_eta']; skip=1000//steps
    times=[t*skip for t in reversed(range(steps))]
    cur,clipped=_ddpm_ve_levels(times,config['sigma_max']); nxt=cur[1:]+[0.]
    all_samples=[]; calls=0; cumulative=[]; begin=time.perf_counter()
    for offset in range(0,len(seeds),config['gpu_batch_size']):
        chosen=seeds[offset:offset+config['gpu_batch_size']]; noise=_chain_noise(chosen,height,width,device)
        shape=(len(chosen),1,height,width)
        z=cur[0]*noise()  # x_T ~ N(0,I) in VE scaling, sigma clipped to sigma_max
        for s,t in zip(cur,nxt):
            x0=net(z.float(),s).double(); calls+=1
            x0_hat=project(x0.flatten(1)).reshape(shape)
            if t==0:
                z=x0_hat; break
            eps=(z-x0)/s
            z=x0_hat+t*(eta*noise()+(1-eta**2)**0.5*eps)
        all_samples.append((z.flatten(1)+1)/2)
        if device.type=='cuda': torch.cuda.synchronize()
        cumulative.append(time.perf_counter()-begin)
    return torch.cat(all_samples),calls,cur,clipped,cumulative,begin

def _sample_diffpir(net, config, seeds, height, width, device, project):
    """Noiseless DiffPIR (Zhu et al., CVPRW 2023; official main_ddpir_inpainting.py defaults).

    Defaults: iter_num=20, skip_type='quad', lambda=1, zeta=1, eta=0, sigma=max(0.001, 0),
    pred_xstart with x0 clamped to [-1,1], guidance_scale=1, start at t=999.
    The analytic data step for our rows uses whitened measurements, so it reduces exactly to the
    official mask formula: x0_p = x0 + (Proj(x0)-x0)/(1+rho_t), rho_t=lambda*sigma^2/sigma_t^2
    (sigma_t: VE level of the [-1,1] domain). Re-noise (eta=0): z'=x0_p+s'*(sqrt(1-zeta)*eps+sqrt(zeta)*n).
    As in the official code the last iteration skips the data step and re-noise and the known part is
    restored at the end; here that is Proj(z), which keeps every sample exactly measurement-consistent.
    The unused network call of the official final iteration is skipped (output unchanged).
    """
    steps=config['diffpir_steps']; lam=config['diffpir_lambda']; zeta=config['diffpir_zeta']
    sigma_y=max(0.001,config.get('diffpir_sigma_y',0.)); top=config['sigma_max']
    if config.get('diffpir_eta',0.)!=0.: raise ValueError('only the official eta=0 DiffPIR update is implemented')
    seq=[int(v) for v in torch.linspace(0,1000**2,steps,dtype=torch.float64).sqrt().tolist()]; seq[-1]-=1
    times=[999-t for t in seq]  # sigmas[seq[i]] = reduced_alpha_cumprod[999-seq[i]]
    cur,clipped=_ddpm_ve_levels(times,top)
    all_samples=[]; calls=0; cumulative=[]; begin=time.perf_counter()
    for offset in range(0,len(seeds),config['gpu_batch_size']):
        chosen=seeds[offset:offset+config['gpu_batch_size']]; noise=_chain_noise(chosen,height,width,device)
        shape=(len(chosen),1,height,width)
        base=project(torch.zeros(len(chosen),height*width,device=device,dtype=torch.float64)).reshape(shape)
        z=base+cur[0]*noise()  # official init: masked y plus sqrt(1-abar_T) noise, VE scaling
        for i,s in enumerate(cur[:-1]):
            x0=net(z.float(),s).double().clamp(-1,1); calls+=1
            rho=lam*sigma_y**2/s**2
            x0=x0+(project(x0.flatten(1)).reshape(shape)-x0)/(1+rho)
            eps=(z-x0)/s; t=cur[i+1]
            z=x0+t*((1-zeta)**0.5*eps+zeta**0.5*noise())
        z=project(z.flatten(1)).reshape(shape)
        all_samples.append((z.flatten(1)+1)/2)
        if device.type=='cuda': torch.cuda.synchronize()
        cumulative.append(time.perf_counter()-begin)
    return torch.cat(all_samples),calls,cur,clipped,cumulative,begin

SINGLE_STEP_SAMPLERS={'ddrm':_sample_ddrm,'ddnm':_sample_ddnm,'diffpir':_sample_diffpir}
STEP_KEYS={'daps':'outer_steps','ddrm':'ddrm_steps','ddnm':'ddnm_steps','diffpir':'diffpir_steps'}

def sampler_steps(config: dict) -> tuple[str, int]:
    """(sampler, its own step count) for metadata; DAPS counts outer steps."""
    sampler=config.get('sampler','daps')
    return sampler, int(config[STEP_KEYS[sampler]])

@torch.no_grad()
def sample(net, state: MeasurementState, config: dict, seeds: list[int], role: str, checkpoint_hash: str, factor=None) -> ScenarioEnsemble:
    if role not in ['O','C','F']: raise ValueError('invalid role')
    sampler=config.get('sampler','daps')
    if sampler!='daps' and sampler not in SINGLE_STEP_SAMPLERS: raise ValueError(f'unknown sampler {sampler}')
    if sampler=='daps' and config['ode_solver']!='heun': raise ValueError('only deterministic heun implemented')
    device=state.A.device
    shape=config.get('image_shape')
    if shape is None:
        side=math.isqrt(state.A.shape[1]); height=width=side
    else:
        height,width=map(int,shape)
    if factor is None:
        Am,ym=model_measurements(state.A,state.y)
        projector=AffineProjector(Am,ym,config['projection_rtol']); project=projector
    else:
        if factor.state_hash != state.state_hash: raise ValueError('stale projection factor')
        projector=factor; project=factor.project_model
    if sampler in SINGLE_STEP_SAMPLERS:
        if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        out,calls,nodes,clipped,cumulative,begin=SINGLE_STEP_SAMPLERS[sampler](net,config,seeds,height,width,device,project)
        if device.type=='cuda': torch.cuda.synchronize()
        per_chain=calls//max(1,math.ceil(len(seeds)/config['gpu_batch_size']))
        return ScenarioEnsemble(out,role,seeds,dict(config,num_scenarios=len(seeds)),checkpoint_hash,state.state_hash,
             dict(seconds=time.perf_counter()-begin,sampler=sampler,nfe_per_chain=per_chain,nfe_set=per_chain*len(seeds),batch_calls=calls,
                  rank=projector.rank,rtol=projector.rtol,outer_nodes=nodes,clipped_levels=clipped,batch_cumulative_seconds=cumulative,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0))
    nodes=sigma_nodes(config['sigma_max'],config['sigma_min'],config['outer_steps'],config['rho'])
    all_samples=[]; calls=0; batch_cumulative_seconds=[]
    if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    for offset in range(0,len(seeds),config['gpu_batch_size']):
        chosen=seeds[offset:offset+config['gpu_batch_size']]
        generators=[torch.Generator(device=device).manual_seed(seed) for seed in chosen]
        def noise() -> torch.Tensor:
            return torch.stack([torch.randn(1,height,width,generator=g,device=device) for g in generators])
        z=nodes[0]*noise()
        for k,s in enumerate(nodes):
            m,nfe,inner=heun(net,z,s,config['ode_steps'],config['inner_sigma_min']); calls+=nfe
            c=project(m.flatten(1)).reshape_as(m)
            if k+1<len(nodes): z=c.float()+nodes[k+1]*noise()
        all_samples.append((c.flatten(1)+1)/2)
        if device.type=='cuda': torch.cuda.synchronize()
        batch_cumulative_seconds.append(time.perf_counter()-begin)
    if device.type=='cuda': torch.cuda.synchronize()
    per_chain=len(nodes)*(2*config['ode_steps']-1)
    return ScenarioEnsemble(torch.cat(all_samples),role,seeds,dict(config,num_scenarios=len(seeds)),checkpoint_hash,state.state_hash,
         dict(seconds=time.perf_counter()-begin,nfe_per_chain=per_chain,nfe_set=per_chain*len(seeds),batch_calls=calls,
              rank=projector.rank,rtol=projector.rtol,outer_nodes=nodes,batch_cumulative_seconds=batch_cumulative_seconds,
              peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0))

def chain_seeds(base: int, image_index: int, decision: int, role: str, count: int,
                schedule: str = 'blocked100') -> list[int]:
    """Independent chain IDs; first 100 decisions retain the archived seed schedule."""
    offset = {'O': 0, 'F': 200000}[role]
    round_offset = decision * 1000 if schedule == 'legacy' else (decision // 100) * 10000000 + (decision % 100) * 1000
    return [base + image_index * 1000000 + round_offset + offset + i for i in range(count)]
