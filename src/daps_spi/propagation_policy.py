"""GT-free adaptive pattern selection; every DMD row is passed through the calibrated optics."""
import math
import random
import torch
from dataclasses import replace
from .physics import quantize
from .io import digest
from .propagation_pinv import PseudoinverseDecoder
from .types import MeasurementState, ScenarioEnsemble, ActionDecision
from .patterns import pattern_batch,constrained,pool_indices,rectangular_pool_indices
from .types import PatternBatch
from .objectives import gains


def mapped(batch,optics):
    batch.rows=optics.effective_rows((2*batch.plus.double()-255)/255)
    return batch


class OpticalPool:
    def __init__(self,state,decoder,optics,pool_size,chunk,design=None,bit_depth: int=8):
        reserved=set(state.pattern_ids)
        self.height=optics.height; self.width=optics.width
        self.side=self.height if self.height==self.width else None
        self.design=design or {'name':'legacy_low_order'}
        pool=(rectangular_pool_indices(self.height,self.width,pool_size,self.design)
              if self.height != self.width or (self.height & (self.height-1))
              else pool_indices(self.height,pool_size,self.design))
        prefix='rectcos:' if self.design.get('name') in ('rectangular_cosine_v1','rect_dct_v1') else 'walsh:'
        self.ids=[i for i in pool if f'{prefix}{i}' not in reserved]
        self.device=str(state.A.device)
        self.decoder=decoder;self.chunk=chunk;self.optics=optics;self.bit_depth=bit_depth
        self.B=torch.cat([mapped(pattern_batch(self.ids[j:j+chunk],self.height,self.width,self.device,self.design,
                                                bit_depth=self.bit_depth),optics).rows for j in range(0,len(self.ids),chunk)]).float()
        self.Q=torch.empty_like(self.B)
        with torch.no_grad():
            for j in range(0,len(self.B),chunk):
                b=self.B[j:j+chunk].double();self.Q[j:j+chunk]=decoder.residual_rows(b).float()
    @torch.no_grad()
    def greedy(self,scenarios: torch.Tensor,count: int=16) -> tuple[dict,list[dict]]:
        """Sequential exact ensemble marginal gain; temporary predictions retain true past y."""
        Q=self.Q.clone(); e=scenarios.double()-self.decoder.xhat; mask=torch.zeros(len(self.B),device=self.B.device,dtype=torch.bool); chosen=[]; log=[]
        for step in range(count):
            scores=[]
            for j in range(0,len(self.B),self.chunk):
                b=self.B[j:j+self.chunk].double(); q=Q[j:j+self.chunk].double(); denom=self.decoder.lam+(b*q).sum(1)
                alpha=(e@b.T)/denom
                score=(2*alpha*(e@q.T)-alpha.square()*q.square().sum(1)).mean(0)/e.shape[1]
                scores.append(score)
            scores=torch.cat(scores); scores[mask]=-torch.inf; selected=int(scores.argmax()); q=Q[selected].double().clone(); denom=self.decoder.lam+self.B[selected].double()@q
            alpha=(e@self.B[selected].double())/denom; e=e-alpha[:,None]*q[None]
            log.append(dict(step=step+1,index=self.ids[selected],remaining=int((~mask).sum()),O_marginal_gain=float(scores[selected])))
            chosen.append(self.ids[selected]); mask[selected]=True
            for j in range(0,len(self.B),self.chunk): Q[j:j+self.chunk]-=(((Q[j:j+self.chunk].double()@self.B[selected].double())[:,None]*q[None]/denom).float())
        batch=self.batch(dict(indices=chosen))
        exact=float(gains(self.decoder,batch.rows,scenarios).mean())
        assert abs(exact-sum(r['O_marginal_gain'] for r in log))<1e-5
        return dict(source='ensemble_greedy_all_unused',indices=chosen,O_gain=exact),log

    @torch.no_grad()
    def contiguous(self,scenarios: torch.Tensor,batch_size: int=16) -> list[dict]:
        """Every consecutive whole batch-size group from ALL unused indices; not all combinations."""
        e=scenarios.double()-self.decoder.xhat; records=[]
        for begin in range(0,len(self.B)-batch_size+1,batch_size):
            b=self.B[begin:begin+batch_size].double(); q=self.Q[begin:begin+batch_size].double()
            S=b@q.T+self.decoder.lam*torch.eye(batch_size,device=b.device,dtype=b.dtype)
            coeff=torch.linalg.solve(S,(e@b.T).T).T
            value=(2*(coeff*(e@q.T)).sum(1)-((coeff@(q@q.T))*coeff).sum(1))/e.shape[1]
            records.append(dict(source='contiguous_all_unused',indices=self.ids[begin:begin+batch_size],O_gain=float(value.mean())))
        return records

    def batch(self,record):
        return mapped(pattern_batch(record['indices'],self.height,self.width,self.device,self.design,
                                    bit_depth=self.bit_depth),self.optics)


def optimize(dec,initial,scenarios,cfg,updates,optics):
    B=len(initial.rows); grid=cfg['control_grid']; grid=(grid,grid) if isinstance(grid,int) else tuple(grid)
    control=torch.zeros(B,1,grid[0],grid[1],device=initial.rows.device,dtype=torch.float64,requires_grad=True)
    adam=torch.optim.Adam([control],lr=cfg['learning_rate']);best=None;bestscore=-math.inf
    for step in range(updates+1):
        with torch.no_grad():
            batch=mapped(quantized_control(initial,control,cfg,optics),optics)
            score=float(gains(dec,batch.rows,scenarios).mean())
        if score>bestscore:best,bestscore=batch,score
        if step==updates:break
        adam.zero_grad()
        for chunk in scenarios.split(cfg['scenario_chunk']):
            p=constrained(initial.continuous,control,(optics.height,optics.width),cfg['rms_bound'])
            objective=gains(dec,optics.effective_rows(p),chunk).sum()/len(scenarios)
            (-objective).backward()
        adam.step()
    return best,bestscore


def unused_ids(state,cfg,optics):
    """Canonical pool IDs not yet measured, in sequency order."""
    design=cfg.get('pattern_pool') or {'name':'legacy_low_order'}
    pool=(rectangular_pool_indices(optics.height,optics.width,cfg['pool_size'],design)
          if optics.height != optics.width or (optics.height & (optics.height-1))
          else pool_indices(optics.height,cfg['pool_size'],design))
    prefix='rectcos:' if design.get('name') in ('rectangular_cosine_v1','rect_dct_v1') else 'walsh:'
    reserved=set(state.pattern_ids)
    return [i for i in pool if f'{prefix}{i}' not in reserved],design


def commit(state,batch,basis):
    """Seal an O-designed or non-adaptive batch before any new physical observation."""
    formal=replace(batch,continuous=batch.continuous.clone(),plus=batch.plus.clone(),minus=batch.minus.clone(),
                   rows=batch.rows.clone(),ids=batch.ids.copy(),constraints=dict(batch.constraints))
    basis=dict(basis,physical_hash=digest(formal.plus,formal.minus,formal.rows),reservation_ids=formal.ids.copy())
    return ActionDecision(formal,digest(state.state_hash,formal.plus),state.state_hash,basis)


def choose(state,dec,O,cfg,options,optics):
    """GT-free selection of the next batch of B effective rows.

    options.candidate_design.name (ablation switch; eigen_v1 is the final method):
      fixed    - next B unused rows in sequency order (non-adaptive; O unused)
      random   - B unused pool rows drawn by a fixed RNG (non-adaptive; O unused)
      greedy   - O-ensemble greedy Walsh rows from the pool
      eigen_v1 - top-k residual directions of the shrunk O scenarios, realized through the optics
      adasense_pca - AdaSense rule: top-k principal components of the mean-centred O samples
                 (posterior covariance only, no bias term, no shrinkage), realized like eigen_v1
      adam     - best pool batch + 3-start Adam refinement of the DMD codes (earlier design)
    Commit the O-designed candidate directly; no independent review ensemble is used.
    """
    name=options['candidate_design']['name']; B=cfg['batch_size']; alpha=options['spread_weight']
    device=str(state.A.device)
    if name == 'offline_pca':
        from .offline_pca import bank_batch
        batch=bank_batch(cfg['offline_pca'],state.M,B,optics,device)
        return commit(state,batch,dict(method=name,rule='sealed training-only PCA order; no scenarios or observations used'))
    ids,design=unused_ids(state,cfg,optics)
    if len(ids)<B: raise ValueError('not enough unreserved IDs')
    if name in ('fixed','random'):
        chosen=ids[:B] if name=='fixed' else sorted(random.Random(cfg['seed']+state.M).sample(ids,B))
        batch=mapped(pattern_batch(chosen,optics.height,optics.width,device,design,
                                   bit_depth=cfg.get('dmd_bit_depth',8)),optics)
        return commit(state,batch,dict(method=name,rule='non-adaptive; no scenarios used'))
    if O is None or O.role!='O' or O.state_hash!=state.state_hash: raise ValueError('adaptive designs need current O scenarios')
    scenarios=shrink_scenarios(O.samples,alpha)
    k=min(options['candidate_design'].get('rows',B),B,len(scenarios))
    directions=None
    if name=='adasense_pca':
        # AdaSense (Elata et al., runners/diffusion.py): SVD of samples minus their mean.
        # Uses the raw O samples, so spread_weight and the offset mean-xhat play no role.
        raw=O.samples.double(); centred=raw-raw.mean(0,keepdim=True)
        directions=torch.linalg.svd(centred,full_matrices=False).Vh[:k]
    if options.get('lightweight_eigen', False):
        # 512 path: no pool scoring; eigen rows plus the first unused sequency rows as fill.
        if name not in ('eigen_v1','adasense_pca'): raise ValueError('lightweight path supports eigen_v1/adasense_pca only')
        # Fill rows are fixed here, not pool-scored. Building every unused row
        # and its residual for O<B wasted GiBs on 512 images (notably O8).
        fill=(mapped(pattern_batch(ids[:B-k],optics.height,optics.width,device,design,
                                   bit_depth=cfg.get('dmd_bit_depth',8)),optics)
              if k<B else None)
        candidate=eigen_center(dec,scenarios,dict(indices=ids[:B]),None,optics,k,B,directions,fill,
                               cfg.get('dmd_bit_depth',8),options['candidate_design'])
    else:
        pool=OpticalPool(state,dec,optics,cfg['pool_size'],options['pool_chunk'],design,
                         cfg.get('dmd_bit_depth',8))
        g,_=pool.greedy(scenarios,B)
        if name=='greedy':
            candidate=pool.batch(g)
        elif name in ('eigen_v1','adasense_pca'):
            # O objective is already at its optimum on these directions, so no local Adam.
            candidate=eigen_center(dec,scenarios,g,pool,optics,k,B,directions,None,
                                   cfg.get('dmd_bit_depth',8),options['candidate_design'])
        elif name=='adam':
            ranks=pool.contiguous(scenarios,B); ranks.append(g)
            ranks.sort(key=lambda r:r['O_gain'],reverse=True); unique=[]; seen=set()
            for r in ranks:
                key=tuple(r['indices'])
                if key not in seen: seen.add(key); unique.append(r)
            candidate=None; bestscore=-math.inf
            for r in unique[:options['initializations']]:
                batch,score=optimize(dec,pool.batch(r),scenarios,cfg['optimization'],cfg['optimization']['adam_steps'],optics)
                if score>bestscore: candidate,bestscore=batch,score
        else: raise ValueError(f'unknown candidate design {name}')
    return commit(state,candidate,dict(method=name,rule='O-designed candidate committed directly',
                                       spread_weight=None if name=='adasense_pca' else alpha,O_seeds=O.seeds))


def invert_rows(v: torch.Tensor, optics, lam: float) -> torch.Tensor:
    """Signed DMD patterns [n,d] in [-1,1] whose effective rows approximate v [n,d].

    Wiener inverse of the adjoint object-leg PSF, then division by the collection weight.
    """
    F=torch.fft.fft2(v.reshape(-1,optics.height,optics.width)); h=optics.h1
    w=torch.fft.ifft2(F*h/(h.abs().square()+lam*h.abs().square().max())).real
    p=(w/optics.collection.clamp_min(1e-12)).flatten(1)
    return p/p.abs().amax(1,keepdim=True)


@torch.no_grad()
def _binary_aligned_batch(dec, p: torch.Tensor, targets: torch.Tensor, optics,
                          quantiles: list[float]) -> tuple[PatternBatch, torch.Tensor, torch.Tensor]:
    """Choose per-row binary thresholds using only a target O direction and known operators.

    The score is cosine alignment of each quantized, physical, *unmeasured*
    row with the corresponding residual direction.  It is deliberately not a
    GT/held-out-bucket score.  The finite quantile grid makes the branch fully
    reproducible and avoids all-on/all-off threshold extrapolation.
    """
    if not quantiles:
        raise ValueError('alignment_quantiles must be a nonempty finite list')
    qs=torch.as_tensor(quantiles,dtype=p.dtype,device=p.device)
    if bool((~torch.isfinite(qs)).any()) or bool((qs<=0).any()) or bool((qs>=1).any()):
        raise ValueError('alignment_quantiles must lie strictly between 0 and 1')
    # torch.quantile produces a threshold per physical inverse row [B,Q].
    thresholds=torch.quantile(p,qs,dim=1).T.contiguous()
    best_score=torch.full((len(p),),-torch.inf,dtype=torch.float64,device=p.device)
    best_threshold=torch.zeros(len(p),dtype=p.dtype,device=p.device)
    best_plus=None
    target_norm=targets.double().norm(dim=1).clamp_min(1e-30)
    for j in range(thresholds.shape[1]):
        trial=mapped(quantize(p,bit_depth=1,thresholds=thresholds[:,j]),optics)
        residual=dec.residual_rows(trial.rows)
        score=(residual*targets.double()).sum(1).abs()/residual.norm(dim=1).clamp_min(1e-30)/target_norm
        take=score>best_score
        if best_plus is None:
            best_plus=trial.plus.clone()
        else:
            best_plus[take]=trial.plus[take]
        best_threshold[take]=thresholds[take,j]
        best_score=torch.maximum(best_score,score)
    assert best_plus is not None
    chosen=PatternBatch(p.clamp(-1,1),best_plus,255-best_plus,
                        torch.empty_like(p,dtype=p.dtype),[f'local:{i}' for i in range(len(p))])
    return mapped(chosen,optics),best_threshold,best_score


@torch.no_grad()
def eigen_center(dec,scenarios,greedy,pool,optics,k,B,directions=None,fill_batch=None,
                 bit_depth: int=8, candidate_design: dict | None=None):
    """Top-k residual second-moment directions of the shrunk O scenarios, made realizable.

    Rows k..B-1 are the first B-k O-greedy Walsh rows. Each eigen row borrows one of the
    remaining greedy Walsh IDs as its pool reservation, so IDs stay unique and the saved
    per-row RMS check still has a canonical center. Lambda is chosen GT-free by alignment.
    `directions` [k,d] overrides the SVD of the shrunk scenarios (null-space unit rows).
    """
    if directions is None:
        e=scenarios.double()-dec.xhat; U=torch.linalg.svd(e,full_matrices=False).Vh[:k]
    else: U=directions[:k].double()
    candidate_design=candidate_design or {}
    threshold_mode=candidate_design.get('binary_threshold_mode','zero')
    if bit_depth != 1 and threshold_mode != 'zero':
        raise ValueError('binary_threshold_mode other than zero requires dmd_bit_depth=1')
    if threshold_mode not in ('zero','alignment'):
        raise ValueError(f'unknown binary_threshold_mode {threshold_mode}')
    best=None
    for lam in [1e-10,1e-9,1e-8]:
        p=invert_rows(U,optics,lam)
        if bit_depth == 1 and threshold_mode == 'alignment':
            b,thresholds,row_align=_binary_aligned_batch(dec,p,U,optics,candidate_design.get('alignment_quantiles',[]))
        else:
            b=mapped(quantize(p,bit_depth=bit_depth),optics); R=dec.residual_rows(b.rows)
            thresholds=torch.zeros(k,dtype=p.dtype,device=p.device)
            row_align=(R*U).sum(1).abs()/R.norm(dim=1).clamp_min(1e-30)/U.norm(dim=1).clamp_min(1e-30)
        align=float(row_align.mean())
        if best is None or align>best[0]: best=(align,lam,b,thresholds,row_align)
    align,lam,eig,thresholds,row_align=best; ids=greedy['indices']
    prefix='rectcos:' if optics.height != optics.width or (optics.height & (optics.height-1)) else 'walsh:'
    parts=[eig.continuous]; codes=[eig.plus]; labels=[f'{prefix}{i}' for i in ids[B-k:]]
    if B>k:
        fill=fill_batch if fill_batch is not None else pool.batch(dict(indices=ids[:B-k]))
        parts.append(fill.continuous); codes.append(fill.plus); labels+=fill.ids
    continuous=torch.cat(parts); plus=torch.cat(codes)
    center=mapped(PatternBatch(continuous,plus,255-plus,torch.empty_like(continuous),labels),optics)
    center.constraints=dict(source='eigen_v1',eigen_rows=k,alignment=align,lambda_=lam,reserved_pool_ids=labels,
                            dmd_bit_depth=bit_depth,binary_threshold_mode=threshold_mode,
                            binary_thresholds=thresholds.detach().cpu().tolist(),
                            binary_row_alignment=row_align.detach().cpu().tolist())
    return center


def shrink_scenarios(samples: torch.Tensor, spread_weight: float) -> torch.Tensor:
    """[L,d] physical scenarios; mean + sqrt(alpha)*(sample-mean), no clipping.

    Exact quadratic gain equals G(mean)+alpha*(E G(sample)-G(mean)).
    Affine measurement consistency is retained when source scenarios are consistent.
    """
    if not 0<=spread_weight<=1: raise ValueError('spread weight must be in [0,1]')
    mean=samples.mean(0,keepdim=True)
    return mean+math.sqrt(spread_weight)*(samples-mean)


def quantized_control(initial: PatternBatch, control: torch.Tensor, cfg: dict, optics) -> PatternBatch:
    p0=initial.continuous; d=p0.shape[1]
    p=constrained(p0,control,(optics.height,optics.width),cfg['rms_bound']).detach()
    for _ in range(100):
        batch=quantize(p,initial.ids.copy(),bit_depth=cfg.get('dmd_bit_depth',8))
        rms=((batch.rows*math.sqrt(d)-p0).square().mean(1)/p0.square().mean(1)).sqrt()
        if float(rms.max())<=cfg['rms_bound']+1e-12: break
        p=p0+.99*(p-p0)
    else:
        batch=quantize(p0,initial.ids.copy(),bit_depth=cfg.get('dmd_bit_depth',8)); rms=torch.zeros(len(p0))
    batch.constraints=dict(max_relative_rms=float(rms.max()),bound=cfg['rms_bound'],center_hash=digest(initial.plus),reserved_pool_ids=initial.ids.copy())
    return batch
