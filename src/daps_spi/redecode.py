"""Offline final-decoder ablations on sealed simulated exposures; never invokes a policy."""
from pathlib import Path
import copy
import gc
import json
import math
import time
import torch
from .compact_output import metrics, save_result
from .data import load_image, load_image_template
from .io import digest, file_hash
from .priors import load_prior
from .propagation_pinv import IntensityOptics, OpticalConfig, MeasurementFactor, PseudoinverseDecoder, LinearMMSEDecoder, SensorDarkNoise
from .sampling import sample, chain_seeds, sampler_steps
from .types import MeasurementState

ROOT=Path(__file__).resolve().parents[2]


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


def selection_signature(recipe):
    """Only acquisition semantics; final decoder and execution batch sizes are excluded."""
    r=copy.deepcopy(recipe); c=r['config']; o=r['options']
    gate=o.pop('validation_gate',False)
    if gate: raise ValueError('source used V; no-V acquisition equivalence is not established')
    c['sampling'].pop('validation_scenarios',None)
    for key in ('device','final_decoder','final_sampling','shared_factor'):
        c.pop(key,None)
    return dict(config=c,options=o,physics=r['physics'],mean=r['mean'],prior=r['prior'],
                rank_rtol=r['rank_rtol'],measurement_rule=r['measurement_rule'])


def verified_source(source_root, name):
    marker=source_root/name/(name+'.json'); source=json.loads(marker.read_text())
    if source['status']!='complete' or source['image_id']!=name: raise ValueError('incomplete source')
    if file_hash(marker.with_suffix('.pt'))!=source['pt_sha256']: raise ValueError('source PT integrity failure')
    if not marker.with_suffix('.png').exists(): raise ValueError('source PNG missing')
    payload=torch.load(marker.with_suffix('.pt'),map_location='cpu',weights_only=False)
    if digest(payload['plus_uint8'])!=source['patterns']['plus_hash']: raise ValueError('source pattern hash mismatch')
    if len(payload['plus_uint8'])!=source['final']['M']: raise ValueError('source budget mismatch')
    if any('V' in seeds for seeds in source['settings']['chain_seeds']): raise ValueError('source actually sampled V')
    return marker,source,payload


def run_redecode(source_root, jobs, names, device):
    """Each job is (resolved recipe, output directory). F16 may reuse identical F64 prefix chains.

    A [M,d] and y [M] are reconstructed from immutable uint8 exposures in the original
    noiseless simulator; y is not a newly selected observation. GT never reaches sampling.
    Intermediate curves retain the source pinv decoder; only the endpoint is replaced.
    """
    loaded={}; hashes={}; code_hash={p.name:file_hash(p) for p in sorted((ROOT/'src/daps_spi').glob('*.py'))}
    for name in names:
        marker,source,payload=verified_source(source_root,name)
        s=source['settings']; original=s['recipe']; cfg=s['config']; plus=payload['plus_uint8']
        expected_signature=selection_signature(original)
        pending=[]
        frozen_path=ROOT/'assets/frozen_assets.json'
        frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
        for recipe,out in jobs:
            if selection_signature(recipe)!=expected_signature:
                raise ValueError(f'{name}: requested config changes acquisition; cannot reuse source')
            if file_hash(ROOT/recipe['prior'])!=s['checkpoint_sha256']: raise ValueError('selection prior changed')
            final_prior=recipe.get('final_prior') if recipe['config']['final_decoder']=='daps_mean' else None
            asset_names=[recipe['mean']]+([final_prior] if final_prior else [])
            if recipe['config']['final_decoder']=='lmmse': asset_names.append(recipe['prior_spectrum'])
            for asset in asset_names:
                if asset not in hashes: hashes[asset]=file_hash(ROOT/asset)
                if asset in frozen and hashes[asset]!=frozen[asset]: raise ValueError(f'frozen asset changed: {asset}')
            provenance=dict(source_json=str(marker.resolve()),source_json_sha256=file_hash(marker),
                            source_pt_sha256=source['pt_sha256'],source_pattern_hash=source['patterns']['plus_hash'],
                            rule='Offline noiseless replay of sealed codes; final decoder only; no action selection',
                            assets={a:hashes[a] for a in asset_names})
            identity=digest('final-redecode-v1',provenance,recipe,device,code_hash)
            dest=out/name/(name+'.json')
            if dest.exists():
                old=json.loads(dest.read_text())
                if old['settings']['run_identity']!=identity: raise ValueError(f'{dest}: incompatible output identity')
                if old['status']!='complete' or file_hash(dest.with_suffix('.pt'))!=old['pt_sha256'] or not dest.with_suffix('.png').exists():
                    raise ValueError(f'{dest}: corrupt completed output')
                print(f'{out.name}/{name}: verified complete, skipped',flush=True);continue
            pending.append((recipe,out,provenance,identity))
        if not pending: continue
        sync(device); replay_begin=time.perf_counter()
        dataset=s['evaluation_dataset']; image_path=Path(dataset['root'])/dataset['image_files'][name]
        if file_hash(image_path)!=s['image_sha256'] or hashes[original['mean']]!=s['mean_sha256']:
            raise ValueError('image/mean identity changed')
        if dataset.get('template_rectangular',False):
            truth,roi=load_image_template(image_path,cfg['image_height'],cfg['image_width'],rotate_portrait=cfg.get('orientation')=='landscape_clockwise90_for_portrait')
        else:
            truth=load_image(image_path,cfg['image_size'],native=dataset.get('native',False));roi=source['evaluation_region']
        truth=truth.to(device).double(); mu=torch.load(ROOT/original['mean'],map_location=device,weights_only=False)['mu']
        optics=IntensityOptics(OpticalConfig(**original['physics']),device)
        A=torch.empty((len(plus),plus[0].numel()),dtype=torch.float64,device=device)
        buckets=[]
        for start in range(0,len(plus),32):
            codes=plus[start:start+32].to(device)
            A[start:start+32]=optics.effective_rows((2*codes.double()-255)/255)
            buckets.append(optics.observe(truth,codes))
        clean=torch.cat(buckets)
        forward_error=float((A@truth.flatten()-clean).abs().max())
        # Same simulator noise as the source (seeded by image index and row index); absent key = noiseless.
        noise=SensorDarkNoise(cfg.get('sensor_noise'),optics,cfg['seed'],s['image_index'])
        if s.get('sensor_noise',dict(sigma_rel=0.)).get('sigma_rel',0.)!=noise.sigma_rel: raise ValueError('sensor noise setting mismatch')
        y=clean+noise(0,len(clean),clean.device)
        if forward_error>1e-8*(1+float(y.abs().max())): raise ValueError('replayed matrix/exposure mismatch')
        state=MeasurementState(A,y,plus.to(device).flatten(1),source['patterns']['ids'])
        sync(device); replay_seconds=time.perf_counter()-replay_begin
        begin=time.perf_counter()
        factor=MeasurementFactor(state,original['rank_rtol']) if cfg.get('shared_factor',False) else None
        decoder=PseudoinverseDecoder(state,mu,original['rank_rtol'],factor)
        information_capture=decoder.captured_energy_ratio(truth)
        pinv=decoder.xhat.reshape_as(truth)
        sync(device); pinv_seconds=time.perf_counter()-begin
        y0,x0,h,w=(int(roi[k]) for k in ('top','left','height','width'))
        def measure(rec,seconds):return metrics(rec[y0:y0+h,x0:x0+w],truth[y0:y0+h,x0:x0+w],state.M,seconds)
        verified_pinv=measure(pinv,pinv_seconds)
        for key in ('mse','psnr','psnr_clamped','ssim'):
            if abs(verified_pinv[key]-s['final_pinv'][key])>1e-7:
                raise ValueError(f'{name}: original/replayed pinv mismatch at {key}: {verified_pinv[key]} vs {s["final_pinv"][key]}')
        print(json.dumps(dict(image=name,replay_seconds=replay_seconds,pinv_seconds=pinv_seconds,pinv_verified=True,forward_error=forward_error)),flush=True)
        pinv_condition=decoder.condition
        # Do not hold a full pinv QR while allocating LMMSE's whitened QR on 512 images.
        del decoder,factor
        factor=None
        gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
        # Preserve exact sampling batch shape: reuse F16 from F64 (both batch16), not F8.
        pending.sort(key=lambda job:(job[0]['config']['final_decoder']=='daps_mean',-job[0]['config'].get('final_sampling',{}).get('chains',0)))
        prefix_cache={}
        for recipe,out,provenance,identity in pending:
            target=recipe['config']; kind=target['final_decoder']; result_settings=copy.deepcopy(s)
            sync(device); begin=time.perf_counter(); condition=pinv_condition; load_seconds=0.; final_daps=None
            if kind=='pinv':
                rec=pinv; elapsed=pinv_seconds
            elif kind=='lmmse':
                spectrum=torch.load(ROOT/recipe['prior_spectrum'],map_location=device,weights_only=False)['spectrum']
                linear=LinearMMSEDecoder(state,mu,spectrum,recipe['rank_rtol']);rec=linear.xhat.reshape_as(truth);condition=linear.condition
                sync(device);elapsed=time.perf_counter()-begin;del linear,spectrum
            elif kind=='daps_mean':
                if cfg.get('shared_factor',False) and factor is None:
                    factor=MeasurementFactor(state,original['rank_rtol'])
                final_path=recipe['final_prior']
                if final_path not in loaded:
                    tick=time.perf_counter();loaded[final_path]=load_prior(str(ROOT/final_path),device);sync(device);load_seconds=time.perf_counter()-tick
                net,meta,checkpoint_hash=loaded[final_path]
                fcfg=dict(target['sampling'],**target['final_sampling'])
                for key in ('sigma_min','sigma_max'):
                    if fcfg[key]!=meta[key]: raise ValueError('sampling/prior sigma mismatch')
                remaining=cfg['final_measurements']-cfg['initial_measurements']
                decisions=math.ceil(remaining/cfg['batch_size'])
                seeds=chain_seeds(cfg['seed'],s['image_index'],decisions,'F',fcfg['chains'])
                if seeds[:min(len(seeds),32)]!=s['final_daps']['seeds'][:min(len(seeds),32)]: raise ValueError('F seed schedule changed')
                key=digest(fcfg,checkpoint_hash,seeds)
                if key in prefix_cache:
                    rec,elapsed,diag=prefix_cache.pop(key)
                    provenance['shared_sampling']='Exact prefix of same F64 sampling call; same seeds, batch shape and projection'
                else:
                    sync(device);tick=time.perf_counter()
                    ens=sample(net,state,fcfg,seeds,'F',checkpoint_hash,factor)
                    sync(device);elapsed=time.perf_counter()-tick;diag=ens.diagnostics
                    if diag['rank']!=state.M: raise ValueError('final sampler truncated measurement rank')
                    rec=ens.samples.mean(0).reshape_as(truth)
                    for other,_,_,_ in pending:
                        if other['config']['final_decoder']!='daps_mean' or other.get('final_prior')!=final_path:continue
                        ocfg=dict(other['config']['sampling'],**other['config']['final_sampling']);n=ocfg['chains']
                        if n>=len(seeds) or n%fcfg['gpu_batch_size'] or dict(ocfg,chains=fcfg['chains'])!=fcfg:continue
                        batch_count=n//fcfg['gpu_batch_size'];partial=diag['batch_cumulative_seconds'][batch_count-1]+(elapsed-diag['seconds'])
                        pdiag=dict(diag,nfe_set=diag['nfe_per_chain']*n,batch_calls=diag['nfe_per_chain']*batch_count)
                        prefix_cache[digest(ocfg,checkpoint_hash,seeds[:n])]=(ens.samples[:n].mean(0).reshape_as(truth),partial,pdiag)
                    # Diagnose replay: the first 32 chains of an F64 with the source's own F settings reproduce its F32.
                    source_f=dict(original['config']['sampling'],**original['config']['final_sampling'])
                    if len(seeds)==64 and sampler_steps(fcfg)==sampler_steps(source_f):
                        check=measure(ens.samples[:32].mean(0).reshape_as(truth),0.)
                        provenance['F32_reproduction_metric_differences']={k:check[k]-source['final'][k] for k in ('mse','psnr','ssim')}
                    del ens
                sampler,steps=sampler_steps(fcfg)
                final_daps=dict(sampler=sampler,chains=fcfg['chains'],steps=steps,outer_steps=fcfg['outer_steps'],seeds=seeds,
                                checkpoint_sha256=checkpoint_hash,nfe_per_chain=diag['nfe_per_chain'])
            else: raise ValueError(kind)
            residual=float((A@rec.flatten()-y).abs().max())
            if residual>1e-6*(1+float(y.abs().max())): raise ValueError(f'{name}/{out.name}: measurement inconsistency {residual}')
            history=copy.deepcopy(payload['history']);history[-1]=dict(measure(rec,elapsed),decoder=kind)
            for row in history[:-1]:row['decoder']='pinv'
            provenance.update(A_hash=digest(A),y_hash=digest(y),forward_error=forward_error,pinv_check_tolerance=1e-7,
                              curve_definition='Reused original pinv intermediate metrics; only final M is decoded with requested decoder')
            result_settings.update(config=copy.deepcopy(target),recipe=copy.deepcopy(recipe),derived_from=provenance,
                                   run_identity=identity,software_hashes=code_hash,final_pinv=verified_pinv)
            result_settings['config']['device']=device
            result_settings.pop('final_daps',None)
            if final_daps:result_settings['final_daps']=final_daps
            result_settings['numerics'].update(final_decoder=kind,max_measurement_residual=residual,final_condition=condition)
            original_final=source['runtime']['components']['final_seconds'];acquisition=source['runtime']['adaptive_wall_seconds']-original_final
            runtime=dict(adaptive_wall_seconds=acquisition+(0. if kind=='pinv' else elapsed),
                         model_load_seconds=load_seconds,device=device,GPU=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
                         components=dict(reused_acquisition_seconds=acquisition,final_seconds=0. if kind=='pinv' else elapsed),
                         offline_replay_seconds=replay_seconds,offline_pinv_check_seconds=pinv_seconds,final_redecode_seconds=elapsed,
                         peak_gpu_allocated_bytes=None,
                         timing_definition='Derived total = source acquisition/pinv trajectory excluding source F + measured new final decoder. Not a new closed-loop wall-time benchmark. Pinv already present in acquisition; offline reconstruction time is reported separately.')
            save_result(out/name,name,plus,state.pattern_ids,history,truth,rec,result_settings,runtime,roi,
                        information_capture=information_capture)
            print(json.dumps(dict(experiment=out.name,image_id=name,final=history[-1],residual=residual)),flush=True)
            if kind=='lmmse':del rec
            gc.collect()
        del state,A,y,factor,pinv,prefix_cache,truth,optics,payload
        gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
