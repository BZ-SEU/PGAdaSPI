"""Adaptive-only frozen simulation runner. GT enters exposure and offline metrics only."""
from pathlib import Path
import argparse
import copy
import gc
import json
import math
from decimal import Decimal, ROUND_FLOOR
from PIL import Image
import time
import torch
import yaml
from .propagation_pinv import OpticalConfig, IntensityOptics, PseudoinverseDecoder, LinearMMSEDecoder, MeasurementFactor, SensorDarkNoise
from .propagation_policy import mapped, choose
from .patterns import pattern_batch, pool_indices, rectangular_pool_indices
from .types import MeasurementState
from .priors import load_prior
from .sampling import sample, chain_seeds, sampler_steps
from .compact_output import metrics, save_result
from .data import load_image, load_image_template
from .io import file_hash, digest

ROOT = Path(__file__).resolve().parents[2]


def sync(device: str) -> None:
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)


def run_image(recipe: dict, image_id: str, image_index: int, split: str,
              manifest: dict, device: str, output: Path, net: torch.nn.Module | None,
              checkpoint_hash: str | None, load_seconds: float, final_net: torch.nn.Module | None=None,
              final_hash: str | None=None) -> dict:
    """Run to the resolved template budget. No GT/metric argument enters choose().

    final_net (daps_mean only) decodes the sealed final state once; it never enters selection.
    """
    cfg = copy.deepcopy(recipe['config']); cfg['device'] = device
    if recipe['options']['candidate_design']['name'] == 'offline_pca':
        from .offline_pca import load_bank
        spec = cfg['offline_pca']
        spec['path'] = str((ROOT/spec['path']).resolve())
        bank = load_bank(spec['path'], spec['sha256'])
    filename = manifest.get('image_files', {}).get(image_id, image_id + '.png')
    path = Path(manifest['root']) / filename
    software = {p.name: file_hash(p) for p in sorted((ROOT/'src/daps_spi').glob('*.py'))}
    identity = digest(recipe, split, image_index, file_hash(path), checkpoint_hash, software, device, *([final_hash] if final_hash else []))
    folder = output / image_id
    marker = folder / (image_id + '.json')
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get('settings', {}).get('run_identity') != identity:
            raise ValueError(f'{marker}: incompatible result; select a fresh output directory')
        if previous.get('status') == 'complete':
            pt = folder / (image_id + '.pt')
            if not pt.exists() or file_hash(pt) != previous['pt_sha256'] or not (folder/(image_id+'.png')).exists():
                raise ValueError(f'{folder}: incomplete/corrupt result')
            print(f'{image_id}: compatible complete result, skipped', flush=True)
            return previous
    if manifest.get('template_rectangular', False):
        truth, roi = load_image_template(path, cfg['image_height'], cfg['image_width'],
                                          rotate_portrait=cfg.get('orientation')=='landscape_clockwise90_for_portrait')
    else:
        truth = load_image(path, cfg['image_size'], native=manifest.get('native',False))
        roi = dict(top=0,left=0,height=cfg['image_size'],width=cfg['image_size'],
                   original_height=cfg['image_size'],original_width=cfg['image_size'],
                   canvas_height=cfg['image_size'],canvas_width=cfg['image_size'])
    truth = truth.to(device).double()
    y0,x0,h,w = (int(roi[k]) for k in ('top','left','height','width'))
    mu = torch.load(ROOT/recipe['mean'], map_location=device, weights_only=False)['mu']
    if mu.numel()!=cfg['image_height']*cfg['image_width']: raise ValueError('mean/template shape mismatch')
    optics = IntensityOptics(OpticalConfig(**recipe['physics']), device)
    # Simulator-side sensor noise; the method never sees it (y is treated as exact).
    noise = SensorDarkNoise(cfg.get('sensor_noise'), optics, cfg['seed'], image_index)
    shared=cfg.get('shared_factor',False)
    if shared and cfg.get('final_decoder','pinv') not in ('pinv','daps_mean'): raise ValueError('shared path requires pinv or daps_mean')
    final_decoder = cfg.get('final_decoder', 'pinv')
    if final_decoder not in ('pinv', 'lmmse', 'daps_mean'): raise ValueError(f'unknown final_decoder {final_decoder}')
    if final_decoder == 'daps_mean' and final_net is None:
        raise ValueError('daps_mean needs a final prior')
    if final_decoder == 'lmmse' and cfg['image_height'] != cfg['image_width']:
        raise ValueError('lmmse uses a square circulant spectrum; use pinv or daps_mean')
    spectrum = (torch.load(ROOT/recipe['prior_spectrum'], map_location=device, weights_only=False)['spectrum']
                if final_decoder == 'lmmse' else None)

    def reconstruct(state, decoder=None):
        """Reported reconstruction; selection always uses the pinv decoder built separately."""
        d = (LinearMMSEDecoder(state, mu, spectrum, recipe['rank_rtol']) if final_decoder == 'lmmse'
             else decoder if decoder is not None else PseudoinverseDecoder(state, mu, recipe['rank_rtol']))
        return d.xhat.reshape(cfg['image_height'], cfg['image_width']), d.condition
    if cfg['image_height'] != cfg['image_width'] or (cfg['image_height'] & (cfg['image_height']-1)):
        active_pool=rectangular_pool_indices(cfg['image_height'],cfg['image_width'],cfg['pool_size'],cfg.get('pattern_pool'))
        prefix='rectcos:'
    else:
        active_pool=pool_indices(cfg['image_height'],cfg['pool_size'],cfg.get('pattern_pool')); prefix='walsh:'
    if active_pool[:cfg['initial_measurements']] != list(range(cfg['initial_measurements'])):
        raise ValueError('pool must preserve the common initial prefix')
    bit_depth=int(cfg.get('dmd_bit_depth',8))
    settings = dict(method=recipe['method'], config=cfg, recipe=recipe, seed=cfg['seed'],
                    seed_schedule='blocked100; stable index in immutable split', split=split,
                    image_index=image_index, image_sha256=file_hash(path),
                    checkpoint_sha256=checkpoint_hash, software_hashes=software,
                    evaluation_dataset=manifest if 'image_hashes' in manifest else dict(root=manifest['root'], split=split, manifest_sha256=file_hash(ROOT/recipe['manifest'])),
                    mean_sha256=file_hash(ROOT/recipe['mean']), manifest_sha256=file_hash(ROOT/recipe['manifest']),
                    optics=optics.metadata(), effective_row=optics.metadata()['effective_row'], run_identity=identity)
    settings['dmd_quantization']=dict(bit_depth=bit_depth,positive_code_levels=([0,255] if bit_depth==1 else list(range(256))),
                                      rule=('p >= per-row fixed threshold -> 255, otherwise 0' if bit_depth==1
                                            else 'round(255*(clamp(p,-1,1)+1)/2)'),
                                      complementary_negative='P_minus = 255 - P_plus')
    settings['sensor_noise']=noise.metadata()
    settings['pattern_pool']=dict(canonical_ids=active_pool,sha256=digest(active_pool),
                                 definition=('IDs index the rectangular low-frequency cosine-sign order (u+v,max(u,v),u,v); no row renormalization'
                                             if cfg.get('pattern_pool',{}).get('name') in ('rectangular_cosine_v1','rect_dct_v1')
                                             else 'IDs index the full 2D Walsh sequency order (u+v,max(u,v),u,v); no row renormalization'),
                                 reservation='Skip all IDs already measured or reserved by an optimized action; local center remains the canonical Walsh row')
    if recipe['options']['candidate_design']['name'] == 'offline_pca':
        settings['offline_pca'] = dict(cfg['offline_pca'], metadata={k:v for k,v in bank['metadata'].items() if k not in ('training_ids','training_image_sha256')})
        settings['pattern_pool'] = dict(definition='Training-only centered PCA variance order; no Walsh prefix or online design', sha256=cfg['offline_pca']['sha256'])
    if cfg.get('offline_pca',{}).get('prefix_qr_reuse',False):
        if recipe['options']['candidate_design']['name'] != 'offline_pca':
            raise ValueError('fixed bank QR reuse is only valid for offline_pca')
        from .offline_pca_runner import run_fixed_bank
        return run_fixed_bank(recipe,cfg,settings,optics,mu,truth,roi,noise,final_net,final_hash,
                              image_id,image_index,output,device,load_seconds)
    sync(device); begin=time.perf_counter()
    initial=mapped(pattern_batch(list(range(cfg['initial_measurements'])), cfg['image_height'], cfg['image_width'], device,
                                 cfg.get('pattern_pool'),bit_depth=bit_depth), optics)
    if recipe['options']['candidate_design']['name'] == 'offline_pca':
        from .offline_pca import bank_batch
        initial=bank_batch(cfg['offline_pca'],0,cfg['initial_measurements'],optics,device)
    y_initial=optics.observe(truth, initial.plus)
    state=MeasurementState(initial.rows, y_initial+noise(0, len(y_initial), y_initial.device), initial.plus, initial.ids)
    del initial
    factor=MeasurementFactor(state,recipe['rank_rtol']) if shared else None
    dec=PseudoinverseDecoder(state, mu, recipe['rank_rtol'],factor)
    information_capture=dec.captured_energy_ratio(truth)
    rec, final_condition=reconstruct(state,dec if shared else None)
    peak_bytes=0
    history=[metrics(rec[y0:y0+h,x0:x0+w], truth[y0:y0+h,x0:x0+w], state.M, 0.)]
    components=dict(O_seconds=0., search_seconds=0., exposure_seconds=0., decode_seconds=0.)
    max_residual=0.; max_condition=final_condition
    # Whole batches of batch_size, then one final partial batch so M ends exactly at final_measurements.
    remaining=cfg['final_measurements']-state.M
    sizes=[cfg['batch_size']]*(remaining//cfg['batch_size'])+([remaining%cfg['batch_size']] if remaining%cfg['batch_size'] else [])
    seeds_used=[]
    # Ablation switches decide which scenario sets the selector needs; fixed/random need none.
    design=recipe['options']['candidate_design']['name']
    roles=[] if design in ('fixed','random','offline_pca') else [('O', cfg['sampling']['num_scenarios'])]
    for decision, size in enumerate(sizes):
        sync(device); tick=time.perf_counter()
        if not shared: dec=PseudoinverseDecoder(state, mu, recipe['rank_rtol'])
        ensembles={'O':None}; round_seeds={}
        for role, count in roles:
            seeds=chain_seeds(cfg['seed'], image_index, decision, role, count)
            round_seeds[role]=seeds
            ensemble=sample(net, state, cfg['sampling'], seeds, role, checkpoint_hash, factor)
            components[role+'_seconds']+=ensemble.diagnostics['seconds']
            peak_bytes=max(peak_bytes,ensemble.diagnostics['peak_gpu_bytes'])
            if ensemble.diagnostics['rank'] != state.M:
                raise ValueError('DAPS projection would truncate physical measurement rows')
            ensembles[role]=ensemble
        seeds_used.append(round_seeds)
        sync(device); search_begin=time.perf_counter()
        action=choose(state, dec, ensembles['O'], dict(cfg, batch_size=size), recipe['options'], optics)
        if len(action.batch.rows) != size:
            raise RuntimeError(f'action has {len(action.batch.rows)} rows, expected {size}')
        seal=digest(action.batch.plus, action.batch.rows, action.action_id)
        sync(device); components['search_seconds']+=time.perf_counter()-search_begin
        observe_begin=time.perf_counter()
        y=optics.observe(truth, action.batch.plus)  # First future observation, AFTER sealing.
        y=y+noise(state.M, len(y), y.device)
        if seal != digest(action.batch.plus, action.batch.rows, action.action_id):
            raise RuntimeError('sealed action mutated')
        state=state.append(action.batch, y, action.action_id)
        sync(device); components['exposure_seconds']+=time.perf_counter()-observe_begin
        decode_begin=time.perf_counter()
        if shared:
            del dec, factor
            factor=MeasurementFactor(state,recipe['rank_rtol'])
            dec=PseudoinverseDecoder(state,mu,recipe['rank_rtol'],factor)
        rec, final_condition=reconstruct(state,dec if shared else None)
        if device.startswith('cuda'): peak_bytes=max(peak_bytes,torch.cuda.max_memory_allocated(device))
        sync(device); components['decode_seconds']+=time.perf_counter()-decode_begin
        history.append(metrics(rec[y0:y0+h,x0:x0+w], truth[y0:y0+h,x0:x0+w], state.M, time.perf_counter()-tick))
        # This uses GT only after the sealed exposure has been appended, for
        # offline reporting.  It is deliberately outside choose()/sampling.
        information_capture=dec.captured_energy_ratio(truth)
        max_condition=max(max_condition, final_condition)
        max_residual=max(max_residual, float((state.A@rec.flatten()-state.y).abs().max()))
        print(json.dumps(dict(image_id=image_id, **history[-1])), flush=True)
        del ensembles, action
        if not shared: del dec
    sync(device)
    if state.M != cfg['final_measurements']:
        raise RuntimeError(f"trajectory ended at M{state.M}, expected M{cfg['final_measurements']}")
    if final_decoder == 'daps_mean':
        # Selection is finished and every observation is sealed; the posterior mean of independent
        # measurement-consistent chains is the reported image. pinv stays in history and settings.
        fcfg=dict(cfg['sampling'], **cfg['final_sampling'])
        fseeds=chain_seeds(cfg['seed'], image_index, len(sizes), 'F', fcfg['chains'])
        final_begin=time.perf_counter()
        fens=sample(final_net, state, fcfg, fseeds, 'F', final_hash, factor if shared else None)
        if fens.diagnostics['rank'] != state.M:
            raise ValueError('final sampler projection would truncate physical measurement rows')
        pinv_final=history[-1]
        rec=fens.samples.mean(0).reshape(cfg['image_height'], cfg['image_width'])
        components['final_seconds']=time.perf_counter()-final_begin
        peak_bytes=max(peak_bytes,fens.diagnostics['peak_gpu_bytes'])
        max_residual=max(max_residual, float((state.A@rec.flatten()-state.y).abs().max()))
        history[-1]=dict(metrics(rec[y0:y0+h,x0:x0+w], truth[y0:y0+h,x0:x0+w], state.M, pinv_final['seconds']+components['final_seconds']),
                         decoder='daps_mean')
        settings['final_pinv']=pinv_final
        sampler,steps=sampler_steps(fcfg)
        settings['final_daps']=dict(sampler=sampler, chains=fcfg['chains'], steps=steps, outer_steps=fcfg['outer_steps'], seeds=fseeds,
                                    checkpoint_sha256=final_hash, nfe_per_chain=fens.diagnostics['nfe_per_chain'])
        print(json.dumps(dict(image_id=image_id, **history[-1])), flush=True)
        del fens
        sync(device)
    runtime=dict(adaptive_wall_seconds=time.perf_counter()-begin, model_load_seconds=load_seconds,
                 components=components, device=device, peak_gpu_allocated_bytes=peak_bytes,
                 GPU=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                 timing_definition='synchronized simulation wall time; excludes final file/plot publishing; load reported separately')
    settings['chain_seeds']=seeds_used
    if max_residual > 1e-6*(1+float(state.y.abs().max())):
        raise RuntimeError(f'reported reconstruction violates measurements: {max_residual}')
    settings['numerics']=dict(max_condition=max_condition, max_measurement_residual=max_residual, final_decoder=final_decoder,
                              condition_definition='pinv: sigma_max/sigma_min of A; lmmse: sigma_max/sigma_min of A C^(1/2)',
                              rank_policy='full rank required; no truncation, regularization or damping')
    settings['input_region'] = roi
    settings['budget_definition'] = dict(rule=recipe['measurement_rule'], roi_pixels=int(roi['height']*roi['width']),
                                         canvas_pixels=int(cfg['image_height']*cfg['image_width']),
                                         final_measurements=int(cfg['final_measurements']))
    return save_result(folder, image_id, state.plus, state.pattern_ids, history, truth, rec, settings, runtime, roi,
                       information_capture=information_capture)


def resolve_measurement_budget(recipe: dict) -> None:
    """Resolve floor(ratio * H * W) after every user-visible ratio override.

    ``sampling_ratio`` is intentionally a top-level, configurable experiment
    parameter.  It cannot be resolved only during template selection, because
    command-line ``--set sampling_ratio=...`` is applied afterwards.
    """
    ratio=Decimal(str(recipe.get('sampling_ratio', 0.01)))
    if not 0 < ratio <= 1:
        raise ValueError('sampling_ratio must be in (0,1]')
    cfg=recipe['config']
    cfg['final_measurements']=int((ratio*int(cfg['image_height']*cfg['image_width'])).to_integral_value(rounding=ROUND_FLOOR))
    recipe['measurement_rule']=f'floor({ratio}*H*W)'


def resolve_template(recipe: dict, template_name: str) -> dict:
    """Resolve an explicit immutable geometry/data template; no max-side inference."""
    out=copy.deepcopy(recipe)
    templates=out.get('templates',{})
    if template_name not in templates: raise ValueError(f'unknown template {template_name}; choose from {sorted(templates)}')
    profile=templates[template_name]
    for name in ['mean','prior']: out[name]=profile[name]
    out['final_prior']=profile.get('final_prior')
    # Final decoder and prior are explicit per template; all production templates use DAPS mean.
    if 'final_decoder' in profile: out['config']['final_decoder']=profile['final_decoder']
    if 'final_sampling' in profile: out['config']['final_sampling']=dict(out['config'].get('final_sampling',{}),**profile['final_sampling'])
    out['template']=template_name
    out['config']['image_height']=out['physics']['height']=int(profile['height'])
    out['config']['image_width']=out['physics']['width']=int(profile['width'])
    out['config']['image_shape']=[int(profile['height']),int(profile['width'])]
    out['config']['sampling']['image_shape']=[int(profile['height']),int(profile['width'])]
    out['config']['image_size']=int(profile['height']) if profile['height']==profile['width'] else None
    out['config']['orientation']=profile.get('orientation','native')
    out['config']['initial_measurements']=profile['initial_measurements']
    out['config']['pool_size']=profile['pool_size']
    out['config']['pattern_pool']=copy.deepcopy(profile.get('pattern_pool',out['config'].get('pattern_pool')))
    out['config']['shared_factor']=profile.get('shared_factor',False)
    out['options']['lightweight_eigen']=profile.get('lightweight_eigen',False)
    if 'batch_size' in profile:
        out['config']['batch_size']=profile['batch_size']
        out['options']['candidate_design']['rows']=profile['batch_size']
        out['config']['sampling']['num_scenarios']=profile['batch_size']
    resolve_measurement_budget(out)
    out['config']['sampling']['gpu_batch_size']=profile.get('gpu_batch_size',4)
    for name,expected in profile.get('asset_hashes',{}).items():
        if expected == 'TBD': continue
        if file_hash(ROOT/name)!=expected: raise ValueError(f'template asset hash mismatch: {name}')
    if out['physics']['image_pitch']!=10.8e-6 or out['physics']['dmd_pitch']!=10.8e-6:
        raise ValueError('templates require frozen 10.8um pitch')
    return out


# Named ablations: each is a list of dotted overrides applied after template resolution.
ABLATIONS = {
    'full': [],
    'fixed': ['options.candidate_design.name=fixed'],
    'fixed_pinv': ['options.candidate_design.name=fixed', 'config.final_decoder=pinv'],
    'random': ['options.candidate_design.name=random'],
    'greedy': ['options.candidate_design.name=greedy'],
    'adam': ['options.candidate_design.name=adam'],
    'adasense': ['options.candidate_design.name=adasense_pca'],
    'no_shrink': ['options.spread_weight=1.0'],
    'pinv_final': ['config.final_decoder=pinv'],
    'lmmse_final': ['config.final_decoder=lmmse'],
}


def apply_overrides(recipe: dict, items: list[str]) -> dict:
    """KEY.PATH=VALUE (YAML-parsed) on the resolved recipe; unknown paths are errors."""
    out=copy.deepcopy(recipe)
    for item in items:
        key,sep,raw=item.partition('=')
        if not sep: raise ValueError(f'override {item!r} must be KEY.PATH=VALUE')
        *parents,leaf=key.split('.'); node=out
        for k in parents:
            if not isinstance(node.get(k),dict): raise ValueError(f'unknown override path {key}')
            node=node[k]
        if leaf not in node:
            raise ValueError(f'unknown override path {key}')
        node[leaf]=yaml.safe_load(raw)
    # CLI overrides are deliberately applied after template expansion.  Refresh
    # the derived integer budget so sampling_ratio really is configurable.
    resolve_measurement_budget(out)
    out['overrides']=list(items)
    return out


def load_recipe(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Load commented YAML; extends is relative to this YAML, all asset paths to ROOT."""
    path=path.resolve()
    if path in seen: raise ValueError(f'cyclic config inheritance: {path}')
    child=yaml.safe_load(path.read_text())
    if not isinstance(child,dict): raise ValueError('config must be a mapping')
    parent=child.pop('extends',None)
    base=load_recipe(path.parent/parent,seen+(path,)) if parent else {}
    def merge(a,b):
        for key,value in b.items():
            if key != 'run' and isinstance(value,dict) and isinstance(a.get(key),dict): merge(a[key],value)
            else: a[key]=copy.deepcopy(value)
    merge(base,child)
    return base


def validate_recipe(recipe: dict) -> dict:
    """Validate the exact budget and estimate dense resident storage, excluding workspace."""
    if 'validation_gate' in recipe['options'] or 'validation_scenarios' in recipe['config']['sampling']:
        raise ValueError('V selection has been removed; use the current no-V YAML configuration')
    cfg=recipe['config']; d=cfg['image_height']*cfg['image_width']; m=cfg['final_measurements']
    if cfg.get('dmd_bit_depth',8) not in (1,8):
        raise ValueError('config.dmd_bit_depth must be 1 or 8')
    if not 0 < cfg['initial_measurements'] <= m <= d: raise ValueError('invalid initial/final measurement budget')
    if cfg['batch_size'] < 1: raise ValueError('batch_size must be positive')
    if not m <= cfg['pool_size'] <= d:
        raise ValueError(f'pool_size={cfg["pool_size"]} must cover all {m} measured/reserved IDs and be <= {d}')
    if recipe['options']['candidate_design']['name'] == 'offline_pca':
        from .offline_pca import load_bank
        spec=cfg['offline_pca']; bank=load_bank(str((ROOT/spec['path']).resolve()),spec['sha256'])
        if cfg.get('dmd_bit_depth',8) != 8 or bank['plus_uint8'].shape[1] != d or len(bank['plus_uint8']) < m:
            raise ValueError('offline PCA bank geometry/budget/8-bit convention mismatch')
    minimum=(17*d*m+8*m*m)/2**30  # A fp64 + Q fp64 + R fp64 + positive codes uint8
    return dict(template=recipe['template'], pixels=d, initial_measurements=cfg['initial_measurements'],
                final_measurements=m, effective_ratio=m/d, physical_exposures=2*m,
                decision_batches=math.ceil((m-cfg['initial_measurements'])/cfg['batch_size']),
                last_batch=(m-cfg['initial_measurements'])%cfg['batch_size'] or cfg['batch_size'],
                dense_storage_lower_bound_gib=minimum,
                dmd_bit_depth=cfg.get('dmd_bit_depth',8),
                limitation='excludes QR/SVD workspace, old/new state copies, network, pool and scenarios; not a peak-memory estimate')


def main() -> None:
    parser=argparse.ArgumentParser(description='Frozen propagation-full-pseudoinverse; adaptive only, three files per image.')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--image-ids', nargs='+', help='IDs without .png; must belong to --split')
    selection.add_argument('--count', type=int, help='Explicit number of images; never automatically run entire split')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    parser.add_argument('--dataset-root', type=Path, help='External evaluation PNG directory')
    parser.add_argument('--template', help='Explicit geometry/data template from final.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/propagation_full_pseudoinverse_nov')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/final.yaml', help='frozen recipe')
    parser.add_argument('--ablation', choices=sorted(ABLATIONS), default='full', help='named ablation preset (see docs/ablation_and_generalization.md)')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY.PATH=VALUE',
                        help='extra override on the resolved recipe, e.g. config.sampling.num_scenarios=16; repeatable')
    parser.add_argument('--redecode-source', type=Path, help='Reuse sealed simulated codes from this result directory; final decoding only')
    parser.add_argument('--redecode-config', type=Path, action='append', default=[], help='Additional final-decoder YAML on the same images; repeatable, one GPU per task')
    parser.add_argument('--check-config', action='store_true', help='validate assets/dataset/budget without loading networks or starting an experiment')
    parser.set_defaults(start=None,split=None,device=None,output=None,ablation=None)
    args=parser.parse_args()
    recipe=load_recipe(args.config)
    run=recipe.pop('run',{})
    allowed={'template','dataset_root','image_ids','count','start','split','device','output','ablation'}
    if set(run)-allowed: parser.error(f'unknown run keys: {set(run)-allowed}')
    explicit_selection=args.image_ids is not None or args.count is not None
    for key,value in run.items():
        if explicit_selection and key in ('image_ids','count'): continue
        if getattr(args,key) is None: setattr(args,key,value)
    for key,value in dict(start=0,split='dev',device='cuda:0',output=str(ROOT/'outputs/propagation_full_pseudoinverse_nov'),ablation='full').items():
        if getattr(args,key) is None: setattr(args,key,value)
    if args.template is None: parser.error('supply run.template in YAML or --template')
    if (args.image_ids is None)==(args.count is None): parser.error('supply exactly one of image_ids/count in YAML or CLI')
    args.output=Path(args.output)
    if args.dataset_root is not None: args.dataset_root=ROOT/args.dataset_root
    resolved=apply_overrides(resolve_template(recipe,args.template),ABLATIONS[args.ablation]+recipe.get('overrides',[])+args.overrides)
    budget=validate_recipe(resolved)
    torch.set_num_threads(4)
    for name, expected in recipe.get('asset_hashes',{}).items():
        if expected == 'TBD':
            continue
        if file_hash(ROOT/name) != expected: raise ValueError(f'frozen asset hash mismatch: {name}')
    manifest=json.loads((ROOT/recipe['manifest']).read_text())
    if args.dataset_root is not None:
        dataset=args.dataset_root.resolve()
        paths=sorted([p for p in dataset.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}])
        if not paths: parser.error(f'no PNG images in {dataset}')
        names=[p.name for p in paths]
        image_files={p.stem:p.name for p in paths}
        if len(image_files) != len(paths): raise ValueError('external dataset has duplicate image stems')
        meta=dataset/'dataset_meta.txt'
        dims={p.name:list(Image.open(p).size) for p in paths}
        resolved_template=resolved
        th,tw=resolved_template['config']['image_height'],resolved_template['config']['image_width']
        if resolved_template['config']['orientation'] == 'landscape_clockwise90_for_portrait':
            preprocessing='native Y PNG L /255; no resize or padding; portrait inputs rotated clockwise 90 degrees'
            rectangular=True
        else:
            preprocessing='native grayscale PNG L /255; explicit template dimensions'
            rectangular=False
        # Validate the requested images, not unrelated sizes in a mixed SET11
        # directory. Keep the full manifest order for stable per-image seeds.
        requested = ({Path(n).stem for n in args.image_ids} if args.image_ids
                     else {p.stem for p in paths[args.start:args.start+args.count]})
        for p in paths:
            if p.stem not in requested:
                continue
            if rectangular:
                w,h=dims[p.name]
                if sorted((h,w)) != sorted((th,tw)):
                    raise ValueError(f'{p.name}: dimensions {(h,w)} do not match template {(th,tw)}')
            elif dims[p.name] != [tw,th]:
                raise ValueError(f'{p.name}: dimensions {dims[p.name]} do not match template {(th,tw)}')
        manifest=dict(root=str(dataset), split={args.split:names}, image_files=image_files,
                      source='external evaluation only; no fit or retuning',
                      preprocessing=preprocessing, template_rectangular=rectangular, native=True,
                      template=args.template, dimensions=dims,
                      image_hashes={p.name:file_hash(p) for p in paths},
                      dataset_metadata=meta.read_text() if meta.exists() else None,
                      physics_rule='Use frozen final.yaml optics; dataset pitch metadata does not override system calibration')
    names=manifest['split'][args.split]
    if args.image_ids:
        selected=[name.removesuffix('.png') for name in args.image_ids]
    else:
        if args.count < 1 or args.start < 0 or args.start+args.count > len(names):
            parser.error('requested interval outside split')
        selected=[Path(name).stem for name in names[args.start:args.start+args.count]]
    if len(selected) != len(set(selected)): parser.error('duplicate image IDs')
    indices=[]
    for name in selected:
        filename=manifest.get('image_files', {}).get(name, name+'.png')
        if filename not in names: parser.error(f'{name} is not in {args.split}')
        indices.append(names.index(filename))
    loaded={}
    frozen_path=ROOT/'assets/frozen_assets.json'
    frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    needs_selection_prior = resolved['options']['candidate_design']['name'] not in ('fixed', 'random', 'offline_pca')
    required_assets = ['mean', 'manifest']
    if needs_selection_prior: required_assets.append('prior')
    if resolved['config']['final_decoder'] == 'daps_mean': required_assets.append('final_prior')
    if resolved['config']['final_decoder'] == 'lmmse': required_assets.append('prior_spectrum')
    for key in required_assets:
        name=resolved.get(key)
        if name in frozen and file_hash(ROOT/name)!=frozen[name]:
            raise ValueError(f'frozen asset hash mismatch: {name}')
    for key in required_assets:
        if not (ROOT/resolved[key]).is_file(): raise FileNotFoundError(ROOT/resolved[key])
    if resolved['config']['final_decoder']=='daps_mean' and not (ROOT/resolved['final_prior']).is_file():
        raise FileNotFoundError(resolved['final_prior'])
    print(json.dumps(dict(config=str(args.config),device=args.device,images=selected,budget=budget,overrides=resolved['overrides']),ensure_ascii=False),flush=True)
    if args.check_config: return
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        free,_=torch.cuda.mem_get_info(args.device)
        if budget['dense_storage_lower_bound_gib']*2**30 > free:
            raise RuntimeError(f'Dense solver resident lower bound {budget["dense_storage_lower_bound_gib"]:.1f} GiB exceeds free GPU memory {free/2**30:.1f} GiB; no experiment started. Workspace needs additional memory.')
    if args.redecode_config and args.redecode_source is None:
        parser.error('--redecode-config requires --redecode-source')
    if args.redecode_source is not None:
        from .redecode import run_redecode
        jobs=[(resolved,args.output.resolve())]
        for path in args.redecode_config:
            other=load_recipe(path); control=other.pop('run')
            if control['template']!=args.template: parser.error('redecode jobs must use the same template')
            if not set(selected)<=set(control.get('image_ids',[])): parser.error('redecode jobs must cover the selected image IDs')
            extra=apply_overrides(resolve_template(other,args.template),ABLATIONS[control.get('ablation','full')]+other.get('overrides',[]))
            validate_recipe(extra)
            jobs.append((extra,(ROOT/control['output']).resolve()))
        run_redecode(args.redecode_source.resolve(),jobs,selected,args.device)
        return
    for name, index in zip(selected, indices):
        net=checkpoint_hash=None
        load_seconds=0.
        if needs_selection_prior:
            prior=resolved['prior']
            if prior not in loaded:
                sync(args.device); begin=time.perf_counter()
                net,meta,checkpoint_hash=load_prior(str(ROOT/prior),args.device)
                sync(args.device); loaded[prior]=(net,meta,checkpoint_hash,time.perf_counter()-begin)
            net,meta,checkpoint_hash,load_seconds=loaded[prior]
            if any(meta[k]!=resolved['config']['sampling'][k] for k in ['sigma_min','sigma_max']):
                raise ValueError('prior/sampling sigma mismatch')
        final_net=final_hash=None
        if resolved['config'].get('final_decoder','pinv') == 'daps_mean':
            fprior=resolved['final_prior']
            if fprior is None: raise ValueError('daps_mean requires a template final_prior')
            if fprior not in loaded:
                sync(args.device); begin=time.perf_counter()
                fnet,fmeta,fhash=load_prior(str(ROOT/fprior),args.device)
                sync(args.device); loaded[fprior]=(fnet,fmeta,fhash,time.perf_counter()-begin)
            final_net,fmeta,final_hash,_=loaded[fprior]
            if any(fmeta[k]!=resolved['config']['sampling'][k] for k in ['sigma_min','sigma_max']):
                raise ValueError('final prior/sampling sigma mismatch')
        run_image(resolved,name,index,args.split,manifest,args.device,args.output.resolve(),net,checkpoint_hash,load_seconds,
                  final_net,final_hash)
        # A 512x512 run temporarily holds several multi-GB QR/optimization
        # tensors.  Keep the loaded prior in `loaded`, but release all
        # per-image Python references and return unused CUDA blocks before the
        # next image.  This prevents the first 512 image from leaving a
        # fragmented allocator state for the next 512 image; it does not alter
        # the physical model, selector, or numerical results.
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
