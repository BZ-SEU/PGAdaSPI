#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. 2: real-data method overview, cameraman, archived E32 no-gate trajectory.

Run this standalone file once to replay archived O/F seeds and save all plotting
arrays. Subsequent runs reuse NPZ automatically. --plot-data PATH redraws on CPU
without the source files or prior; --resample explicitly regenerates the data.
Only this figure's PNG/JSON/NPZ are written. No experiment code is modified.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import SymLogNorm
def save_figure(fig, path, **kwargs):
    """Save PNG or vector EMF (requires Inkscape); intermediates are temporary."""
    import shutil
    import struct
    import subprocess
    import tempfile

    path = Path(path)
    if path.suffix.lower() != '.emf':
        fig.savefig(path, **kwargs)
        return
    executable = shutil.which('inkscape') or shutil.which('inkscape-paper')
    if executable is None:
        local = Path.home() / '.local/bin/inkscape-paper'
        executable = str(local) if local.is_file() else None
    if executable is None:
        raise RuntimeError('EMF export requires Inkscape (inkscape on PATH).')
    with tempfile.TemporaryDirectory(prefix='paper_figure_emf_') as directory:
        temporary = Path(directory)
        source = temporary / 'figure.pdf'
        fig.savefig(source, format='pdf', **kwargs)
        converted = source.with_suffix('.emf')
        result = subprocess.run(
            [executable, str(source), '--pdf-poppler', '--export-type=emf',
             '--export-filename=' + str(converted)],
            capture_output=True, text=True, timeout=180)
        if result.returncode or not converted.exists():
            raise RuntimeError('EMF conversion failed: ' + result.stdout + result.stderr)
        content = converted.read_bytes()
        if (len(content) < 88 or content[40:44] != b' EMF'
                or struct.unpack_from('<I', content, 48)[0] != len(content)):
            raise RuntimeError('Invalid EMF output: ' + str(path))
        path.write_bytes(content)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def generate(args):
    import torch
    from daps_spi.io import digest
    from daps_spi.data import load_image_template
    from daps_spi.priors import load_prior
    from daps_spi.propagation_pinv import IntensityOptics, OpticalConfig, PseudoinverseDecoder, MeasurementFactor
    from daps_spi.propagation_policy import shrink_scenarios, eigen_center
    from daps_spi.compact_output import metrics
    from daps_spi.sampling import sample
    from daps_spi.physics import model_measurements
    from daps_spi.reconstruction import AffineProjector
    from daps_spi.types import MeasurementState
    source = args.results/'cameraman/cameraman.json'
    meta = json.loads(source.read_text()); s = meta['settings']; cfg = s['config']; recipe = s['recipe']
    require(meta['status'] == 'complete' and meta['method'] == 'propagation-full-pseudoinverse', 'Invalid source')
    require(recipe['options']['candidate_design'] == {'name':'eigen_v1','rows':32}, 'Expected E32')
    require(not any('V' in x for x in s['chain_seeds']), 'Expected no-gate source')
    require(cfg['initial_measurements'] == 64 and cfg['batch_size'] == 32, 'Expected M64/B32')
    require(64 <= args.before_m < 640 and (args.before_m-64)%32 == 0, 'Select a full archived adaptive batch')
    pt = source.with_suffix('.pt'); require(sha(pt) == meta['pt_sha256'], 'Source PT hash mismatch')
    payload = torch.load(pt, map_location='cpu', weights_only=True); plus = payload['plus_uint8']
    require(plus.dtype == torch.uint8 and list(plus.shape) == [655,256,256], 'Invalid codes')
    require(digest(plus) == meta['patterns']['plus_hash'], 'Pattern hash mismatch')
    paths = {'image': ROOT/'data/set11_proxunroll/native'/s['evaluation_dataset']['image_files']['cameraman'],
             'prior': ROOT/recipe['prior'], 'mean': ROOT/recipe['mean']}
    for key, field in [('image','image_sha256'),('prior','checkpoint_sha256'),('mean','mean_sha256')]:
        require(sha(paths[key]) == s[field], f'{key} hash mismatch')
    frozen = json.loads((ROOT/'assets/frozen_assets.json').read_text())
    require(s['checkpoint_sha256'] == frozen['outputs/prior_daps_v2/final_65k.pt'], 'Expected frozen v2 prior')
    require(s['final_daps']['checkpoint_sha256'] == s['checkpoint_sha256'], 'Final prior differs')
    physics = s['optics']['parameters']
    require(physics['image_pitch'] == physics['dmd_pitch'] == 10.8e-6, 'Physical pitch mismatch')
    require(meta['patterns']['max_relative_rms'] <= cfg['optimization']['rms_bound']+1e-12, 'Archived RMS violation')
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    gt, _ = load_image_template(paths['image'],256,256); gt = gt.to(args.device).double()
    mu = torch.load(paths['mean'],map_location=args.device,weights_only=True)['mu'].flatten().double()
    optics = IntensityOptics(OpticalConfig(**physics),args.device)
    net, _, loaded = load_prior(str(paths['prior']),args.device)
    require(loaded == s['checkpoint_sha256'], 'Loaded prior mismatch')
    codes = plus.flatten(1).to(args.device)
    A = torch.cat([optics.effective_rows((2*c.double()-255)/255) for c in codes.split(32)])
    y = optics.observe(gt,codes); tolerance = 1e-6*(1+float(y.abs().max()))
    forward = float((A@gt.flatten()-y).abs().max())
    require(forward <= tolerance, 'Forward/adjoint mismatch')
    ids = meta['patterns']['ids']; m = args.before_m
    state = MeasurementState(A[:m],y[:m],codes[:m],ids[:m])
    factor = MeasurementFactor(state,recipe['rank_rtol']) if cfg.get('shared_factor',False) else None
    dec = PseudoinverseDecoder(state,mu,recipe['rank_rtol'],factor)
    history = {x['M']:x for x in payload['history']}
    require(m in history and m+32 in history, 'Missing batch boundary')
    pinv_metrics = metrics(dec.xhat.reshape(256,256),gt,m,0.)
    require(abs(pinv_metrics['psnr']-history[m]['psnr']) < 1e-3, 'Archived pinv replay mismatch')
    sampling = copy.deepcopy(cfg['sampling']); sampling.pop('validation_scenarios',None)
    require(sampling.get('sampler') == 'diffpir' and sampling['diffpir_steps'] == 50 and sampling['num_scenarios'] == 32, 'Expected DiffPIR50 O32')
    seeds = s['chain_seeds'][(m-64)//32]['O']
    print(f'M={m}: replaying archived O32 / DiffPIR50',flush=True)
    ens = sample(net,state,sampling,seeds,'O',loaded,factor)
    require(ens.diagnostics['rank'] == m, 'O projection lost rank')
    O = ens.samples; residual = float((O@state.A.T-state.y).abs().max())
    require(residual <= tolerance and bool(torch.isfinite(O).all()), 'O consistency failed')
    shrunk = shrink_scenarios(O,recipe['options']['spread_weight'])
    e = shrunk.double()-dec.xhat
    _, singular, U = torch.linalg.svd(e,full_matrices=False)
    reserved = [int(x.split(':')[1]) for x in ids[m:m+32]]
    # B=k=32: greedy IDs are reservation labels only; no pool search is needed.
    replay = eigen_center(dec,shrunk,{'indices':reserved},None,optics,32,32,directions=U)
    actual = codes[m:m+32]
    same = (replay.plus == actual).double().mean(1)
    flipped = (255-replay.plus == actual).double().mean(1)
    agreement = torch.maximum(same,flipped)
    actual_rows = A[m:m+32]
    R = dec.residual_rows(actual_rows)
    alignment = (R*U).sum(1)/(R.norm(dim=1)*U.norm(dim=1))
    # Eigenvector signs are arbitrary: orient displayed directions to their actual rows.
    oriented = U*torch.where(alignment < 0,-1.,1.)[:,None]
    e_residual = float((e@state.A.T).abs().max())
    require(e_residual <= tolerance, 'Residuals leave unmeasured subspace')
    print(f'O residual={residual:.3e}; replayed code agreement (sign invariant)={float(agreement.mean()):.6f}',flush=True)
    def cpu(t): return t.detach().cpu().numpy()
    project = factor.project_model if factor is not None else AffineProjector(*model_measurements(state.A,state.y),sampling['projection_rtol'])
    initial_base = project(torch.zeros(1,256*256,device=args.device,dtype=torch.float64)).reshape(256,256)
    noise = torch.randn((1,256,256),generator=torch.Generator(device=args.device).manual_seed(seeds[0]),device=args.device)
    initial = initial_base + ens.diagnostics['outer_nodes'][0]*noise[0]
    o_diagnostics = ens.diagnostics
    require(o_diagnostics['nfe_per_chain'] == 49, 'Unexpected DiffPIR NFE')
    data = dict(gt=cpu(gt), O=cpu(O.reshape(32,256,256)), pinv=cpu(dec.xhat.reshape(256,256)),
                shrunk=cpu(shrunk.reshape(32,256,256)), residuals=cpu(e.reshape(32,256,256)),
                directions=cpu(oriented.reshape(32,256,256)), singular_values=cpu(singular),
                executed_codes=cpu(actual.reshape(32,256,256)), effective_rows=cpu(actual_rows.reshape(32,256,256)),
                A=cpu(state.A), y=cpu(state.y), initial_state=cpu(initial), initial_base=cpu(initial_base),
                initial_code=plus[0].numpy(), row_alignment=cpu(alignment.abs()),
                replay_code_agreement=cpu(agreement))
    del dec, O, e, shrunk, U, R, replay
    final_state = MeasurementState(A,y,codes,ids)
    fcfg = dict(cfg['sampling'],**cfg['final_sampling']); fcfg.pop('validation_scenarios',None)
    fseeds = s['final_daps']['seeds']
    require(fcfg.get('sampler') == 'diffpir' and fcfg['diffpir_steps'] == 50 and len(fseeds) == 32, 'Expected DiffPIR50 F32')
    final_reuse = None
    if args.final_data:
        with np.load(args.final_data,allow_pickle=False) as saved:
            previous=json.loads(str(saved['metadata']))
            require(previous['source_json_sha256'] == sha(source)
                    and previous['source_pt_sha256'] == meta['pt_sha256']
                    and previous['prior_sha256'] == loaded
                    and previous['F_seeds'] == fseeds
                    and previous['F_sampling'] == fcfg
                    and previous['core_hashes'] == {p.name:sha(p) for p in sorted((ROOT/'src/daps_spi').glob('*.py'))},
                    'Final plotting data provenance mismatch')
            F=torch.from_numpy(saved['F']).to(args.device).reshape(32,-1)
            require(bool(torch.isfinite(F).all()), 'Nonfinite cached F samples')
        final_reuse=dict(path=str(args.final_data.resolve()),sha256=sha(args.final_data))
        print(f'M=655: reusing saved F32/DiffPIR50 from {args.final_data}',flush=True)
    else:
        print('M=655: replaying archived F32 / DiffPIR50',flush=True)
        final_factor = MeasurementFactor(final_state,recipe['rank_rtol']) if cfg.get('shared_factor',False) else None
        final_ens = sample(net,final_state,fcfg,fseeds,'F',loaded,final_factor)
        require(final_ens.diagnostics['rank'] == 655, 'Final projection lost rank')
        F = final_ens.samples
    mean = F.mean(0).reshape(256,256)
    fres = float((F@A.T-y).abs().max())
    require(fres <= tolerance, 'Final sample consistency failed')
    fm = metrics(mean,gt,655,0.)
    require(abs(fm['psnr']-meta['final']['psnr']) < 1e-3, 'Final mean disagrees with archived PSNR')
    require(abs(fm['ssim']-meta['final']['ssim']) < 1e-4, 'Final mean disagrees with archived SSIM')
    data.update(F=cpu(F.reshape(32,256,256)),final_mean=cpu(mean),final_y=cpu(y))
    report = dict(image_id='cameraman',before_m=m,after_m=m+32,final_m=655,
                  source_json=str(source.resolve()),source_json_sha256=sha(source),source_pt_sha256=meta['pt_sha256'],
                  image_sha256=s['image_sha256'],prior_sha256=loaded,mean_sha256=s['mean_sha256'],
                  source_run_identity=s['run_identity'],sampling_script_sha256=sha(__file__),
                  reused_final_data=final_reuse,
                  core_hashes={p.name:sha(p) for p in sorted((ROOT/'src/daps_spi').glob('*.py'))},
                  O_diagnostics=o_diagnostics,O_seeds=seeds,F_seeds=fseeds,O_sampling=sampling,F_sampling=fcfg,physics=physics,
                  spread_weight=recipe['options']['spread_weight'],pinv_metrics=pinv_metrics,final_metrics=fm,
                  forward_residual=forward,O_residual=residual,F_residual=fres,residual_nullspace_error=e_residual,
                  residual_tolerance=tolerance,code_replay_agreement=cpu(agreement).tolist(),
                  definitions={'O':'raw unshrunk O32/DiffPIR50; first three chains displayed without GT selection',
                    'F':'F32/DiffPIR50 at M655 using archived F seeds; mean of all 32 unshrunk unclipped chains',
                    'residuals':'e_l = mean(O) + sqrt(alpha)*(O_l-mean(O)) - centered full pinv',
                    'directions':'right singular vectors of samples-as-rows residual matrix; sign aligned to executed rows',
                    'second_moment':'E.T @ E / 32, not covariance centered about sample mean; SVD avoids d by d matrix',
                    'A_display':'all measurement rows, 256 equally spaced flattened pixel columns; no row normalization',
                    'y':'differential complementary-exposure bucket values regenerated with archived physical model',
                    'codes':'verbatim archived positive uint8 exposures; complementary negative=255-positive',
                    'initial_state':'actual first O chain model-coordinate Proj(0) + sigma_start * randn; archived seed',
                    'diffpir':'single denoiser per update; clamp x0; soft measurement correction; re-noise; exact final projection of last re-noised state',
                    'rho':'lambda * max(0.001,sigma_y)^2 / sigma_t^2, whitened measurement correction',
                    'initial_display':'model-coordinate initial state, fixed grayscale range [-60,60]',
                    'scope':'offline replay of sealed measurements, no retraining or adaptive acquisition rerun'})
    print(f'F residual={fres:.3e}; archived final metrics reproduced: {fm["psnr"]:.4f}/{fm["ssim"]:.4f}',flush=True)
    return data, report


def draw(data, r):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig = plt.figure(figsize=(24,13),facecolor='white')
    navy='#12264b'; blue='#287de0'; purple='#8846d8'; teal='#009c9c'; orange='#d59600'
    def text(x,y,s,size=11,color=navy,ha='center',weight='normal'):
        fig.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight)
    def box(x,y,w,h,color,fill='white'):
        fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.008',
            transform=fig.transFigure,facecolor=fill,edgecolor=color,linewidth=1.5,zorder=0))
    def arrow(x1,y1,x2,y2,color=orange,scale=17):
        fig.add_artist(FancyArrowPatch((x1,y1),(x2,y2),transform=fig.transFigure,arrowstyle='-|>',
                        mutation_scale=scale,color=color,linewidth=2,zorder=3))
    def pic(arr,x,y,w=.054,cmap='gray',vmin=0,vmax=1,label=None):
        h=w*24/13
        ax=fig.add_axes([x,y,w,h]); ax.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.set_axis_off()
        if label: text(x+w/2,y-.013,label,9)
        return ax
    def strip(arr,x,y,w=.047,gap=.006,cmap='gray',vmin=0,vmax=1):
        for i in range(3): pic(arr[i],x+i*(w+gap),y,w,cmap,vmin,vmax)
    m=r['before_m']; end=r['after_m']
    text(.018,.995,'(a) Adaptive acquisition and reconstruction',20,ha='left',weight='bold')
    # The upper loop returns the sealed new batch to acquisition.
    for x1,y1,x2,y2 in [(.88,.918,.88,.944),(.88,.944,.15,.944),(.15,.944,.15,.916)]:
        arrow(x1,y1,x2,y2,orange,14)
    text(.52,.956,'Lock batch → acquire new measurements',12,orange,weight='bold')
    for x,w,title,col in [(.018,.272,'Single-pixel acquisition',blue),(.319,.17,'Measurement state (A, y)',blue),
                          (.518,.214,'Posterior sampling',purple),(.761,.221,'DMD pattern generation',teal)]:
        box(x,.743,w,.171,col); text(x+w/2,.895,title,14,weight='bold')
    pic(data['gt'],.027,.778,.052,label='Object')
    pic(data['executed_codes'][0],.104,.797,.040,vmax=255)
    pic(255-data['executed_codes'][0],.148,.797,.040,vmax=255)
    text(.146,.774,'Complementary uint8 codes',9)
    arrow(.082,.827,.102,.827,blue)
    arrow(.191,.827,.211,.827,blue)
    ax=fig.add_axes([.218,.799,.058,.062]); ax.plot(data['y'],color=blue,lw=.7)
    ax.set_yscale('symlog',linthresh=max(float(np.quantile(np.abs(data['y']),.75)),1e-12))
    ax.set_xticks([]);ax.set_yticks([]);ax.spines[['top','right']].set_visible(False)
    text(.247,.774,'Bucket signal (symlog)',8)
    arrow(.291,.831,.318,.831)
    av=data['A'][:,np.linspace(0,65535,256,dtype=int)]; lim=float(np.abs(av).max())
    ax=fig.add_axes([.331,.786,.111,.083]);ax.imshow(av,cmap='RdBu_r',norm=SymLogNorm(max(float(np.quantile(np.abs(av),.75)),1e-12),vmin=-lim,vmax=lim),aspect='auto');ax.set_axis_off()
    ax=fig.add_axes([.457,.786,.016,.083]);ax.imshow(data['y'][:,None],cmap='RdBu_r',aspect='auto',norm=SymLogNorm(max(float(np.quantile(np.abs(data['y']),.75)),1e-12),vmin=-float(np.abs(data['y']).max()),vmax=float(np.abs(data['y']).max())));ax.set_axis_off()
    text(.387,.769,f'A: {m} × 65,536',9);text(.465,.769,'y',10)
    arrow(.490,.831,.517,.831)
    strip(data['O'],.534,.781,.057,.008)
    text(.625,.76,'O: 32 chains | DiffPIR50 | 49 NFE/chain',10)
    arrow(.733,.831,.760,.831)
    strip(data['executed_codes'],.775,.781,.057,.008,vmax=255)
    text(.871,.76,f'Executed E32 batch: M={m} → {end}',10)
    # Final branch, a separate F ensemble after all measurements are sealed.
    arrow(.405,.741,.405,.691,purple)
    text(.300,.713,'Budget complete: M=655',10,purple)
    box(.365,.592,.185,.095,purple,'#faf6ff')
    text(.458,.649,'Final posterior sampling',14,purple,weight='bold')
    text(.458,.620,'F: 32 chains | DiffPIR50 | 49 NFE/chain',11)
    arrow(.551,.639,.581,.639,purple)
    strip(data['F'],.586,.586,.054,.007)
    text(.673,.574,'Final posterior samples',10)
    arrow(.764,.639,.788,.639,purple)
    box(.794,.608,.080,.065,purple,'#faf6ff');text(.834,.64,'Sample\nmean',12,purple,weight='bold')
    arrow(.876,.639,.909,.639,purple)
    pic(data['final_mean'],.918,.586,.054,label='Final reconstruction')
    text(.025,.661,'cameraman  |  256 × 256',13,ha='left',weight='bold')
    text(.025,.628,'Archived E32 trajectory, no validation gate',11,ha='left')
    text(.025,.598,f'Design state: M={m}; batch size: 32',11,ha='left')
    # b: DiffPIR soft data updates and a distinct final exact projection.
    box(.018,.072,.397,.457,purple)
    text(.029,.505,'(b) DiffPIR measurement-conditioned sampling',16,ha='left',weight='bold')
    text(.216,.465,r'$z_{init}=\mathrm{Proj}(0)+\sigma_{start}\epsilon$',13,purple)
    pic(data['initial_state'],.034,.326,.067,vmin=-60,vmax=60,label='Initial state (O₁ seed)')
    arrow(.105,.388,.123,.388,purple)
    box(.128,.351,.096,.075,purple,'#f3eaff')
    text(.176,.389,'Denoise\n+ clamp',13,purple,weight='bold')
    arrow(.225,.388,.246,.388,purple)
    box(.250,.351,.144,.075,blue,'#eff6ff')
    text(.322,.389,'Soft measurement\nconsistency',12,blue,weight='bold')
    text(.260,.328,r'$\widetilde z_0=z_0+[\mathrm{Proj}(z_0)-z_0]/(1+\rho_t)$',10)
    arrow(.322,.350,.322,.292,purple,12)
    arrow(.322,.292,.176,.292,purple,12)
    arrow(.176,.292,.176,.350,purple,12)
    text(.245,.274,'Re-noise; 49 denoiser calls',10,purple)
    arrow(.322,.292,.404,.292,blue,12)
    arrow(.404,.292,.404,.230,blue,12)
    arrow(.404,.230,.385,.230,blue,12)
    box(.268,.198,.117,.057,blue,'#eff6ff')
    text(.326,.227,'Final exact projection',10,blue,weight='bold')
    arrow(.268,.224,.247,.224,blue,12)
    strip(data['O'],.040,.110,.061,.014)
    text(.146,.091,'O samples; same frozen prior, independent noise',8,purple)
    text(.326,.172,r'$Ax=y$',13,blue)
    text(.326,.128,'Max. |Ax − y|\n'+f'{r["O_residual"]:.2e}',10,blue)
    # c: actual shrinkage and uncentered second moment, no GT enters this block.
    box(.437,.072,.545,.457,teal)
    text(.449,.505,'(c) Second-moment direction design',18,ha='left',weight='bold')
    text(.450,.467,'1. Shrink the sample spread, then form residuals',12,ha='left',weight='bold')
    text(.710,.438,r'$\widetilde{x}_\ell=\overline{x}_O+\sqrt{\alpha}(x_\ell-\overline{x}_O),\quad \alpha=0.25$',12)
    strip(data['shrunk'],.452,.338,.043,.004)
    text(.520,.322,'Shrunk O samples',9)
    text(.607,.379,'−',22)
    pic(data['pinv'],.630,.338,.045,label=r'$\hat{x}=\mu+A^+(y-A\mu)$')
    text(.704,.379,'=',20)
    # Display signed errors with one common symmetric scale, preserving signs.
    rlim=float(np.quantile(np.abs(data['residuals'][:3]),.99))
    strip(data['residuals'],.765,.338,.048,.008,'RdBu_r',-rlim,rlim)
    text(.845,.322,r'$e_\ell=\widetilde{x}_\ell-\hat{x}$  (signed residuals)',10)
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cax=fig.add_axes([.934,.344,.005,.087])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(-rlim,rlim),cmap='RdBu_r'),cax=cax,extend='both')
    cb.set_ticks([-rlim,0,rlim]);cb.ax.tick_params(labelsize=7,length=2)
    # Formula placed explicitly to avoid implying pinv minus sample.
    text(.720,.292,r'$E=[e_1,\ldots,e_{32}],\qquad S=EE^{\mathsf{T}}/32$',13)
    text(.450,.252,'2. Residual second-moment directions → effective rows',12,ha='left',weight='bold')
    text(.486,.187,'SVD\nof E',12,teal,weight='bold');arrow(.514,.184,.535,.184,teal)
    dlim=float(np.quantile(np.abs(data['directions'][:3]),.99))
    strip(data['directions'],.542,.128,.048,.006,'RdBu_r',-dlim,dlim)
    text(.617,.112,r'$u_1, u_2, u_3$'+f'  (color ±{dlim:.3f})',9)
    arrow(.701,.184,.723,.184,teal)
    text(.757,.209,'Optical inverse',9);text(.757,.187,'+ uint8 quantization',9)
    text(.757,.165,'+ forward optics',9)
    arrow(.793,.184,.810,.184,teal)
    effective_limit=float(np.abs(data['effective_rows'][:3]).max())
    strip(data['effective_rows'],.819,.128,.045,.005,'RdBu_r',-effective_limit,effective_limit)
    text(.891,.112,'Effective rows 1–3',10)
    text(.891,.096,f'Shared linear color scale: ±{effective_limit:.2e}',7)
    text(.520,.084,'Mean-error direction + sample spread; GT is not a design input.',10,ha='left')
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'outputs/ablation/full')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/paper_figures/fig2')
    p.add_argument('--before-m',type=int,default=224)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--plot-data',type=Path)
    p.add_argument('--resample',action='store_true')
    p.add_argument('--final-data',type=Path,help='Reuse verified F32/DiffPIR50 from another Fig. 2 NPZ when generating a new design stage')
    args=p.parse_args()
    require(not (args.resample and args.plot_data),'--resample conflicts with --plot-data')
    require(args.dpi>=72,'dpi must be >=72')
    for protected in [args.results, ROOT/'data']:
        require(not args.output.resolve().is_relative_to(protected.resolve()),'Output must be outside source data')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.plot_data or args.output/f'fig2_cameraman_M{args.before_m}_data.npz'
    start=time.perf_counter()
    if args.plot_data or (path.exists() and not args.resample):
        with np.load(path,allow_pickle=False) as saved:
            require(int(saved['format_version'])==2,'Old plotting data; regenerate with --resample')
            require(json.loads(str(saved['metadata']))['O_sampling'].get('sampler') == 'diffpir', 'Expected DiffPIR plotting data')
            report=json.loads(str(saved['metadata']))
            data={k:saved[k] for k in saved.files if k not in ['metadata','format_version']}
        if not args.plot_data:
            require(report['source_json_sha256']==sha(args.results/'cameraman/cameraman.json'),'Source changed; use --resample')
        print(f'Redrawing from {path}; no sampling',flush=True)
    else:
        data,report=generate(args)
        require(all(np.isfinite(v).all() for v in data.values()),'Nonfinite plot array')
        temp=path.with_suffix('.npz.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f,format_version=np.array(2),metadata=np.array(json.dumps(report,allow_nan=False)),**data)
        temp.replace(path)
        print(f'Saved plotting data: {path}',flush=True)
    fig=draw(data,report)
    stem=args.output/'fig2_cameraman'
    for ext in ['png', 'emf']:
        save_figure(fig, stem.with_suffix('.'+ext),dpi=args.dpi,facecolor='white',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    report['display'] = dict(A_y='symlog with linear threshold at 75th percentile of absolute displayed values',
        residual_limit=float(np.quantile(np.abs(data['residuals'][:3]),.99)),
        direction_limit=float(np.quantile(np.abs(data['directions'][:3]),.99)),
        effective_rows='actual quantized exposure codes propagated through archived optics; raw signed physical rows; common symmetric linear scale without clipping',
        effective_row_limit=float(np.abs(data['effective_rows'][:3]).max()),
        signed_maps='one common symmetric 99th-percentile scale per type; 1% saturation for display only',
        image_range=[0,1],code_range=[0,255],shown_chains=[0,1,2],shown_batch_rows=[1,2,3])
    report.update(script_sha256=sha(__file__),plotting_data=str(path.resolve()),plotting_data_sha256=sha(path))
    stem.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Finished: {stem}.png / .json ({time.perf_counter()-start:.1f} s)',flush=True)

if __name__=='__main__':
    main()
