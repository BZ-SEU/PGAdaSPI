#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. 1: acquisition paradigms, physical pattern synthesis, and Set11 curves.

Read-only extraction of archived scalar histories and verified Fig. 2 samples.
No acquisition, diffusion sampling, or reconstruction is rerun. Each image is
weighted equally. The x axis is nominal sampling rate: M / floor(.01*H*W),
so the native 256 and 512 results share the same 1% endpoint. Piecewise-linear
interpolation on the union of archived knots aligns their different budgets.
The last PT history entry is the posterior mean: replace it with final_pinv
for the PInv curve. Final means are plotted only at the terminal budget.

Run with /home/baojr/.conda/envs/PIE2E/bin/python figures/fig1_teaser.py
Subsequent runs reuse a compact NPZ; --refresh re-extracts source data.
HATNet 22.54 dB is a user-supplied literature reference, not a reproduced result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

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
NAMES = ['Monarch', 'Parrots', 'barbara', 'boats', 'cameraman', 'fingerprint',
         'flinstones', 'foreman', 'house', 'lena256', 'peppers256']
METHODS = {'ours': 'full', 'ada': 'adasense_ddrm', 'fixed': 'fixed'}
COLORS = {'ours': '#167EAB', 'ada': '#D98232', 'fixed': '#77828B'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def collect(results, synthesis):
    import torch
    records, identities, sources = {}, {}, []
    for key, folder in METHODS.items():
        records[key] = []
        for name in NAMES:
            path = results / folder / name / f'{name}.json'
            d = json.loads(path.read_text()); s = d['settings']; cfg = s['config']
            require(d['status'] == 'complete', f'Incomplete {path}')
            require(d['method'] == 'propagation-full-pseudoinverse', 'Unexpected physical model')
            require(sha(path.with_suffix('.pt')) == d['pt_sha256'], f'PT hash mismatch: {path}')
            identity = (s['image_sha256'], s['mean_sha256'], s['optics'], cfg['image_shape'])
            if name in identities:
                require(identity == identities[name], f'Incomparable image/mean/physics: {name}')
            identities[name] = identity
            pt = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
            history = pt['history']; end = cfg['final_measurements']
            require(end == int(.01*np.prod(cfg['image_shape'])), 'Unexpected 1% budget')
            expected = list(range(cfg['initial_measurements'], end, cfg['batch_size'])) + [end]
            require([r['M'] for r in history] == expected, 'Incomplete history')
            require(all(r.get('decoder', 'pinv') == 'pinv' for r in history[:-1]), 'Non-PInv intermediate value')
            require(history[-1]['decoder'] == 'daps_mean', 'Expected final mean')
            require(abs(history[-1]['psnr']-d['final']['psnr']) < 1e-9, 'Final mismatch')
            history = history[:-1] + [s['final_pinv']]
            require(history[-1]['M'] == end, 'PInv endpoint mismatch')
            require(all(abs(-10*np.log10(r['mse'])-r['psnr']) < 1e-8 for r in history), 'Invalid PSNR')
            sampler = s['final_daps'].get('sampler', 'daps')
            require(sampler == ('ddrm' if key == 'ada' else 'diffpir'), 'Unexpected final sampler')
            records[key].append(dict(image_id=name, image_shape=cfg['image_shape'],
                M=[r['M'] for r in history], psnr=[r['psnr'] for r in history],
                nominal_rate=[r['M']/end for r in history], final=d['final']['psnr'],
                sampler=sampler, source_json=str(path.resolve()), source_json_sha256=sha(path),
                pt_sha256=d['pt_sha256']))
            del pt
        print(f'Validated {folder}: {len(records[key])} images', flush=True)
    # The union preserves all kinks in the mean of the individual linear interpolants.
    knots = [np.array(r['nominal_rate']) for rows in records.values() for r in rows]
    start = max(x[0] for x in knots)
    grid = np.unique(np.concatenate(knots)); grid = grid[grid >= start]
    data = {'nominal_rate': grid}
    finals = {}
    for key, rows in records.items():
        per_image = np.array([np.interp(grid, r['nominal_rate'], r['psnr']) for r in rows])
        data[f'{key}_per_image'] = per_image
        data[f'{key}_pinv'] = per_image.mean(0)
        finals[key] = float(np.mean([r['final'] for r in rows]))
    prox_values = []
    for name in NAMES:
        p = ROOT/'comparable_method/ProxUnroll-main/test_results/set11_cr001_hqs'/name/f'{name}.json'
        d = json.loads(p.read_text())
        require(d['status'] == 'complete' and d['input_sha256'] == identities[name][0], 'ProxUnroll identity mismatch')
        prox_values.append(d['final']['psnr'])
        sources.append(dict(path=str(p), sha256=sha(p), measurements=d['effective_measurements']))
    prox = float(np.mean(prox_values))
    y = data['ours_pinv']; j = int(np.flatnonzero(y >= prox)[0])
    require(j > 0, 'Crossing is outside the observed interval')
    crossing = float(grid[j-1] + (prox-y[j-1])/(y[j]-y[j-1])*(grid[j]-grid[j-1]))
    with np.load(synthesis, allow_pickle=False) as f:
        sm = json.loads(str(f['metadata']))
        source = Path(sm['source_json'])
        require(sha(source) == sm['source_json_sha256'], 'Synthesis source changed')
        require(sm['O_sampling']['sampler'] == 'diffpir', 'Expected current DiffPIR synthesis')
        require(np.all(f['replay_code_agreement'] == 1), 'Quantized pattern replay mismatch')
        index = 1  # j=2, fixed choice among Fig. S2 columns; not selected by performance.
        data.update(std=f['O'].std(axis=0, ddof=1), direction=f['directions'][index],
                    codes=f['executed_codes'][index], effective_row=f['effective_rows'][index])
        align = f['row_alignment'].copy()
        require(data['codes'].dtype == np.uint8, 'Expected actual positive-exposure uint8 codes')
    source_pt = source.with_suffix('.pt')
    require(sha(source_pt) == sm['source_pt_sha256'], 'Synthesis PT changed')
    payload = torch.load(source_pt, map_location='cpu', weights_only=True)
    np.testing.assert_array_equal(data['codes'], payload['plus_uint8'][sm['before_m']+index].numpy())
    require(all(np.isfinite(a).all() for a in data.values()), 'Nonfinite plotting data')
    meta = dict(figure=1, images=NAMES, methods=records, final_means=finals,
        proxunroll=dict(mean=prox, per_image=prox_values, sources=sources),
        hatnet=dict(mean=22.54, source='User-supplied literature score; not locally reproduced'),
        crossing_nominal_percent=crossing, crossing_definition='First crossing of the piecewise-linear Set11 mean PInv curve',
        gains=dict(over_prox=float(y[-1]-prox), over_ada_pinv=float(y[-1]-data['ada_pinv'][-1])),
        synthesis=dict(image_id=sm['image_id'], before_m=sm['before_m'], after_m=sm['after_m'],
            round_t=(sm['before_m']-64)//32+1, row_j=index+1, row_alignment=float(align[index]),
            batch_mean_alignment=float(align.mean()), source=str(synthesis.resolve()), source_sha256=sha(synthesis),
            source_json=str(source), source_json_sha256=sm['source_json_sha256'], source_pt_sha256=sm['source_pt_sha256'],
            alignment_definition='cos(u_j, r_j), r_j=(I-P_A)a_j; not raw effective-row cosine'),
        definitions=dict(metric='Mean of per-image unclipped PSNR (dB); equal image weights',
            x_axis='Nominal sampling rate (%): M / floor(0.01*H*W). Integer final budgets map to 1%.',
            interpolation='Linear interpolation per image on union of archived normalized measurement knots; no smoothing',
            terminal_markers='Final 32-chain posterior means; DiffPIR for Ours/Fixed, DDRM for AdaSense*',
            baseline='AdaSense* is the project variance-only baseline, with physical propagation/uint8 realization',
            fixed='Fixed Walsh also uses physical propagation and quantization; learned baseline has ideal measurements',
            attribution='Ours-versus-AdaSense gap is a method comparison, not a controlled bias-only ablation',
            std='Sample standard deviation, ddof=1, of the same 32 O draws used for synthesis'))
    return data, meta


def draw(data, meta):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10,
                         'axes.spines.top': True, 'axes.spines.right': True})
    fig = plt.figure(figsize=(15.2, 8.0), facecolor='white')
    blue, orange, gray = [COLORS[k] for k in ['ours', 'ada', 'fixed']]
    def text(x, y, s, size=10, **kw):
        return fig.text(x, y, s, fontsize=size, **kw)
    text(.025, .995, 'Fig. 1 | Scene-adaptive physical sensing at 1%', 18, weight='bold', va='top')
    text(.025, .917, f"Linear pseudo-inverse: {data['ours_pinv'][-1]:.2f} dB  |  "
         f"+{meta['gains']['over_prox']:.2f} dB over ProxUnroll", 13, color=blue, weight='bold', va='top')
    text(.025, .858, '(a) SPI acquisition paradigms under a 1% budget', 12, weight='bold')
    cards = [(.728, gray, '#F0F3F5', 'Fixed / learned patterns',
              'Scene-independent patterns; learned decoder uses ideal measurements',
              f"Fixed + diffusion: {meta['final_means']['fixed']:.2f} dB   |   ProxUnroll: {meta['proxunroll']['mean']:.2f} dB"),
             (.607, orange, '#FFF3E8', 'Variance-only adaptive: AdaSense*',
              'Posterior spread drives acquisition; no explicit bias term',
              f"PInv: {data['ada_pinv'][-1]:.2f} dB   |   Final mean: {meta['final_means']['ada']:.2f} dB"),
             (.486, blue, '#EAF4F9', 'Ours: posterior bias + spread',
              'Closed-form second-moment gain + physical uint8 patterns',
              f"PInv: {data['ours_pinv'][-1]:.2f} dB   |   Final mean: {meta['final_means']['ours']:.2f} dB")]
    for y, color, face, title, desc, score in cards:
        fig.add_artist(FancyBboxPatch((.025,y), .475,.105,boxstyle='round,pad=0.007,rounding_size=0.012',
                       transform=fig.transFigure,fc=face,ec=color,lw=.8))
        text(.037,y+.078,title,11.5,color=color,weight='bold',va='center')
        text(.037,y+.047,desc,9.2,color='#35414A',va='center')
        text(.037,y+.018,score,10.5,weight='bold',va='center')
    text(.025,.439,'(b) Scene-adaptive physical pattern synthesis',12,weight='bold')
    s = meta['synthesis']; j=s['row_j']
    text(.025,.410,f"{s['image_id']}  |  round {s['round_t']}  |  M = {s['before_m']} → {s['after_m']}  |  j = {j}",9.5,color='#4D5862')
    u=data['direction']; a=data['effective_row']
    u=u/np.sqrt(np.mean(u*u)); a=a/np.sqrt(np.mean(a*a))
    limit=float(np.quantile(np.abs(np.stack([u,a])),.99))
    stdmax=float(np.quantile(data['std'],.99))
    maps=[data['std'],u,2*data['codes'].astype(float)/255-1,a]
    labels=[r'Sample spread $s_O$',rf'Target $u_{j}$',rf'8-bit pattern $p_{j}$',rf'Effective row $a_{j}$']
    for i,(arr,label) in enumerate(zip(maps,labels)):
        x=.025+i*.123
        ax=fig.add_axes([x,.180,.106,.2014])
        cmap,lo,hi=('magma',0,stdmax) if i==0 else (('gray',-1,1) if i==2 else ('RdBu_r',-limit,limit))
        ax.imshow(arr,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');ax.set_axis_off()
        ax.set_title(label,fontsize=9.5,pad=5)
        if i in [1,2]:
            text(x+.114,.282,'→',15,color=gray,ha='center')
    text(.025,.156,'Posterior uncertainty',8.5,color='#555555')
    text(.148,.156,'Wiener inversion',8.5,color='#555555')
    text(.271,.156,'uint8 / PWM levels',8.5,color='#555555')
    text(.394,.156,r'Propagation: $a=Kp$',8.5,color='#555555')
    text(.025,.119,rf"Residual-row alignment: {100*s['row_alignment']:.1f}% (j={j});  "
         f"{100*s['batch_mean_alignment']:.1f}% (32-row batch mean)",10,color=blue,weight='bold')
    text(.555,.858,'(c) Set11 performance versus sampling rate',12,weight='bold')
    ax=fig.add_axes([.565,.243,.405,.565])
    x=data['nominal_rate']
    styles={'ours':'-','ada':'--','fixed':':'}
    labels={'ours':'Ours (PInv)','ada':'AdaSense* (PInv)','fixed':'Fixed Walsh (PInv)'}
    for key in METHODS:
        y=data[f'{key}_pinv']; color=COLORS[key]
        ax.plot(x,y,color=color,ls=styles[key],lw=2.5,label=labels[key],zorder=4)
        ax.plot([1,1],[y[-1],meta['final_means'][key]],color=color,ls=':',lw=1.2,zorder=4)
        ax.scatter([1],[meta['final_means'][key]],marker='*',s=165,color=color,edgecolors='white',linewidth=.7,zorder=8)
    prox=meta['proxunroll']['mean']; hat=meta['hatnet']['mean']; cross=meta['crossing_nominal_percent']
    ax.axhline(prox,color='#71618E',lw=1.3,ls='-.',zorder=1)
    ax.axhline(hat,color='#A49BAE',lw=1.1,ls=(0,(2,3)),zorder=1)
    ax.text(.13,prox+.12,f'ProxUnroll  {prox:.2f}',color='#65547F',fontsize=9,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    ax.text(.13,hat-.35,f'HATNet†  {hat:.2f}',color='#81758E',fontsize=9,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    ax.scatter([cross],[prox],s=45,facecolors='white',edgecolors=blue,linewidth=1.7,zorder=7)
    ax.annotate(f'PInv crosses ProxUnroll\nat ≈{cross:.2f}% sampling',xy=(cross,prox),xytext=(.31,24.8),
                fontsize=10,color=blue,weight='bold',arrowprops=dict(arrowstyle='->',color=blue,lw=1.2),
                bbox=dict(boxstyle='round,pad=.35',fc='white',ec='#DBE9F0',alpha=.97))
    ymax=data['ours_pinv'][-1]; ada=data['ada_pinv'][-1]
    ax.annotate('',xy=(1.055,ymax),xytext=(1.055,prox),arrowprops=dict(arrowstyle='<->',color=blue,lw=1.6))
    ax.text(1.075,(ymax+prox)/2,f"+{meta['gains']['over_prox']:.2f}",color=blue,fontsize=10,weight='bold',va='center')
    ax.annotate('',xy=(1.205,ymax),xytext=(1.205,ada),arrowprops=dict(arrowstyle='<->',color=orange,lw=1.6))
    ax.text(1.225,(ymax+ada)/2,f"+{meta['gains']['over_ada_pinv']:.2f}",color=orange,fontsize=10,weight='bold',rotation=90,va='center')
    ax.text(.985,25.37,f"Final mean {meta['final_means']['ours']:.2f}",color=blue,fontsize=9,ha='right')
    ax.text(.985,24.92,f"PInv {ymax:.2f}",color=blue,fontsize=9,ha='right',weight='bold')
    for key,dy in [('ada',.12),('fixed',-.38)]:
        ax.text(1.02,meta['final_means'][key]+dy,f"{meta['final_means'][key]:.2f}",color=COLORS[key],fontsize=9,va='center')
    ax.axvline(1,color='#D4D9DD',lw=.8,zorder=0)
    ax.set(xlim=(.07,1.30),ylim=(16.1,26.25),xlabel='Nominal sampling rate (%)',ylabel='Mean PSNR (dB)')
    ax.set_xticks([.1,.2,.4,.6,.8,1.0]);ax.grid(axis='y',color='#E6EAEE',lw=.7);ax.set_axisbelow(True)
    handles, legendlabels=ax.get_legend_handles_labels()
    handles.append(Line2D([],[],marker='*',ms=11,color='#465560',ls='None'))
    legendlabels.append('Final posterior mean')
    ax.legend(handles,legendlabels,loc='lower right',fontsize=8.8,frameon=True,facecolor='white',edgecolor='#DEE4E9')
    meta['display']=dict(pattern_definition='p = 2*archived_positive_uint8/255 - 1; 256 signed levels', row_rms_normalized=True,signed_limit=limit,std_limit=stdmax,percentile=99)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results',type=Path,default=ROOT/'outputs/ablation')
    p.add_argument('--synthesis-data',type=Path,default=ROOT/'outputs/paper_figures/fig2/fig2_cameraman_M224_data.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/paper_figures/fig1')
    p.add_argument('--plot-data',type=Path,help='Redraw from a saved compact plotting NPZ')
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--dpi',type=int,default=300)
    args=p.parse_args();out=args.output.resolve()
    require(not out.is_relative_to(args.results.resolve()),'Output must be outside production results')
    require(args.dpi >= 72,'dpi must be at least 72')
    cache=args.plot_data or out/'fig1_teaser_data.npz'
    if args.plot_data or (cache.exists() and not args.refresh):
        with np.load(cache,allow_pickle=False) as f:
            require(int(f['format_version'])==1,'Unsupported plotting cache')
            data={k:f[k].copy() for k in f.files if k not in {'metadata','format_version'}}
            meta=json.loads(str(f['metadata']))
        require(all(np.isfinite(a).all() for a in data.values()),'Nonfinite cache')
        print(f'Redraw from {cache}; no experiment replay',flush=True)
    else:
        data,meta=collect(args.results,args.synthesis_data)
        out.mkdir(parents=True,exist_ok=True)
        with cache.with_suffix('.npz.tmp').open('wb') as f:
            np.savez_compressed(f,format_version=np.array(1),metadata=np.array(json.dumps(meta,allow_nan=False)),**data)
        cache.with_suffix('.npz.tmp').replace(cache)
    out.mkdir(parents=True,exist_ok=True)
    fig=draw(data,meta)
    save_figure(fig, out/'fig1_teaser.png',dpi=args.dpi,facecolor='white',bbox_inches='tight',pad_inches=.12)
    save_figure(fig, out/'fig1_teaser.emf',dpi=args.dpi,facecolor='white',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    meta.update(script_sha256=sha(__file__),plotting_data=str(cache.resolve()),plotting_data_sha256=sha(cache))
    (out/'fig1_teaser.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(f"Finished {out/'fig1_teaser.png'}; crossing={meta['crossing_nominal_percent']:.6f}%; gains={meta['gains']}",flush=True)


if __name__=='__main__':
    main()
