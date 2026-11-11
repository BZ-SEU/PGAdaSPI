"""Offline evaluation/output only. Three files per image; no policy imports.

PT contains executed positive uint8 codes [M,H,W] and scalar metric history only.
Negative codes are exactly 255-positive. GT/reconstruction are not stored in PT.
"""
from pathlib import Path
import json
import math
import os
import numpy as np
import torch
from skimage.metrics import structural_similarity
from .evaluation import mse_psnr
from .io import atomic_save, digest, file_hash
from .patterns import pattern_batch


def ssim(image: torch.Tensor, truth: torch.Tensor) -> float:
    """SSIM exactly as ProxUnroll test_proxunroll.py (CVPR 2025): reconstruction clamped to
    [0,1], both images x255, skimage.metrics.structural_similarity(data_range=255) defaults
    (7x7 uniform window, sample covariance, K1=.01, K2=.03)."""
    x = image.detach().cpu().double().clamp(0, 1).numpy()*255; y = truth.detach().cpu().double().numpy()*255
    if x.ndim != 2 or min(x.shape) < 7 or x.shape != y.shape:
        raise ValueError('SSIM requires matching 2D images at least 7x7')
    return float(structural_similarity(y, x, data_range=255))


def metrics(image: torch.Tensor, truth: torch.Tensor, M: int, seconds: float) -> dict:
    """Offline GT metrics; never consumed by the selector. psnr is unclipped (physical);
    psnr_clamped clamps the reconstruction to [0,1] first, as ProxUnroll does."""
    mse, psnr = mse_psnr(image, truth); _, psnr_clamped = mse_psnr(image.clamp(0, 1), truth)
    return dict(M=M, mse=mse, psnr=psnr, psnr_clamped=psnr_clamped, ssim=ssim(image, truth), seconds=seconds)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'inf' if value > 0 else '-inf' if value < 0 else 'nan'
    if isinstance(value, dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def save_result(root: Path, image_id: str, plus: torch.Tensor, ids: list[str],
                history: list[dict], truth: torch.Tensor, reconstruction: torch.Tensor,
                settings: dict, runtime: dict, roi: dict | None = None,
                information_capture: dict | None = None) -> dict:
    """Publish adaptive-only .pt/.json/.png; display clipping never alters metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root.mkdir(parents=True, exist_ok=True)
    shape = settings.get('config', {}).get('image_shape') or settings.get('optics', {}).get('parameters', {}).get('height')
    if isinstance(shape, (list, tuple)):
        height, width = map(int, shape)
    else:
        height = int(shape); width = int(shape)
    if plus.ndim == 2:
        if plus.shape[-1] != height * width:
            raise ValueError(f'pattern dimension {plus.shape[-1]} does not match template {(height,width)}')
        plus = plus.detach().cpu().reshape(-1, height, width).to(torch.uint8)
    elif plus.ndim == 3:
        if tuple(plus.shape[-2:]) != (height, width): raise ValueError('pattern shape mismatch')
        plus = plus.detach().cpu().to(torch.uint8)
    else: raise ValueError('plus must be [M,d] or [M,H,W]')
    if len(ids) != len(plus) or history[-1]['M'] != len(plus):
        raise ValueError('pattern/history mismatch')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate reserved pool IDs')
    offline_pca = 'offline_pca' in settings
    display_name = 'Offline-PCA Fixed' if offline_pca else 'Adaptive'
    relative_rms=[]
    for begin in range(0,len(plus),64):
        if offline_pca:
            from .offline_pca import load_bank
            bank=load_bank(settings['offline_pca']['path'],settings['offline_pca']['sha256'])
            if not torch.equal(plus[begin:begin+64].flatten(1),bank['plus_uint8'][begin:begin+64]):
                raise ValueError('Offline PCA executed codes differ from frozen bank')
            relative_rms.extend([0.]*len(plus[begin:begin+64]))
            continue
        centers = []
        for i in ids[begin:begin+64]:
            if ':' not in i: raise ValueError(f'invalid pattern id {i}')
            centers.append(int(i.split(':',1)[1]))
        design=settings['config'].get('pattern_pool')
        center=pattern_batch(centers,height,width,'cpu',design,
                             bit_depth=settings['config'].get('dmd_bit_depth',8)).continuous
        effective=(2*plus[begin:begin+64].double().flatten(1)-255)/255
        relative_rms.extend(((effective-center).square().mean(1)/center.square().mean(1)).sqrt().tolist())
    bound=settings['config']['optimization']['rms_bound']
    if max(relative_rms) > bound+1e-12:
        raise ValueError('executed pattern exceeds original-center RMS bound')
    bit_depth=int(settings['config'].get('dmd_bit_depth',8))
    binary_codes_only=bool(torch.all((plus==0)|(plus==255)))
    if bit_depth==1 and not binary_codes_only:
        raise ValueError('1-bit DMD result contains a positive code other than 0 or 255')
    paths = {ext:root/(image_id+'.'+ext) for ext in ['pt','json','png']}
    unexpected = set(root.iterdir()) - set(paths.values())
    if unexpected: raise ValueError(f'Output directory not dedicated to this image: {unexpected}')
    atomic_save(dict(plus_uint8=plus, history=history), paths['pt'])
    final = {k:history[-1][k] for k in ['M','mse','psnr','psnr_clamped','ssim']}
    result = dict(schema='adaptive-only-three-files-v2', image_id=image_id, method='propagation-full-pseudoinverse',
                  status='complete', final=final, runtime=runtime, settings=settings,
                  evaluation_region=roi,
                  patterns=dict(count=len(plus), prefix_count=settings['config']['initial_measurements'], adaptively_selected_count=0 if offline_pca else len(plus)-settings['config']['initial_measurements'],
                                shape=list(plus.shape), dtype='uint8', ids=ids,
                                plus_hash=digest(plus), minus_definition='255 - plus_uint8',
                                id_definition=('offline_pca:i indexes the immutable training-only PCA bank; RMS is zero relative to that bank' if offline_pca else 'reserved initialization-pool IDs; optimized patterns are the saved codes, not original Walsh rows'),
                                row_relative_rms=relative_rms,max_relative_rms=max(relative_rms),
                                physical_checks=dict(complement_sum=255,uint8_min=int(plus.min()),uint8_max=int(plus.max()),
                                                     dmd_bit_depth=bit_depth,binary_codes_only=binary_codes_only,
                                                     positive_code_levels=([0,255] if bit_depth==1 else 'integer range 0..255'),
                                                     unique_reserved_ids=True,rms_bound=bound),
                                effective_row=settings['effective_row'],
                                effective_measurements=len(plus), physical_exposures=2*len(plus),
                                includes_prefix='First prefix_count rows are the executed initial prefix; no separate fixed result'),
                  pt_sha256=file_hash(paths['pt']),
                  ssim_definition='ProxUnroll: clamp recon [0,1], x255, skimage structural_similarity data_range=255 defaults (7x7 uniform, sample cov)',
                  psnr_definition='psnr unclipped physical; psnr_clamped = recon clamped to [0,1] (ProxUnroll)',
                  retention='No O/F, candidates, optimizer states, GT arrays, reconstruction arrays or other strategies saved. Simulation rerun required after interruption; no hardware resume claim.')
    if information_capture is not None:
        result['information_capture'] = information_capture
    gt_canvas = truth.detach().cpu().reshape(height,width).numpy()
    rec_canvas = reconstruction.detach().cpu().reshape(height,width).numpy()
    if roi is None:
        roi = dict(top=0,left=0,height=height,width=width,original_height=height,
                   original_width=width,canvas_height=height,canvas_width=width)
        result['evaluation_region'] = roi
    y0,x0,h,w = (int(roi[k]) for k in ('top','left','height','width'))
    gt = gt_canvas[y0:y0+h,x0:x0+w]
    rec = rec_canvas[y0:y0+h,x0:x0+w]
    fig, axes = plt.subplots(2,3,figsize=(15,9),layout='constrained')
    axes[0,0].imshow(gt,cmap='gray',vmin=0,vmax=1); axes[0,0].set_title('GT ROI (offline evaluation only)')
    axes[0,1].imshow(rec,cmap='gray',vmin=0,vmax=1)
    axes[0,1].set_title(f"{display_name} M{len(plus)}\nPSNR {final['psnr']:.4f} dB | SSIM {final['ssim']:.5f}")
    im=axes[0,2].imshow(np.abs(rec-gt),cmap='magma',vmin=0,vmax=1)
    axes[0,2].set_title('Absolute error (fixed scale 0–1)')
    fig.colorbar(im,ax=axes[0,2],label='Physical intensity')
    for ax in axes[0]: ax.set_axis_off()
    for ax,key,label in zip(axes[1],['psnr','mse','ssim'],['PSNR (dB)','MSE (intensity squared)','SSIM']):
        curve_label = 'adaptive (pinv history; final decoder at endpoint)' if settings.get('derived_from',{}).get('curve_definition') else display_name
        ax.plot([r['M'] for r in history],[r[key] for r in history],label=curve_label)
        ax.set(xlabel='Effective pattern count',ylabel=label); ax.grid(alpha=.25); ax.legend()
    fig.suptitle(f"{image_id} | S{settings['config']['initial_measurements']} / B{settings['config']['batch_size']} | {display_name} | PSNR unclipped, SSIM ProxUnroll\n"+
                 f"{settings['method']} | final {settings['config'].get('final_decoder','pinv')} | seed {settings.get('seed',20260921)}")
    fig.savefig(paths['png'],dpi=150); plt.close(fig)
    temp=paths['json'].with_suffix('.json.tmp')
    temp.write_text(json.dumps(json_safe(result),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temp,paths['json'])  # JSON complete is the commit marker.
    return result
