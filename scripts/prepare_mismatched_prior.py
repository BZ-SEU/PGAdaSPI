"""Freeze OOD lists and RGB-to-Y256 images, without observing method performance."""
from pathlib import Path
import csv, hashlib, json, random
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/mnt/polar/dataset')
SEED=20261106

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    target=ROOT/'data/ood_mismatched_prior'
    if (target/'manifest.json').exists():
        raise SystemExit('Manifest already frozen; refusing to resample or overwrite')
    rng=random.Random(SEED)
    with (SOURCE/'CelebA/list_eval_partition.csv').open() as f:
        celeba=list(csv.DictReader(f))
    groups={}
    for phase,partition,n in [('check','1',4),('test','2',20)]:
        names=sorted(r['image_id'] for r in celeba if r['partition']==partition)
        groups['celeba',phase]=[(SOURCE/'CelebA/img_align_celeba/img_align_celeba'/s,Path(s).stem,None) for s in rng.sample(names,n)]
    for phase,split,n in [('check','val1',4),('test','test1',20)]:
        lines=(SOURCE/'DTD/labels'/f'{split}.txt').read_text().splitlines()
        classes=sorted(set(s.split('/')[0] for s in lines))
        chosen=rng.sample(classes,n)
        groups['dtd',phase]=[]
        for c in chosen:
            s=rng.choice(sorted(s for s in lines if s.startswith(c+'/')))
            groups['dtd',phase].append((SOURCE/'DTD/images'/s,Path(s).stem,c))
    records=[]
    for (domain,phase),items in groups.items():
        dest=target/domain/phase;dest.mkdir(parents=True,exist_ok=True)
        for src,id,category in items:
            with Image.open(src) as im:
                im=im.convert('RGB');w,h=im.size;side=min(w,h)
                box=((w-side)//2,(h-side)//2,(w-side)//2+side,(h-side)//2+side)
                rgb=np.asarray(im.crop(box).resize((256,256),Image.Resampling.LANCZOS),dtype=np.float64)
            # ITU-R BT.601 limited-range Y, explicit coefficients and uint8 rounding.
            y=np.rint(16+rgb@np.array([65.481,128.553,24.966])/255).clip(0,255).astype('uint8')
            out=dest/f'{id}.png';Image.fromarray(y).save(out)
            records.append(dict(domain=domain,phase=phase,image_id=id,category=category,source=str(src),source_sha256=sha(src),original_wh=[w,h],crop_box=box,path=str(out.relative_to(ROOT)),sha256=sha(out)))
    manifest=dict(seed=SEED,selection='CelebA official val/test; DTD split1 val/test, one image per uniformly sampled class; no performance selection',preprocessing='RGB center square crop, PIL LANCZOS resize 256, round(16 + [65.481,128.553,24.966] dot RGB /255), uint8 Y; runner divides by255',limitations=['CelebA is upsampled','No cross-dataset near-duplicate audit performed'],records=records)
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    # Experiment YAMLs are versioned separately; preparation only freezes data.
    print(json.dumps(dict(images=len(records),manifest=str(target/'manifest.json'))))
if __name__=='__main__': main()
