"""Sequential frozen OOD experiment queue; stop on any failed image/config.

Run with PIE2E Python and CUDA_VISIBLE_DEVICES. Check phase finishes before test.
No model selection, data selection, or parameter tuning based on results.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from daps_spi.runner import load_recipe,resolve_template,apply_overrides,ABLATIONS,validate_recipe

def main():
    out=ROOT/'outputs/ablation/mismatched_prior_o32_o33';out.mkdir(parents=True,exist_ok=True)
    jobs=[f'abl_ood_{d}_{p}_{m}_o{n}' for p in ['check','test'] for d in ['celeba','dtd'] for m,n in [('bias_spread',32),('spread_only',33)]]
    protocol={'jobs':jobs,'created_unix':time.time(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'comparison':'Bias+Spread O32 vs Spread-only O33; unequal sample count explicitly reported'+' DiffPIR50 F32; centred PCA vs residual second moment, spread=.25; no V/Adam','resolved':{},'assets':{}}
    for name in jobs:
        config=ROOT/'configs'/f'{name}.yaml'
        subprocess.run([sys.executable,'scripts/run.py','--config',str(config),'--check-config'],cwd=ROOT,check=True)
        r=load_recipe(config);run=r.pop('run');r=apply_overrides(resolve_template(r,run['template']),ABLATIONS[run['ablation']]+r.get('overrides',[]));validate_recipe(r)
        protocol['resolved'][name]=r
        for key in ['prior','final_prior','mean']:
            f=ROOT/r[key];protocol['assets'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    f=ROOT/'data/ood_mismatched_prior/manifest.json';protocol['manifest_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'experiment_protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    for name in jobs:
        status=dict(job=name,state='running',time=time.time(),pid=os.getpid())
        (out/'status.json').write_text(json.dumps(status,indent=2))
        print('START',name,flush=True)
        with (out/f'{name}.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u','scripts/run.py','--config',f'configs/{name}.yaml'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='failed',returncode=result.returncode);(out/'status.json').write_text(json.dumps(status,indent=2));raise SystemExit(result.returncode)
        print('DONE',name,flush=True)
    (out/'status.json').write_text(json.dumps(dict(state='complete',time=time.time()),indent=2))
if __name__=='__main__': main()
