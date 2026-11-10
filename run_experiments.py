"""Run paired few-shot ablations with fixed splits and separate output artifacts."""
import argparse
from pathlib import Path
import json
from inspect_defects import fit,evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('data/demo'))
    p.add_argument('--backend',choices=['dino','rgb'],default='dino')
    p.add_argument('--seeds',type=int,nargs='+',default=[17,42,73]); p.add_argument('--shots',type=int,nargs='+',default=[1,4])
    p.add_argument('--output',type=Path,default=Path('experiments')); a=p.parse_args()
    if a.output.exists(): p.error('Choose a fresh output folder; experiments do not overwrite previous runs.')
    a.output.mkdir(parents=True); summary=[]
    for shots in a.shots:
        for seed in a.seeds:
            for method,augment in [('knn',False),('pca',False),('pca',True)]:
                name=f'{a.backend}-{method}-aug{int(augment)}-k{shots}-s{seed}'
                model=a.output/(name+'.npz')
                fit(a.data/'train',a.data/'calibration',model,shots=shots,backend=a.backend,method=method,seed=seed,augment=augment)
                metrics=evaluate(model,a.data/'test.json',a.output/(name+'-metrics.json'))
                summary.append(dict(run=name,metrics=metrics))
                (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Finished paired runs. Interpret independent seeds at the image-reference level, not as independent test datasets.')
if __name__=='__main__': main()
