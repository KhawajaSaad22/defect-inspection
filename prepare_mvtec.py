"""Create disjoint manifests from a locally downloaded MVTec AD category."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from inspect_defects import image_files

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('category',type=Path); p.add_argument('--output',type=Path,default=Path('data/mvtec'))
    p.add_argument('--seed',type=int,default=42); p.add_argument('--calibration-count',type=int,default=20)
    a=p.parse_args(); normal=image_files(a.category/'train'/'good')
    if a.output.exists(): p.error('Output already exists; use a fresh output folder to prevent mixing splits.')
    if len(normal)<a.calibration_count+4: p.error('Need calibration-count plus at least four normal training images.')
    rng=np.random.default_rng(a.seed); rng.shuffle(normal)
    for split,paths in [('calibration',normal[:a.calibration_count]),('train',normal[a.calibration_count:])]:
        (a.output/split).mkdir(parents=True)
        for path in paths: shutil.copyfile(path,a.output/split/path.name)
    rows=[]
    for path in image_files(a.category/'test'):
        kind=path.parent.name; relative=Path('test')/kind/path.name
        (a.output/relative).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,a.output/relative)
        row=dict(path=relative.as_posix(),label=int(kind!='good'))
        mask=a.category/'ground_truth'/kind/(path.stem+'_mask.png')
        if mask.exists():
            dest=Path('masks')/kind/mask.name; (a.output/dest).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(mask,a.output/dest); row['mask']=dest.as_posix()
        rows.append(row)
    (a.output/'test.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(f'Prepared {len(rows)} test images; original dataset is unchanged.')
if __name__=='__main__': main()
