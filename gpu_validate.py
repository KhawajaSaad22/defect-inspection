"""Run cached DINO features and verify CPU/CUDA scoring parity."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--allow-download',action='store_true')
    args=p.parse_args();out=Path(args.output)
    if out.exists():p.error('Choose a new report path')
    os.environ['HF_HUB_OFFLINE']='0' if args.allow_download else '1'
    import numpy as np
    import torch
    from PIL import Image
    from inspect_defects import Encoder,patch_scores,MODEL
    with tempfile.TemporaryDirectory() as folder:
        image=Path(folder)/'generated.png'
        Image.fromarray(np.random.default_rng(42).integers(0,256,(224,224,3),np.uint8)).save(image)
        encoder=Encoder(device=args.device);features=encoder.encode(image)
        assert next(encoder.model.parameters()).device.type==args.device
        assert np.isfinite(features).all()
        states=[dict(method=np.array('knn'),bank=features[:32]),
                dict(method=np.array('pca'),mean=features.mean(0),basis=np.eye(features.shape[1],dtype=np.float32)[:8])]
        deltas={}
        for state in states:
            cpu=patch_scores(features,state,'cpu');selected=patch_scores(features,state,args.device)
            np.testing.assert_allclose(cpu,selected,atol=1e-3,rtol=1e-3)
            deltas[str(state['method'])]=float(np.max(np.abs(cpu-selected)))
        report=dict(device=encoder.device,model=MODEL,model_device=str(next(encoder.model.parameters()).device),
                    torch=torch.__version__,cuda_runtime=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                    feature_shape=list(features.shape),scoring_max_abs_difference=deltas,
                    note='Real DINO features with synthetic scoring banks; not real defect accuracy. PCA fitting remains CPU.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
