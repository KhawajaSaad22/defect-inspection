"""Few-shot normal-only inspection, with held-out calibration and shift evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F

MODEL = 'facebook/dinov2-small'
SHIFTS = ('clean', 'dark', 'bright', 'blur', 'contrast')


def shifted(image, shift):
    image = image.convert('RGB')
    if shift == 'dark': return ImageEnhance.Brightness(image).enhance(.65)
    if shift == 'bright': return ImageEnhance.Brightness(image).enhance(1.35)
    if shift == 'contrast': return ImageEnhance.Contrast(image).enhance(.6)
    if shift == 'blur': return image.filter(ImageFilter.GaussianBlur(1.5))
    if shift != 'clean': raise ValueError(f'Unknown shift: {shift}')
    return image


class Encoder:
    def __init__(self, backend='dino', size=224, device='auto'):
        if size < 28 or size % 14: raise ValueError('Image size must be a multiple of 14, at least 28')
        if device not in ('auto','cpu','cuda'):raise ValueError('Device must be auto, cpu or cuda')
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
        self.backend, self.size = backend, size
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if backend == 'dino':
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(MODEL).to(self.device).eval()
        elif backend != 'rgb': raise ValueError('Unknown backend')

    @torch.inference_mode()
    def encode(self, path, shift='clean'):
        with Image.open(path) as im:
            image = shifted(im, shift).resize((self.size, self.size))
            array = np.array(image, dtype=np.float32) / 255
        x = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.backend == 'dino':
            mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
            std = x.new_tensor([.229, .224, .225])[None, :, None, None]
            features = self.model(pixel_values=(x-mean)/std).last_hidden_state[:, 1:]
            features = F.normalize(features, dim=-1)[0]
        else:
            # Explicit smoke-test baseline, not a semantic substitute for DINO.
            features = F.unfold(x, kernel_size=14, stride=14).transpose(1, 2)[0]
        return features.float().cpu().numpy()


def image_files(folder):
    return sorted(p for p in Path(folder).rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'} and p.is_file())


def fingerprint(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_scores(features, state,device='cpu'):
    if device not in ('cpu','cuda'):raise ValueError('Scoring device must be cpu or cuda')
    if device=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
        query=torch.as_tensor(features,device='cuda',dtype=torch.float32)
        if str(state['method'])=='pca':
            mean=torch.as_tensor(state['mean'],device='cuda',dtype=torch.float32)
            basis=torch.as_tensor(state['basis'],device='cuda',dtype=torch.float32)
            centered=query-mean;residual=centered-(centered@basis.T)@basis
            return residual.square().mean(1).sqrt().cpu().numpy()
        reference=torch.as_tensor(state['bank'],device='cuda',dtype=torch.float32)
        return torch.cat([torch.cdist(part[None],reference[None])[0].min(1).values for part in query.split(128)]).cpu().numpy()
    if str(state['method']) == 'pca':
        centered = features - state['mean']
        residual = centered - (centered @ state['basis'].T) @ state['basis']
        return np.sqrt(np.mean(residual ** 2, axis=1))
    reference = torch.from_numpy(state['bank'])
    query = torch.from_numpy(features)
    return torch.cat([torch.cdist(part[None], reference[None])[0].min(1).values
                      for part in query.split(128)]).numpy()


def image_score(scores):
    # Fixed top-1% pooling; do not tune it on test labels.
    count = max(1, int(np.ceil(.01 * len(scores))))
    return float(np.sort(scores)[-count:].mean())


def fit(train, calibration, output, shots=4, backend='dino', method='pca', seed=42,
        size=224, fpr=.05, augment=False, max_patches=4096, device='auto'):
    if shots < 1 or not 0 < fpr < 1: raise ValueError('Require positive shots and 0 < fpr < 1')
    training, validation = image_files(train), image_files(calibration)
    if len(training) < shots or not validation: raise ValueError('Need enough normal training images and separate calibration normals')
    all_hashes = {fingerprint(p) for p in training}
    if all_hashes & {fingerprint(p) for p in validation}: raise ValueError('Train/calibration content overlap')
    rng = np.random.default_rng(seed)
    chosen = [training[i] for i in sorted(rng.choice(len(training), shots, replace=False))]
    encoder = Encoder(backend, size, device)
    variants = ('clean','dark','bright','contrast') if augment else ('clean',)
    features = np.concatenate([encoder.encode(p, s) for p in chosen for s in variants])
    if len(features) > max_patches: features = features[rng.choice(len(features), max_patches, replace=False)]
    state = dict(backend=np.array(backend), method=np.array(method), size=np.array(size),
                 train_hashes=np.array(sorted(all_hashes)), calibration_hashes=np.array([fingerprint(p) for p in validation]))
    if method == 'pca':
        mean = features.mean(0)
        _, singular, basis = np.linalg.svd(features - mean, full_matrices=False)
        energy = np.cumsum(singular ** 2) / max(float((singular ** 2).sum()), 1e-12)
        rank = min(int(np.searchsorted(energy, .95))+1, max(1, features.shape[1]-1), len(basis))
        state.update(mean=mean, basis=basis[:rank])
    elif method == 'knn': state['bank'] = features
    else: raise ValueError('Unknown method')
    calibration_maps = [patch_scores(encoder.encode(p), state,encoder.device) for p in validation]
    values = [image_score(scores) for scores in calibration_maps]
    all_calibration_scores = np.concatenate(calibration_maps)
    state['visual_low'] = np.array(np.quantile(all_calibration_scores, .95))
    state['visual_high'] = np.array(np.quantile(all_calibration_scores, .999))
    state['threshold'] = np.array(np.quantile(values, 1-fpr, method='higher'))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **state)
    metadata = dict(model=MODEL if backend=='dino' else 'RGB pixel smoke baseline', shots=shots,
                    selected=[str(p) for p in chosen], seed=seed, augment=augment,
                    target_fpr=fpr, calibration_count=len(validation), device=encoder.device)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def load(path):
    with np.load(path, allow_pickle=False) as archive: return {k: archive[k] for k in archive.files}


def predict(path, state, encoder, shift='clean'):
    scores = patch_scores(encoder.encode(path, shift), state,encoder.device)
    side = encoder.size // 14
    return image_score(scores), scores.reshape(side, side)


def overlay(path, heatmap, output, scale, low=0):
    with Image.open(path) as im: image = im.convert('RGB')
    heat = Image.fromarray(np.uint8(np.clip((heatmap-low) / max(scale-low, 1e-8), 0, 1)*255)).resize(image.size)
    red = Image.new('RGB', image.size, (255, 0, 0))
    Image.composite(Image.blend(image, red, .65), image, heat).save(output)


def evaluate(model, manifest, output, device='auto'):
    from sklearn.metrics import roc_auc_score, average_precision_score
    state = load(model); encoder = Encoder(str(state['backend']), int(state['size']), device)
    manifest = Path(manifest)
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    if not rows or set(r['label'] for r in rows) != {0,1}: raise ValueError('Test manifest needs both normal (0) and defect (1) labels')
    forbidden = set(state['train_hashes']) | set(state['calibration_hashes'])
    for row in rows:
        row['resolved'] = manifest.parent / row['path']
        if fingerprint(row['resolved']) in forbidden: raise ValueError('Test content overlaps training or calibration')
    report, predictions = [], []
    for shift in SHIFTS:
        start = time.perf_counter(); labels, scores, pixel_y, pixel_s = [], [], [], []
        for row in rows:
            score, heat = predict(row['resolved'], state, encoder, shift)
            labels.append(row['label']); scores.append(score)
            predictions.append(dict(path=row['path'], label=row['label'], shift=shift, score=score))
            if row.get('mask'):
                with Image.open(manifest.parent / row['mask']) as mask:
                    target = np.array(mask.convert('L').resize((encoder.size, encoder.size), Image.Resampling.NEAREST)) > 0
                resized = F.interpolate(torch.tensor(heat)[None,None], size=target.shape, mode='bilinear', align_corners=False)[0,0].numpy()
                pixel_y.extend(target.ravel().tolist()); pixel_s.extend(resized.ravel().tolist())
        y = np.array(labels); detected = np.array(scores) > float(state['threshold'])
        report.append(dict(shift=shift, image_auroc=float(roc_auc_score(y,scores)),
                           image_ap=float(average_precision_score(y,scores)),
                           false_positive_rate=float(detected[y==0].mean()), recall=float(detected[y==1].mean()),
                           seconds_per_image=(time.perf_counter()-start)/len(y),
                           pixel_auroc=float(roc_auc_score(pixel_y,pixel_s)) if len(set(pixel_y))==2 else None))
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(metrics=report, threshold=float(state['threshold']), predictions=predictions,device=encoder.device,scoring_device=encoder.device,
                note='Synthetic photometric shifts; not proof of real-world robustness. Pixel metrics cover only rows with masks.')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    fitp=sub.add_parser('fit')
    for name in ('train','calibration','output'): fitp.add_argument('--'+name,required=True)
    fitp.add_argument('--shots',type=int,default=4); fitp.add_argument('--seed',type=int,default=42)
    fitp.add_argument('--backend',choices=['dino','rgb'],default='dino'); fitp.add_argument('--method',choices=['pca','knn'],default='pca')
    fitp.add_argument('--augment',action='store_true'); fitp.add_argument('--size',type=int,default=224)
    fitp.add_argument('--fpr',type=float,default=.05)
    evalp=sub.add_parser('evaluate')
    for name in ('model','manifest','output'): evalp.add_argument('--'+name,required=True)
    pred=sub.add_parser('predict'); pred.add_argument('--model',required=True); pred.add_argument('--image',required=True); pred.add_argument('--output',default='overlay.png')
    for parser in (fitp,evalp,pred): parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    a=vars(p.parse_args()); command=a.pop('command')
    if command=='fit': print(json.dumps(fit(**a),indent=2))
    elif command=='evaluate': print(json.dumps(evaluate(**a),indent=2))
    else:
        state=load(a['model']); encoder=Encoder(str(state['backend']),int(state['size']),a['device'])
        score,heat=predict(a['image'],state,encoder)
        overlay(a['image'],heat,a['output'],float(state.get('visual_high',state['threshold'])),float(state.get('visual_low',0)))
        print(json.dumps(dict(score=score,threshold=float(state['threshold']),flagged=score>float(state['threshold']))))

if __name__=='__main__': main()
