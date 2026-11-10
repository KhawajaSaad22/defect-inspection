"""Generate deterministic synthetic data for pipeline checks, not a benchmark."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def main():
    root=Path('data/demo'); rng=np.random.default_rng(17); manifest=[]
    for split,count in [('train',8),('calibration',20),('test',30)]:
        (root/split).mkdir(parents=True,exist_ok=True)
        for i in range(count):
            array=np.clip(rng.normal(125,2,(224,224,3)),0,255).astype('uint8')
            im=Image.fromarray(array); draw=ImageDraw.Draw(im)
            draw.rectangle((32,32,192,192),outline=(165,165,165),width=4)
            mask=Image.new('L',im.size,0); defect=split=='test' and i>=count//2
            if defect:
                x,y=map(int,rng.integers(50,150,2)); box=(x,y,x+28,y+12)
                draw.rectangle(box,fill=(245,245,245)); ImageDraw.Draw(mask).rectangle(box,fill=255)
            path=root/split/f'{i:03}.png'; im.save(path)
            if split=='test':
                maskpath=root/split/f'{i:03}_mask.png'; mask.save(maskpath)
                manifest.append(dict(path=str(path.relative_to(root)).replace('\\','/'),label=int(defect),mask=str(maskpath.relative_to(root)).replace('\\','/')))
    (root/'test.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Created data/demo; synthetic pipeline checks only.')
if __name__=='__main__': main()
