"""Read-only dataset audit with explicit groups and optional near-duplicate checks."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_manifest(path,near_distance=4):
    path=Path(path)
    records=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records,list) or not records: raise ValueError('Expected a nonempty record list')
    if not -1<=near_distance<=64: raise ValueError('Near distance must be -1 (disabled) or 0..64')
    if len(records)>2000 and near_distance>=0:
        raise ValueError('Near-duplicate pair scan is limited to 2000 samples; use --near-distance -1 for a larger exact audit')
    issues=[]; ids=set(); groups=defaultdict(list); hashes=defaultdict(list); fingerprints=[]
    splits=Counter(); labels=Counter()
    for row in records:
        identity=row.get('id'); split=row.get('split'); group=row.get('group')
        if not isinstance(identity,str) or not identity or identity in ids:
            issues.append(dict(kind='invalid_or_duplicate_id',id=identity))
        ids.add(str(identity))
        if split not in ('train','validation','calibration','test','pool'):
            issues.append(dict(kind='invalid_split',id=identity,split=split))
        splits[str(split)]+=1
        if not isinstance(group,str) or not group:
            issues.append(dict(kind='missing_group',id=identity))
        else: groups[group].append((identity,split))
        if 'label' in row: labels[str(row['label'])]+=1
        if not row.get('image'):
            issues.append(dict(kind='missing_image',id=identity)); continue
        image_path=path.parent/row['image']
        try:
            hashes[digest(image_path)].append((identity,split))
            with Image.open(image_path) as image:
                size=image.size
                pixels=np.asarray(image.convert('L').resize((9,8)),dtype=np.int16)
                fingerprint=(pixels[:,1:]>pixels[:,:-1]).ravel()
            fingerprints.append((identity,split,fingerprint))
            if row.get('mask'):
                with Image.open(path.parent/row['mask']) as mask:
                    if mask.size!=size: issues.append(dict(kind='mask_size_mismatch',id=identity))
                    if np.asarray(mask).ndim!=2: issues.append(dict(kind='multichannel_mask',id=identity))
            elif split!='pool' and row.get('label')!=0:
                issues.append(dict(kind='missing_mask_or_normal_label',id=identity))
        except (OSError,ValueError) as exc:
            issues.append(dict(kind='unreadable_image_or_mask',id=identity,error=str(exc)))
    for group,entries in groups.items():
        if len({split for _,split in entries})>1:
            issues.append(dict(kind='group_crosses_splits',group=group,samples=entries))
    for entries in hashes.values():
        if len(entries)>1:
            issues.append(dict(kind='exact_duplicate',samples=entries,cross_split=len({s for _,s in entries})>1))
    if near_distance>=0:
        for i,(identity,split,fp) in enumerate(fingerprints):
            for other,other_split,other_fp in fingerprints[i+1:]:
                if split==other_split: continue
                distance=int(np.count_nonzero(fp!=other_fp))
                if distance<=near_distance:
                    issues.append(dict(kind='near_duplicate_candidate',samples=[identity,other],distance=distance))
    blocking=[issue for issue in issues if issue['kind']!='near_duplicate_candidate']
    return dict(schema_version=1,manifest_sha256=digest(path),samples=len(records),
                splits=dict(splits),class_counts=dict(labels),issues=issues,blocking_issues=len(blocking),
                note='Near-duplicate candidates need human review; dHash can confuse uniform surfaces. Groups must identify related objects/capture sessions. No files were changed.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('--output',required=True)
    parser.add_argument('--near-distance',type=int,default=4)
    args=parser.parse_args(); output=Path(args.output)
    if output.exists(): parser.error('Choose a new report path')
    result=audit_manifest(args.manifest,args.near_distance)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if result['blocking_issues']: parser.exit(2,'Dataset audit found issues; inspect the report\n')


if __name__=='__main__': main()
