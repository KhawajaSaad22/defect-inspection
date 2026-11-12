import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from dataset_audit import audit_manifest,digest


class AuditTests(unittest.TestCase):
    def test_seeded_errors_and_immutable_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); Image.new('RGB',(8,8),'red').save(root/'a.png')
            Image.new('L',(4,4),255).save(root/'mask.png')
            manifest=root/'data.json'
            manifest.write_text(json.dumps([
                dict(id='a',image='a.png',split='train',group='object-1',label=0),
                dict(id='b',image='a.png',split='test',group='object-1',mask='mask.png',label=1)]))
            before=digest(manifest); result=audit_manifest(manifest)
            kinds={issue['kind'] for issue in result['issues']}
            self.assertTrue({'exact_duplicate','group_crosses_splits','mask_size_mismatch'}<=kinds)
            self.assertEqual(digest(manifest),before)
            self.assertEqual(result['class_counts'],{'0':1,'1':1})

    def test_valid_normal_and_pool_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            Image.new('RGB',(8,8),'red').save(root/'a.png')
            Image.new('RGB',(8,8),'green').save(root/'b.png')
            path=root/'data.json'; path.write_text(json.dumps([
                dict(id='a',image='a.png',split='train',group='a',label=0),
                dict(id='b',image='b.png',split='pool',group='b')]))
            self.assertEqual(audit_manifest(path,near_distance=-1)['blocking_issues'],0)
