import unittest
import tempfile
from pathlib import Path
import numpy as np
from inspect_defects import patch_scores,image_score,shifted,Encoder,fit,load,predict,evaluate
from PIL import Image

class Tests(unittest.TestCase):
    def test_saved_model_validation_and_legacy_support(self):
        from inspect_defects import validate_model
        state=dict(backend=np.array('rgb'),method=np.array('knn'),size=np.array(28),threshold=np.array(1.),
                   train_hashes=np.array(['a'*64]),calibration_hashes=np.array(['b'*64]),bank=np.ones((2,588),dtype='float32'))
        self.assertIs(validate_model(state),state)
        for change in ({'threshold':np.array(float('nan'))},{'bank':np.ones((2,12))},{'size':np.array(29)},
                       {'format_version':np.array(2)},{'model_id':np.array('different')},{'method':np.array('unknown')}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_model(dict(state,**change))
    def test_incompatible_encoder_is_rejected_before_encoding(self):
        from types import SimpleNamespace
        encoder=SimpleNamespace(backend='rgb',size=28)
        with self.assertRaisesRegex(ValueError,'does not match'):
            predict('unused',dict(backend=np.array('dino'),size=np.array(224)),encoder)
    def test_scoring_device_validation(self):
        with self.assertRaisesRegex(ValueError,'Scoring device'):
            patch_scores(np.zeros((1,2)),{},device='invalid')

    def test_cuda_scoring_matches_cpu(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        rng=np.random.default_rng(4)
        query=rng.normal(size=(257,8)).astype('float32')
        for state in ({'method':'knn','bank':rng.normal(size=(31,8)).astype('float32')},
                      {'method':'pca','mean':np.zeros(8,dtype='float32'),'basis':np.eye(8,dtype='float32')[:3]}):
            np.testing.assert_allclose(patch_scores(query,state,'cuda'),patch_scores(query,state,'cpu'),atol=1e-4,rtol=1e-4)

    def test_residual(self):
        state={'method':'pca','mean':np.zeros(2),'basis':np.array([[1.,0.]])}
        scores=patch_scores(np.array([[3.,0.],[0.,4.]]),state)
        self.assertEqual(scores[0],0); self.assertGreater(scores[1],2)
    def test_knn(self):
        scores=patch_scores(np.array([[0.,0.],[3.,4.]],dtype='float32'),{'method':'knn','bank':np.array([[0.,0.]],dtype='float32')})
        np.testing.assert_allclose(scores,[0,5])
    def test_pooling(self): self.assertEqual(image_score(np.arange(100)),99)
    def test_shift(self):
        im=Image.new('RGB',(28,28),(100,100,100))
        self.assertLess(np.array(shifted(im,'dark')).mean(),100)
    def test_size(self):
        with self.assertRaises(ValueError): Encoder('rgb',225)

    def test_calibration_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'train').mkdir(); (root/'cal').mkdir()
            image=Image.new('RGB',(28,28),(100,100,100))
            image.save(root/'train'/'a.png'); image.save(root/'cal'/'b.png')
            with self.assertRaisesRegex(ValueError,'overlap'):
                fit(root/'train',root/'cal',root/'model.npz',shots=1,backend='rgb',size=28)

    def test_persistence_and_test_leakage(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'train').mkdir(); (root/'cal').mkdir()
            rng=np.random.default_rng(5)
            for split in ('train','cal'):
                Image.fromarray(rng.integers(0,255,(28,28,3),dtype='uint8')).save(root/split/'a.png')
            model=root/'model.npz'
            fit(root/'train',root/'cal',model,shots=1,backend='rgb',size=28,method='knn')
            state=load(model); score,heat=predict(root/'cal'/'a.png',state,Encoder('rgb',28))
            self.assertEqual(heat.shape,(2,2)); self.assertTrue(np.isfinite(score))
            manifest=root/'test.json'
            manifest.write_text(json.dumps([{'path':'train/a.png','label':0},{'path':'cal/a.png','label':1}]))
            with self.assertRaisesRegex(ValueError,'overlaps'):
                evaluate(model,manifest,root/'report.json')

if __name__=='__main__': unittest.main()
