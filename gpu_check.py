"""Fail explicitly unless the container can perform a CUDA computation."""
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: use CUDA-enabled PyTorch and GPU passthrough')
x=torch.ones(1024,device='cuda')
assert x.sum().item()==1024
torch.cuda.synchronize()
print(json.dumps({'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
                  'cuda_runtime':torch.version.cuda,'compute_smoke':'passed'}))
