
import torch
from flash_sink_attn import SinkCacheManager
from flash_sink_attn import flash_sink_attn_func

num_tokens = 32123
num_kv_heads = 4
num_query_heads = 28
sink = 64
sliding = 2048
dtype = torch.bfloat16

query = torch.randn((1, num_tokens, num_query_heads, 128), device='cuda', dtype=dtype) * 0.01
key = torch.randn((1, num_tokens, num_kv_heads, 128), device='cuda', dtype=dtype) * 0.01
value = torch.randn((1, num_tokens, num_kv_heads, 128), device='cuda', dtype=dtype) * 0.01

manager = SinkCacheManager(1, num_kv_heads, 128, sink, sliding)

query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)

from profiler import WallTime
ref_time_bwd = WallTime('ref-bwd', cuda=0)
our_time_bwd = WallTime('our-bwd', cuda=0)

rng = torch.arange(num_tokens)
mask = torch.where(rng[:, None] >= rng[None, :], 0, float('-inf')).to(dtype)
mask = torch.where((rng[None, :] >= sink) & (rng[None, :] <= rng[:, None] - sliding), float('-inf'), mask)
mask = mask[None, None, ...].cuda()

for _ in range(20):

    query.grad = None
    key.grad = None
    value.grad = None

    ref_bwd = torch.nn.functional.scaled_dot_product_attention(
        query=query.transpose(1,2),
        key=key.transpose(1,2).repeat_interleave(num_query_heads // num_kv_heads, dim=1),
        value=value.transpose(1,2).repeat_interleave(num_query_heads // num_kv_heads, dim=1),
        attn_mask=mask).transpose(1,2)

    with ref_time_bwd:
        ref_bwd.sum().backward()
        
    ref_grad_q = query.grad.clone()
    ref_grad_k = key.grad.clone()
    ref_grad_v = value.grad.clone()

    query.grad = None
    key.grad = None
    value.grad = None

    manager.reset()
    manager.update(key, value)
    our_bwd = flash_sink_attn_func(query, key, value, manager)

    with our_time_bwd:
        our_bwd.sum().backward()

    our_grad_q = query.grad.clone()
    our_grad_k = key.grad.clone()
    our_grad_v = value.grad.clone()

print(torch.dist(ref_bwd, our_bwd))
print(torch.dist(ref_grad_q, our_grad_q))
print(torch.dist(ref_grad_k, our_grad_k))
print(torch.dist(ref_grad_v, our_grad_v))

ref_time_bwd.result(detail=True)
our_time_bwd.result(detail=True)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(121)
plt.imshow(ref_grad_v[0,:128,0].float().cpu().numpy())
plt.subplot(122)
plt.imshow(our_grad_v[0,:128,0].float().cpu().numpy())
plt.savefig("value_grad_compare.jpg", dpi=640)
import IPython
IPython.embed(header='debug')