
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_sink_attn import SlidingCacheManager
from flash_sink_attn import flash_sink_attn_func

num_tokens = 4224
num_kv_heads = 4
num_query_heads = 28
sliding = 256
dtype = torch.float16

query = torch.randn((1, num_tokens, num_query_heads, 128), device='cuda', dtype=dtype)
key = torch.randn((1, num_tokens, num_kv_heads, 128), device='cuda', dtype=dtype)
value = torch.randn((1, num_tokens, num_kv_heads, 128), device='cuda', dtype=dtype)
sink = torch.randn((num_query_heads,), dtype=dtype, device='cuda')


query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)
sink.requires_grad_(True)

from profiler import WallTime
ref_time_bwd = WallTime('ref-bwd', cuda=0)
our_time_bwd = WallTime('our-bwd', cuda=0)


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
):
    num_key_value_groups = query.shape[2] // key.shape[2]
    num_tokens = key.shape[1]
    scaling = 1 / query.shape[-1] ** 0.5

    rng = torch.arange(num_tokens)

    cond1 = rng[:, None] >= rng[None, :]
    cond2 = rng[None, :] > rng[:, None] - sliding if sliding is not None else True
    causal_mask = torch.where(cond1 & cond2, 0, float('-inf')).to(dtype).cuda()

    query_states = query.transpose(1,2)
    key_states = key.transpose(1,2).repeat_interleave(num_key_value_groups, dim=1)
    value_states = value.transpose(1,2).repeat_interleave(num_key_value_groups, dim=1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + causal_mask[None, None, :, :]

    sinks = sinks.reshape(1, -1, 1, 1).expand(query_states.shape[0], -1, query_states.shape[-2], -1)

    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    attn_output = torch.matmul(scores, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


for _ in range(20):

    query.grad = None
    key.grad = None
    value.grad = None
    sink.grad = None

    ref_bwd = eager_attention_forward(
        query,
        key,
        value,
        sink)

    with ref_time_bwd:
        ref_bwd.sum().backward()
        
    ref_grad_q = query.grad.clone()
    ref_grad_k = key.grad.clone()
    ref_grad_v = value.grad.clone()
    ref_grad_s = sink.grad.clone()

    query.grad = None
    key.grad = None
    value.grad = None
    sink.grad = None

    manager = SlidingCacheManager(sliding)
    manager.update(key, value)
    our_bwd = flash_sink_attn_func(query, key, value, sink, manager)

    with our_time_bwd:
        our_bwd.sum().backward()

    our_grad_q = query.grad.clone()
    our_grad_k = key.grad.clone()
    our_grad_v = value.grad.clone()
    our_grad_s = sink.grad.clone()

print(torch.dist(ref_bwd, our_bwd))
print(torch.dist(ref_grad_q, our_grad_q))
print(torch.dist(ref_grad_k, our_grad_k))
print(torch.dist(ref_grad_v, our_grad_v))
print(torch.dist(ref_grad_s, our_grad_s))

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