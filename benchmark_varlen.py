import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from flash_sink_attn import flash_sink_attn_varlen_func, SlidingCacheManager

batch_size = 1
seqlens = [2271, 4212, 1152] # 有误差
# seqlens = [2048, 4096, 1152] # 基本没有误差
num_kv_heads = 4
num_query_heads = 28
head_dim = 128
sliding = 177
dtype = torch.float16

max_seqlen = max(seqlens)
total_tokens = sum(seqlens)

cu_seqlens = torch.tensor(np.cumsum([0] + seqlens), dtype=torch.int32, device='cuda')

query = torch.randn((total_tokens, num_query_heads, head_dim), device='cuda', dtype=dtype)
key = torch.randn((total_tokens, num_kv_heads, head_dim), device='cuda', dtype=dtype)
value = torch.randn((total_tokens, num_kv_heads, head_dim), device='cuda', dtype=dtype)
sink = torch.randn((num_query_heads,), dtype=dtype, device='cuda')

query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)
sink.requires_grad_(True)

from profiler import WallTime
ref_time_bwd = WallTime('ref-bwd', cuda=0)
our_time_bwd = WallTime('our-bwd', cuda=0)

def eager_attention_varlen_forward(
    query_unpad: torch.Tensor,
    key_unpad: torch.Tensor,
    value_unpad: torch.Tensor,
    sinks: torch.Tensor,
    cu_seqlens: torch.Tensor
):
    num_key_value_groups = query_unpad.shape[1] // key_unpad.shape[1]
    scaling = 1 / query_unpad.shape[-1] ** 0.5
    batch_size = len(cu_seqlens) - 1
    
    outputs = []
    
    for i in range(batch_size):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        seqlen = end - start
        
        q_i = query_unpad[start:end].transpose(0, 1)
        k_i = key_unpad[start:end].transpose(0, 1)
        v_i = value_unpad[start:end].transpose(0, 1)
        
        k_i = k_i.repeat_interleave(num_key_value_groups, dim=0)
        v_i = v_i.repeat_interleave(num_key_value_groups, dim=0)
        
        rng = torch.arange(seqlen, device='cuda')
        cond1 = rng[:, None] >= rng[None, :]
        cond2 = rng[None, :] > rng[:, None] - sliding if sliding is not None else True
        causal_mask = torch.where(cond1 & cond2, 0, float('-inf')).to(dtype)
        
        attn_weights = torch.matmul(q_i, k_i.transpose(1, 2)) * scaling
        attn_weights += causal_mask[None, :, :]
        
        sinks_i = sinks.reshape(-1, 1, 1).expand(-1, seqlen, 1)
        
        combined_logits = torch.cat([attn_weights, sinks_i], dim=-1)
        
        probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32).to(dtype)
        scores = probs[..., :-1]
        
        attn_output_i = torch.matmul(scores, v_i)
        outputs.append(attn_output_i.transpose(0,1))
        
    return torch.cat(outputs, dim=0)


for _ in range(20):
    query.grad, key.grad, value.grad, sink.grad = None, None, None, None

    ref_bwd = eager_attention_varlen_forward(query, key, value, sink, cu_seqlens)

    with ref_time_bwd:
        ref_bwd.sum().backward()

    ref_grad_q = query.grad.clone()
    ref_grad_k = key.grad.clone()
    ref_grad_v = value.grad.clone()
    ref_grad_s = sink.grad.clone()

    query.grad, key.grad, value.grad, sink.grad = None, None, None, None

    with our_time_bwd:
        manager = SlidingCacheManager(sliding)
        manager.update(key, value)
        our_bwd = flash_sink_attn_varlen_func(
            query, key, value, sink, cu_seqlens, manager, True)
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

import matplotlib.pyplot as plt
e = (ref_grad_q - our_grad_q).detach().abs().flatten(1).float().cpu()

ref_time_bwd.result(detail=True)
our_time_bwd.result(detail=True)

import IPython
IPython.embed(header='debug')