
import torch
from flash_sink_attn import SinkCacheManager
from flash_sink_attn import flash_sink_attn_func
from flash_attn import flash_attn_func

from torch.nn import MultiheadAttention

num_tokens = [1024 * 2 ** i for i in range(6)]
batch_size = 1
num_kv_heads = 4
num_query_heads = 28
sink = 16
sliding = 2048
dtype = torch.float16


def test(closure, title):

    for tokens in num_tokens:
        query = torch.randn((batch_size, tokens, num_query_heads, 128), device='cuda', dtype=dtype) * 0.01
        key = torch.randn((batch_size, tokens, num_kv_heads, 128), device='cuda', dtype=dtype) * 0.01
        value = torch.randn((batch_size, tokens, num_kv_heads, 128), device='cuda', dtype=dtype) * 0.01

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        from profiler import WallTime
        time_bwd = WallTime('title', cuda=0)

        rng = torch.arange(tokens)
        mask = torch.where(rng[:, None] >= rng[None, :], 0, float('-inf')).to(dtype)
        mask = torch.where((rng[None, :] >= sink) & (rng[None, :] <= rng[:, None] - sliding), float('-inf'), mask)
        mask = mask[None, None, ...].cuda()

        for _ in range(2):
            with time_bwd:
                closure(query, key, value, mask)

        time_bwd.result(detail=True, postfix=f"-{torch.cuda.max_memory_allocated() // 1024 ** 2}")
        time_bwd.reset()

    print()


def flash_attn(query, key, value, mask):
    output = flash_attn_func(
        query,
        key,
        value,
        window_size=(sliding, 0),
        causal=True)
    output.sum().backward()


def flash_sink_attn(query, key, value, mask):
    manager = SinkCacheManager(1, num_kv_heads, 128, sink, sliding)
    manager.update(key, value)
    output = flash_sink_attn_func(query, key, value, manager)
    output.sum().backward()


def eager_attn(query, key, value, mask):
    num_groups = query.shape[2] // key.shape[2]

    key = torch.repeat_interleave(key, num_groups, dim=2)
    value = torch.repeat_interleave(value, num_groups, dim=2)

    query = query.transpose(1,2)
    key = key.transpose(1,2)
    value = value.transpose(1,2)

    attn_matrix = query @ key.transpose(2,3) / query.shape[-1] ** 0.5
    attn_matrix = torch.softmax(attn_matrix, dim=-1, dtype=torch.float32).to(query.dtype)

    return attn_matrix @ value

def flash_attn_with_sink(query, key, value, mask):
    




test(flash_attn, "flash_attn")
test(flash_sink_attn, "flash_sink_attn")
test(eager_attn, "eager_attn")