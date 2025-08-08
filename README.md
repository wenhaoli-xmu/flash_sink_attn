# 安装
```
git clone https://github.com/wenhaoli-xmu/flash_sink_attn.git
cd flash_sink_attn
pip install .
```

Tips: 最好安装最新的triton 3.4.0版本（如果能安装的上）

```
# [可选]安装profiler（运行benchmark.py需要）
git clone https://github.com/wenhaoli-xmu/lm-profiler.git
cd profiler
pip install -e .
```

# 调参数
```
对于H20/H100，可以打开 flash_sink_attn/flash_sink_attn.py，里面有两个global变量：

BLOCK_M = 64
BLOCK_N = 64

可以尝试将他们增大到128
```

# 评估速度
```
python benchmark.py

脚本中的如下参数可以修改：
num_tokens = 4224
num_kv_heads = 4
num_query_heads = 28
sliding = 256
dtype = torch.float16
```

# 使用方法
```python
from .flash_sink_attn import flash_sink_attn_func
from .sink_cache import SinkCacheManager

from types import MethodType
from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

def _forward_gpt_oss(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        raise NotImplementedError(f"Inference mode is not implemented, please switch to eager attention.")
        # cache_kwargs = {"cache_position": cache_position}
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # ================================================
    # NOTE: 最关键的代码
    # 算子仅支持训练，不支持推理，推理请转换为eager attention
    manager = SlidingCacheManager(
        self.sliding_window)
    manager.update(key_states, value_states)
    attn_output = flash_sink_attn_func(
        query_states,
        key_states,
        value_states,
        self.sinks,
        manager)
    # ================================================

    attn_output = self.o_proj(attn_output)
    return attn_output, None


def replace_gpt_oss_with_flash_sink_attn(model):
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(_forward_gpt_oss, layer.self_attn)
```