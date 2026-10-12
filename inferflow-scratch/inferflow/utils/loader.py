import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    按参数名把 safetensors 权重加载进模型。
    TODO(Step4):
      遍历 path 下所有 *.safetensors（safe_open(file, "pt", "cpu")），对每个 weight_name：
        - 若包含 model.packed_modules_mapping 的某个 key（如 "q_proj"），
          把名字替换为合并后的模块名（"qkv_proj"），调用 param.weight_loader(param, tensor, shard_id)
        - 否则 model.get_parameter(weight_name)（不存在的参数直接跳过），
          用 param 上挂的 weight_loader（没有则 default_weight_loader）加载
    """
    raise NotImplementedError
