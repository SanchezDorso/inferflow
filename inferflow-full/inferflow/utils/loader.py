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
    packed_modules_mapping 把 HF 中拆开的 q/k/v、gate/up 映射到合并后的并行线性层，
    每个并行层的 weight_loader 负责只取属于当前 TP rank 的分片。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = glob(os.path.join(path, "*.safetensors"))
    assert files, f"no *.safetensors found under {path}"
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k, (v, shard_id) in packed_modules_mapping.items():
                    if k in weight_name:
                        param = model.get_parameter(weight_name.replace(k, v))
                        param.weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError:
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
