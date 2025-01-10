# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/1/3 18:43
# @File: utils
# @Email: mlshenkai@163.com
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Plugin
import logging


@torch.no_grad()
def all_reduce_mean(tensor: torch.Tensor, plugin: Plugin = None) -> torch.Tensor:
    if plugin is not None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=plugin.dp_group)
        tensor.div_(plugin.dp_size)
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())

    return tensor


def get_model_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            logger.log(*args, **kwargs)
    else:
        logger.log(*args, **kwargs)
