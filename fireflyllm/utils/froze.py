# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/1/3 16:51
# @File: froze
# @Email: mlshenkai@163.com
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Union


def freeze_non_embeds_parameters(model: Union[nn.Module, PreTrainedModel]) -> None:
    """
    freeze all parameters except embeddings
    :param model:
    :return:
    """
    for name, params in model.named_parameters():
        if "embed_tokens" not in name and "lm_head" not in name:
            params.requires_grad = False
        else:
            params.requires_grad = True


def unfreeze_parameters(model: Union[nn.Module, PreTrainedModel]) -> None:
    for name, params in model.named_parameters():
        params.requires_grad = True

